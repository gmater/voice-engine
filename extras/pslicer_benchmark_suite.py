"""
Ultimate multi-speaker benchmark: ≥10 distinct Edge voices, overlapping speech,
ground-truth timestamps → pslicer (WhisperX + diarize + smooth) → quantitative compare.

Outputs (under --workdir, default ./pslicer_benchmark_run):
  benchmark_mix.wav       — mixed timeline
  benchmark_truth.json    — per-utterance t0/t1, voice ids, overlap graph
  out/                    — pslicer WAV clips + pslicer_manifest.json
  benchmark_report.json   — IoU, speaker mapping, overlap caveats

Usage (from ``voice_engine`` repo root):

  set HUGGING_FACE_HUB_TOKEN=...
  venv\\Scripts\\python.exe extras\\pslicer_benchmark_suite.py --model base -v

  venv\\Scripts\\python.exe extras\\pslicer_benchmark_suite.py --synth-only
  venv\\Scripts\\python.exe extras\\pslicer_benchmark_suite.py --compare-only --workdir pslicer_benchmark_run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pslicer

try:
    import librosa
except ImportError:
    librosa = None

# Twelve distinct US English neural voices (≥10 required; names must match `edge-tts --list-voices`).
BENCH_VOICES: list[tuple[str, str]] = [
    ("v00", "en-US-JennyNeural"),
    ("v01", "en-US-GuyNeural"),
    ("v02", "en-US-AriaNeural"),
    ("v03", "en-US-BrianNeural"),
    ("v04", "en-US-EmmaNeural"),
    ("v05", "en-US-AnaNeural"),
    ("v06", "en-US-AndrewNeural"),
    ("v07", "en-US-MichelleNeural"),
    ("v08", "en-US-EricNeural"),
    ("v09", "en-US-ChristopherNeural"),
    ("v10", "en-US-RogerNeural"),
    ("v11", "en-US-SteffanNeural"),
]


@dataclass
class ScheduledUtterance:
    utterance_id: str
    voice_id: str
    edge_voice: str
    text: str
    start_sec: float
    t0: float
    t1: float
    bench_segment: str  # "solo" | "overlap_stress"


def _intersect_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = _intersect_len(a0, a1, b0, b1)
    if inter <= 0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def _build_schedule() -> list[tuple[int, str, float]]:
    """
    Twelve distinct voices: staggered solos, then two **dyadic** crosstalk scenes
    (only two speakers overlapping at a time) so overlap-stress IoU can stay meaningful
    while still exercising real overlap. A prior six-voice pile-up drove IoU to ~0.
    """
    out: list[tuple[int, str, float]] = []
    gap = 3.05  # seconds between solo starts (limits accidental bleed between voices)
    for i in range(12):
        out.append(
            (
                i,
                f"Solo segment for speaker {i} in the twelve voice benchmark.",
                round(i * gap, 2),
            )
        )
    t_a = round(12 * gap + 1.2, 2)
    # Scene A: voices 4 and 8; second starts ~0.35s later (substantial overlap, only 2 streams)
    out.extend(
        [
            (4, "Crosstalk scene A, speaker four leads this line.", t_a),
            (8, "Crosstalk scene A, speaker eight talks over four.", round(t_a + 0.35, 2)),
        ]
    )
    # Scene B: separate in time so diarization / ASR can re-lock; voices 2 and 11
    t_b = round(t_a + 5.8, 2)
    out.extend(
        [
            (2, "Crosstalk scene B, speaker two leads this line.", t_b),
            (11, "Crosstalk scene B, speaker eleven talks over two.", round(t_b + 0.32, 2)),
        ]
    )
    return out


_SCHEDULE_TEMPLATE: list[tuple[int, str, float]] = _build_schedule()


async def _tts_to_numpy(text: str, voice: str, sr: int, tmp_mp3: Path) -> np.ndarray:
    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(tmp_mp3))
    y, _ = librosa.load(str(tmp_mp3), sr=sr, mono=True)
    y = y.astype(np.float32)
    mx = float(np.max(np.abs(y)) or 1.0)
    y = (y / mx).astype(np.float32)
    return y


async def synthesize_benchmark(
    workdir: Path,
    sr: int = 16000,
) -> tuple[Path, list[ScheduledUtterance]]:
    if librosa is None:
        raise SystemExit("librosa required")
    workdir.mkdir(parents=True, exist_ok=True)
    tmpdir = workdir / "_tts_tmp"
    tmpdir.mkdir(exist_ok=True)

    # Pre-generate all TTS in parallel (grouped batches to avoid hammering Edge).
    payloads: list[tuple[str, str, str, float, Path]] = []
    for i, (vix, text, start_sec) in enumerate(_SCHEDULE_TEMPLATE):
        vid, edge = BENCH_VOICES[vix]
        uid = f"u{i:02d}"
        mp3 = tmpdir / f"{uid}.mp3"
        payloads.append((uid, vid, edge, start_sec, mp3))

    sem = asyncio.Semaphore(6)

    async def run_one(idx: int, uid: str, vid: str, edge: str, start_sec: float, mp3: Path):
        async with sem:
            text = _SCHEDULE_TEMPLATE[idx][1]
            y = await _tts_to_numpy(text, edge, sr, mp3)
        return uid, vid, edge, start_sec, text, y

    results = await asyncio.gather(
        *[
            run_one(i, uid, vid, edge, start_sec, mp3)
            for i, (uid, vid, edge, start_sec, mp3) in enumerate(payloads)
        ]
    )

    # Mix down (preserve schedule index for solo vs overlap_stress labeling)
    pieces: list[tuple[float, np.ndarray, str, str, str, str, int]] = []
    max_end = 0.0
    for sched_idx, (uid, vid, edge, start_sec, text, y) in enumerate(results):
        t0 = start_sec
        t1 = start_sec + len(y) / sr
        max_end = max(max_end, t1)
        pieces.append((t0, y, uid, vid, edge, text, sched_idx))

    pad = int(0.8 * sr)
    mix = np.zeros(int(max_end * sr) + pad, dtype=np.float32)
    utterances: list[ScheduledUtterance] = []

    for t0, y, uid, vid, edge, text, sched_idx in sorted(pieces, key=lambda x: x[0]):
        i0 = int(t0 * sr)
        i1 = i0 + len(y)
        mix[i0:i1] += y.astype(np.float32) * 0.38
        t1 = t0 + len(y) / sr
        seg = "solo" if sched_idx < 12 else "overlap_stress"
        utterances.append(
            ScheduledUtterance(
                uid, vid, edge, text, t0, round(t0, 4), round(t1, 4), seg
            )
        )

    peak = float(np.max(np.abs(mix)) or 1.0)
    mix *= 0.94 / peak

    wav_path = workdir / "benchmark_mix.wav"
    sf.write(str(wav_path), mix, sr, subtype="PCM_16")

    # Overlap graph
    ov: dict[str, list[str]] = {u.utterance_id: [] for u in utterances}
    for i, a in enumerate(utterances):
        for b in utterances[i + 1 :]:
            if _intersect_len(a.t0, a.t1, b.t0, b.t1) > 0.02:
                ov[a.utterance_id].append(b.utterance_id)
                ov[b.utterance_id].append(a.utterance_id)

    truth = {
        "sample_rate": sr,
        "duration_sec": round(len(mix) / sr, 3),
        "voice_table": [{"voice_id": v, "edge_voice": e} for v, e in BENCH_VOICES],
        "utterances": [asdict(u) for u in utterances],
        "overlap_adjacency": ov,
        "notes": "Overlapping regions are hard for ASR/diarization; expect lower IoU and speaker swaps there.",
    }
    (workdir / "benchmark_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"Wrote {wav_path} ({truth['duration_sec']}s), {len(utterances)} utterances, truth JSON.")
    return wav_path, utterances


def _greedy_speaker_map(
    truth_utts: list[dict],
    pred_chunks: list[dict],
) -> dict[str, str]:
    """Map pyannote SPEAKER_xx -> our voice_id (v00..) by total overlap duration."""
    t_voices = sorted({u["voice_id"] for u in truth_utts})
    p_spk = sorted({c["speaker"] for c in pred_chunks})
    mat: dict[tuple[str, str], float] = {}
    for u in truth_utts:
        for c in pred_chunks:
            d = _intersect_len(
                float(u["t0"]),
                float(u["t1"]),
                float(c["t0_sec"]),
                float(c["t1_sec"]),
            )
            if d > 0:
                key = (u["voice_id"], c["speaker"])
                mat[key] = mat.get(key, 0.0) + d

    mapping: dict[str, str] = {}
    used_t: set[str] = set()
    used_p: set[str] = set()
    pairs = sorted(mat.items(), key=lambda x: x[1], reverse=True)
    for (tv, ps), _w in pairs:
        if tv in used_t or ps in used_p:
            continue
        mapping[ps] = tv
        used_t.add(tv)
        used_p.add(ps)
    # Unmapped preds keep identity
    for ps in p_spk:
        mapping.setdefault(ps, "?")
    return mapping


def _dominant_truth_voice_in_interval(
    t0: float, t1: float, utts: list[dict]
) -> str | None:
    scores: dict[str, float] = {}
    for u in utts:
        d = _intersect_len(t0, t1, float(u["t0"]), float(u["t1"]))
        if d > 0:
            vid = u["voice_id"]
            scores[vid] = scores.get(vid, 0.0) + d
    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


def compare_truth_to_manifest(workdir: Path) -> dict:
    truth_path = workdir / "benchmark_truth.json"
    manifest_path = workdir / "out" / "pslicer_manifest.json"
    if not truth_path.is_file():
        raise SystemExit(f"Missing {truth_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}; run pslicer step first.")

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    mani = json.loads(manifest_path.read_text(encoding="utf-8"))
    utts = truth["utterances"]
    chunks = mani["chunks"]

    spk_map = _greedy_speaker_map(utts, chunks)

    def is_solo_segment(u: dict) -> bool:
        if u.get("bench_segment"):
            return u["bench_segment"] == "solo"
        return len(truth["overlap_adjacency"].get(u["utterance_id"], [])) == 0

    per_utt: list[dict] = []
    ious: list[float] = []
    speaker_ok: list[bool] = []
    dominant_ok: list[bool] = []

    for u in utts:
        best_iou = 0.0
        best_chunk = None
        for c in chunks:
            iu = _iou(float(u["t0"]), float(u["t1"]), float(c["t0_sec"]), float(c["t1_sec"]))
            if iu > best_iou:
                best_iou = iu
                best_chunk = c
        ious.append(best_iou)
        pred_spk = best_chunk["speaker"] if best_chunk else None
        mapped = spk_map.get(pred_spk, "?") if pred_spk else "?"
        ok = mapped == u["voice_id"]
        speaker_ok.append(ok)
        dom = None
        if best_chunk:
            dom = _dominant_truth_voice_in_interval(
                float(best_chunk["t0_sec"]),
                float(best_chunk["t1_sec"]),
                utts,
            )
        dom_match = dom == u["voice_id"] if dom else False
        dominant_ok.append(dom_match)
        ov = len(truth["overlap_adjacency"].get(u["utterance_id"], [])) > 0
        per_utt.append(
            {
                "utterance_id": u["utterance_id"],
                "voice_id": u["voice_id"],
                "best_iou": round(best_iou, 4),
                "in_overlap_region": ov,
                "bench_segment": u.get("bench_segment", "legacy"),
                "matched_pred_speaker": pred_spk,
                "mapped_to_voice": mapped,
                "speaker_match": ok,
                "dominant_truth_in_best_clip": dom,
                "dominant_truth_matches_utterance": dom_match,
            }
        )

    non_ov_ok = [
        ok
        for ok, u in zip(speaker_ok, utts)
        if is_solo_segment(u)
    ]
    solo_dominant_ok = [
        d
        for d, u in zip(dominant_ok, utts)
        if is_solo_segment(u)
    ]
    solo_ious = [iu for iu, u in zip(ious, utts) if is_solo_segment(u)]
    stress_ious = [
        iu
        for iu, u in zip(ious, utts)
        if u.get("bench_segment") == "overlap_stress"
    ]
    report = {
        "mean_best_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "median_best_iou": round(float(np.median(ious)) if ious else 0.0, 4),
        "mean_best_iou_solo_only": round(float(np.mean(solo_ious)) if solo_ious else 0.0, 4),
        "mean_best_iou_overlap_stress_only": round(
            float(np.mean(stress_ious)) if stress_ious else 0.0, 4
        ),
        "speaker_accuracy_all": round(float(np.mean(speaker_ok)) if speaker_ok else 0.0, 4),
        "speaker_accuracy_solo_segments": round(
            float(np.mean(non_ov_ok)) if non_ov_ok else 0.0,
            4,
        ),
        "dominant_truth_match_rate_all": round(
            float(np.mean(dominant_ok)) if dominant_ok else 0.0, 4
        ),
        "dominant_truth_match_rate_solo": round(
            float(np.mean(solo_dominant_ok)) if solo_dominant_ok else 0.0, 4
        ),
        "pyannote_to_voice_guess": spk_map,
        "per_utterance": per_utt,
        "pred_chunk_count": len(chunks),
        "truth_utterance_count": len(utts),
    }
    out_report = workdir / "benchmark_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Benchmark report ===")
    print(f"Truth utterances: {len(utts)}  |  Predicted chunks: {len(chunks)}")
    print(f"Mean best IoU (truth vs nearest clip): {report['mean_best_iou']}")
    print(f"Median best IoU: {report['median_best_iou']}")
    print(f"Speaker match rate (after greedy time-alignment map): {report['speaker_accuracy_all']}")
    print(f"Mean IoU solo segments only: {report['mean_best_iou_solo_only']}")
    print(f"Mean IoU overlap-stress only: {report['mean_best_iou_overlap_stress_only']}")
    print(f"Speaker match (solo segments only): {report['speaker_accuracy_solo_segments']}")
    print(f"Dominant-truth in best IoU clip matches utterance (all): {report['dominant_truth_match_rate_all']}")
    print(f"Dominant-truth match (solo only): {report['dominant_truth_match_rate_solo']}")
    print(f"Greedy pyannote->voice map: {spk_map}")
    print(f"Wrote {out_report}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multi-speaker pslicer benchmark (>=10 voices + overlap)."
    )
    ap.add_argument("--workdir", type=Path, default=_REPO / "pslicer_benchmark_run")
    ap.add_argument("--model", default="base", help="WhisperX model")
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if os.name == "nt":
        pslicer.register_windows_torchcodec_dll_paths()
    pslicer._preconfigure_cuda_for_pyannote()

    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        compare_truth_to_manifest(args.workdir)
        return 0

    if not args.synth_only:
        wav = args.workdir / "benchmark_mix.wav"
        truth = args.workdir / "benchmark_truth.json"
        if not wav.is_file() or not truth.is_file():
            asyncio.run(synthesize_benchmark(args.workdir))
    else:
        asyncio.run(synthesize_benchmark(args.workdir))
        return 0

    hf, src = pslicer._resolve_hf_token()
    if not hf:
        print("Need HUGGING_FACE_HUB_TOKEN or HF_TOKEN for diarization.", file=sys.stderr)
        return 1
    if args.verbose:
        print("HF:", src)

    wav_path = args.workdir / "benchmark_mix.wav"
    out_dir = args.workdir / "out"
    manifest = out_dir / "pslicer_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running pslicer pipeline (this may take several minutes)...", flush=True)
    pslicer.run_auto_trim(
        str(wav_path),
        str(out_dir),
        model_name=args.model,
        language="en",
        diarize=True,
        min_speakers=8,
        max_speakers=16,
        batch_size=8,
        padding_ms=120,
        hf_token=hf,
        verbose=args.verbose,
        smooth_speakers=True,
        manifest_path=str(manifest),
    )

    compare_truth_to_manifest(args.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
