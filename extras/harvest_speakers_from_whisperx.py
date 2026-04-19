"""
Sentence-level WAV export from WhisperX JSON + clean source WAVs.

Each entry in `segments` is one utterance: a separate file under
  harvested_sentences/<EpisodeName>/<SpeakerID>/
named
  EpisodeName_SpeakerID_Sentence###.wav

Pads each slice with 250 ms lead/trail (audio where available, silence at file edges).
Drops segments shorter than 1.5 s (JSON start/end), before padding.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any

from pydub import AudioSegment

clean_dir = r"C:\AI\SanctumCore\voice_assets\raw_source\clean"
analysis_dir = r"C:\AI\SanctumCore\voice_assets\analysis"
harvest_dir = r"C:\AI\SanctumCore\voice_assets\harvested_sentences"

_WIN_INVALID = re.compile(r'[<>:"/\\|?*]')

PAD_MS = 250
MIN_SEGMENT_S = 1.5


def _safe_label(name: str, max_len: int = 180) -> str:
    name = _WIN_INVALID.sub("_", name.strip()) or "unknown"
    return name[:max_len]


def _segments_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    segs = data.get("segments")
    return segs if isinstance(segs, list) else []


def _build_padded_slice(
    audio: AudioSegment,
    seg_start_s: float,
    seg_end_s: float,
) -> AudioSegment:
    """Expand [seg_start, seg_end] by PAD_MS on each side; pad with silence beyond file bounds."""
    t0_ms = seg_start_s * 1000.0
    t1_ms = seg_end_s * 1000.0
    want_start_ms = t0_ms - PAD_MS
    want_end_ms = t1_ms + PAD_MS

    prefix = AudioSegment.silent(duration=max(0, int(round(-want_start_ms))))
    clip_lo = max(0, int(round(want_start_ms)))
    clip_hi = min(len(audio), int(round(want_end_ms)))
    body = audio[clip_lo:clip_hi]
    tail_need = int(round(want_end_ms)) - clip_hi
    suffix = AudioSegment.silent(duration=max(0, tail_need))
    return prefix + body + suffix


def main() -> None:
    os.makedirs(harvest_dir, exist_ok=True)

    print("Sentence-level harvest from WhisperX segments (utterance = one WAV)...")

    json_files = [f for f in os.listdir(analysis_dir) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files in {analysis_dir}")
        return

    total_kept = 0
    total_skipped_short = 0

    for json_file in sorted(json_files):
        base_name = json_file[: -len(".json")]
        wav_name = base_name + ".wav"
        wav_path = os.path.join(clean_dir, wav_name)

        if not os.path.exists(wav_path):
            print(f"Warning: clean WAV missing for {wav_name}; skipping JSON.")
            continue

        episode_key = _safe_label(base_name)

        with open(os.path.join(analysis_dir, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = _segments_list(data)
        if not segments:
            print(f"  {wav_name}: no segments in JSON; skip.")
            continue

        full_audio = AudioSegment.from_wav(wav_path)
        sentence_index: dict[str, int] = defaultdict(int)

        print(f"\nProcessing: {wav_name} ({len(segments)} segments)")

        for seg in segments:
            try:
                start_s = float(seg.get("start", 0))
                end_s = float(seg.get("end", 0))
            except (TypeError, ValueError):
                continue

            dur = end_s - start_s
            if dur < MIN_SEGMENT_S:
                total_skipped_short += 1
                continue

            speaker = seg.get("speaker", "UNKNOWN_SPEAKER")
            spk_key = _safe_label(str(speaker))

            sentence_index[spk_key] += 1
            idx = sentence_index[spk_key]

            # Clamp times to audio length for sensible slicing
            audio_len_s = len(full_audio) / 1000.0
            start_s = max(0.0, min(start_s, audio_len_s))
            end_s = max(start_s, min(end_s, audio_len_s))

            chunk = _build_padded_slice(full_audio, start_s, end_s)
            if len(chunk) == 0:
                continue

            out_dir = os.path.join(harvest_dir, episode_key, spk_key)
            os.makedirs(out_dir, exist_ok=True)

            out_name = f"{episode_key}_{spk_key}_Sentence{idx:03d}.wav"
            out_path = os.path.join(out_dir, out_name)
            chunk.export(out_path, format="wav")
            total_kept += 1
            print(f"  -> {out_name}  ({dur:.2f}s utterance + padding)")

    print(
        f"\nDone. Exported {total_kept} sentence WAVs to {harvest_dir}; "
        f"skipped {total_skipped_short} segments under {MIN_SEGMENT_S}s."
    )


if __name__ == "__main__":
    main()
