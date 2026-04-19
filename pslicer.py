"""
pslicer — Auto-trim WAVs at sentence boundaries with optional per-speaker separation.

Pipeline (industry-standard for multi-speaker speech):
  1) WhisperX ASR (faster-whisper) + VAD
  2) Forced alignment → accurate word-level timestamps
  3) Pyannote speaker diarization (optional, recommended for multiple voices)
  4) assign_word_speakers → each word tagged with a speaker
  5) Contextual speaker smoothing (duration-weighted local vote + short “island” collapse)
  6) Chunk audio on: (a) sentence-ending punctuation, (b) speaker change
  7) Merge same-speaker clips across pauses when the text/timing suggests one utterance
     (abbreviations, breath gaps, discourse continuations, lowercase follow-ons, full stop + new
     capitalized sentence on the same diarization label — not long scene breaks).
     Optional: speaker-embedding voice gate — boundary crops plus multi-crop chunk centroids, cross-boundary
     crops, and a per-speaker profile map (running x-vector means per diarization label) for robustness
     to cadence / prosody variation within the same voice.
  8) Optional preview before export: terminal (``--preview``) or Voice Engine GUI (AI trim button).
  9) Export each chunk as WAV (filenames lead with a zero-padded start time in ms). Sentence-final
     chunks get extra tail time (capped by the next word) so phrase endings are not clipped sharply.

Diarization requires Hugging Face token + accepting model terms (same idea as ``extras/batch_whisperx_analysis.py``).
Voice pause gate (optional) loads a speaker embedding: default ``auto`` tries ONNX WeSpeaker
(``hbredin/wespeaker-voxceleb-resnet34-LM``, needs ``onnxruntime``), then pyannote/embedding (gated).
Diarization weights are freed before the embedder loads to reduce GPU memory spikes.
Avoid installing ``speechbrain`` in the same venv: it can break pyannote/Lightning model loading.

Examples:
  set HF_TOKEN=hf_...
  python pslicer.py recording.wav
  python pslicer.py recording.wav --out C:\\exports --model medium --max-speakers 4
  python pslicer.py recording.wav --no-diarize
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Callable

import wav_metadata as _wav_meta

# WhisperX shells out to ffmpeg; ensure venv Scripts is on PATH on Windows.
_bindir = os.path.dirname(os.path.abspath(sys.executable))
os.environ["PATH"] = _bindir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
from pydub import AudioSegment

warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.models\.blocks\.pooling")
warnings.filterwarnings("ignore", message=r".*TensorFloat-32 \(TF32\).*", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"onnxruntime\.capi\.onnxruntime_inference_collection",
)
warnings.filterwarnings("ignore", message=r".*Lightning automatically upgraded your loaded checkpoint.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pydub\.utils")

# Windows: TorchCodec / pyannote DLL paths (same as whisperx_windows_entry).
try:
    from whisperx_windows_entry import _preconfigure_cuda_for_pyannote
    from whisperx_windows_entry import register_windows_torchcodec_dll_paths
except ImportError:
    def register_windows_torchcodec_dll_paths() -> bool:
        return True

    def _preconfigure_cuda_for_pyannote() -> None:
        pass


def _resolve_hf_token() -> tuple[str | None, str | None]:
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t and t.strip():
        return t.strip(), "HF_TOKEN / HUGGING_FACE_HUB_TOKEN"
    try:
        from huggingface_hub import get_token

        t2 = get_token()
        if t2:
            return t2.strip(), "huggingface-cli cache"
    except Exception:
        pass
    return None, None


class PslicerUserError(RuntimeError):
    """Recoverable pipeline error (missing HF token, bad audio/ASR, etc.)."""


def resolve_hf_token() -> tuple[str | None, str | None]:
    """Return ``(token, source_description)`` for Hugging Face Hub; token may be ``None``."""
    return _resolve_hf_token()


_SENTENCE_END_RE = re.compile(r"([.?!…]+|\.[\"')\]]*)\s*$")


def _sentence_ends(word: str) -> bool:
    """Heuristic: Whisper word token often includes trailing punctuation."""
    w = (word or "").strip()
    if not w:
        return False
    if w[-1] in ".?!…":
        return True
    # Closing quote after period:  word." or word.'
    if len(w) >= 2 and w[-1] in "\"'\u201d\u2019)" and w[-2] in ".?!…":
        return True
    return bool(_SENTENCE_END_RE.search(w))


# Periods that usually should not end an exported utterance (false sentence splits).
_ABBREV_PERIOD = frozenset(
    x.lower()
    for x in (
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "i.e.",
        "e.g.",
        "st.",
        "ave.",
        "no.",
        "approx.",
    )
)


def _strip_outer_quotes(s: str) -> str:
    t = (s or "").strip()
    while len(t) >= 2 and t[0] in "'\"\u2018\u201c" and t[-1] in "'\"\u2019\u201d":
        t = t[1:-1].strip()
    return t


def _last_token(text: str) -> str:
    parts = (text or "").strip().split()
    return parts[-1] if parts else ""


def _first_token(text: str) -> str:
    parts = (text or "").strip().split()
    return parts[0] if parts else ""


def _likely_abbreviation_period(token: str) -> bool:
    t = _strip_outer_quotes(token).lower()
    if t in _ABBREV_PERIOD:
        return True
    # Single capital letter + dot (initials)
    if re.match(r"^[a-z]\.$", t, re.I):
        return True
    return False


def _probably_full_sentence_terminal(last_token: str) -> bool:
    """True if this token likely ends a complete sentence (not Mr./Dr./etc.)."""
    if not _sentence_ends(last_token):
        return False
    if _likely_abbreviation_period(last_token):
        return False
    return True


def _starts_lowercase_continuation(text: str) -> bool:
    """Next chunk begins like a dependent clause / same breath (e.g. 'and ...', 'who ...')."""
    t = _strip_outer_quotes((text or "").strip())
    if not t:
        return False
    i = 0
    while i < len(t) and t[i] in "'\u2018":
        i += 1
    if i >= len(t):
        return False
    return t[i].isalpha() and t[i].islower()


def _starts_new_sentence_continuation(text: str) -> bool:
    """
    Next chunk begins like a new sentence after a full stop (capital letter, digit, etc.).

    Same-speaker diarization often still splits on ``.``; this pairs with the voice gate for
    pauses longer than a breath so embeddings decide identity across the gap.
    """
    t = (text or "").strip()
    if not t:
        return False
    i = 0
    while i < len(t):
        c = t[i]
        if c.isspace():
            i += 1
            continue
        if c in "'\"\u2018\u201c([{":
            i += 1
            continue
        if c.isdigit():
            return True
        if c.isalpha():
            return c.isupper()
        return False
    return False


_DISCOURSE_CONTINUE_RE = re.compile(
    r"(?i)^(and|but|so|then|or|nor|yet|still|because|although|though|unless|until|while|where|when)\b"
)


def _discourse_continuation_turn(text: str) -> bool:
    """Same speaker likely continuing their turn (new sentence, same beat)."""
    t = (text or "").lstrip().lstrip('\'"“‘')
    return bool(_DISCOURSE_CONTINUE_RE.match(t))


def _pause_merge_strong_text_signal(last_tok: str, text: str) -> bool:
    """Text cues that justify a merge even when acoustic match is only moderate."""
    return (
        _likely_abbreviation_period(last_tok)
        or _discourse_continuation_turn(text)
        or (
            _probably_full_sentence_terminal(last_tok)
            and _starts_lowercase_continuation(text)
        )
        or (
            _probably_full_sentence_terminal(last_tok)
            and _starts_new_sentence_continuation(text)
        )
    )


def _cosine_similarity_1d(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float64) if n > 1e-12 else v.astype(np.float64)


class ChunkVoiceComparator:
    """
    Speaker embedding voice gate: boundary crops, multi-crop chunk profiles, and per-label maps.
    Embeddings are the measured voice parameters (fixed-dimensional timbre vectors); we aggregate
    several crops per chunk so one sentence can vary in cadence/prosody without breaking identity.
    """

    def __init__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        device: torch.device,
        hf_token: str | None,
        embedding_model: str = "pyannote/embedding",
        model_dir: str | None = None,
        window_sec: float = 0.45,
    ) -> None:
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

        self._emb = PretrainedSpeakerEmbedding(
            embedding_model, device=device, token=hf_token, cache_dir=model_dir
        )
        self._device = device
        self._sr_model = int(self._emb.sample_rate)
        self._window = float(window_sec)
        self._min_n = int(self._emb.min_num_samples)
        self._profile_by_speaker: dict[str, np.ndarray] = {}
        self._centroid_cache: dict[tuple[float, float], np.ndarray] = {}

        w = np.asarray(waveform, dtype=np.float32)
        if sample_rate != self._sr_model:
            import torchaudio

            t = torch.from_numpy(w).unsqueeze(0)
            w = (
                torchaudio.functional.resample(
                    t, orig_freq=sample_rate, new_freq=self._sr_model
                )
                .squeeze(0)
                .numpy()
                .astype(np.float32)
            )
        self._wav = w

    def _embed_samples(self, i0: int, i1: int) -> np.ndarray | None:
        nmax = len(self._wav)
        i0 = max(0, min(nmax, int(i0)))
        i1 = max(0, min(nmax, int(i1)))
        if i1 - i0 < self._min_n:
            return None
        clip = self._wav[i0:i1]
        with torch.inference_mode():
            t = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).to(self._device)
            try:
                e = np.asarray(self._emb(t)[0], dtype=np.float64)
            finally:
                del t
        return _l2_normalize_vec(e)

    def _embed_centered_sec(self, center_sec: float, t_lo: float, t_hi: float) -> np.ndarray | None:
        sr = self._sr_model
        nmax = len(self._wav)
        win = max(self._min_n, int(round(self._window * sr)))
        lo_b = max(0, int(round(float(t_lo) * sr)))
        hi_b = min(nmax, int(round(float(t_hi) * sr)))
        if hi_b - lo_b < self._min_n:
            return None
        c = int(round(float(center_sec) * sr))
        c = max(lo_b + win // 2, min(hi_b - (win - win // 2), c))
        lo = c - win // 2
        hi = lo + win
        lo = max(lo_b, lo)
        hi = min(hi_b, hi)
        if hi - lo < self._min_n:
            lo = max(lo_b, hi_b - self._min_n)
            hi = hi_b
        return self._embed_samples(lo, hi)

    def chunk_profile_centroid(self, t0: float, t1: float) -> np.ndarray | None:
        """Mean of L2-normalized embeddings at several times in the chunk (identity vs. local prosody)."""
        key = (round(float(t0), 4), round(float(t1), 4))
        if key in self._centroid_cache:
            return self._centroid_cache[key]
        dur = float(t1) - float(t0)
        sr = self._sr_model
        if dur <= 0:
            return None
        min_sec = self._min_n / float(sr)
        if dur < min_sec * 1.25:
            c = self._embed_centered_sec((t0 + t1) / 2.0, t0, t1)
            if c is not None:
                self._centroid_cache[key] = c
            return c
        fracs = (0.18, 0.45, 0.72)
        vecs: list[np.ndarray] = []
        for a in fracs:
            center = float(t0) + a * dur
            e = self._embed_centered_sec(center, t0, t1)
            if e is not None:
                vecs.append(e)
        if not vecs:
            return None
        raw = np.mean(np.stack(vecs, axis=0), axis=0)
        out = _l2_normalize_vec(raw)
        self._centroid_cache[key] = out
        return out

    def precompute_speaker_profiles(
        self,
        chunks: list[tuple[float, float, str, str]],
    ) -> None:
        """Map each diarization label -> mean chunk centroid (measurable voice summary for the file)."""
        acc: dict[str, list[np.ndarray]] = {}
        for t0, t1, spk, _ in chunks:
            c = self.chunk_profile_centroid(t0, t1)
            if c is None:
                continue
            acc.setdefault(spk, []).append(c)
        self._profile_by_speaker = {}
        for spk, vecs in acc.items():
            if not vecs:
                continue
            m = np.mean(np.stack(vecs, axis=0), axis=0)
            self._profile_by_speaker[spk] = _l2_normalize_vec(m)

    def tail_head_cosine(
        self,
        prev_t0: float,
        prev_t1: float,
        next_t0: float,
        next_t1: float,
    ) -> float | None:
        """Cosine similarity between embedding(tail of prev) and embedding(head of next)."""
        sr = self._sr_model
        nmax = len(self._wav)
        n_tail = max(self._min_n, int(round(self._window * sr)))
        n_head = max(self._min_n, int(round(self._window * sr)))

        i1 = min(nmax, max(0, int(round(prev_t1 * sr))))
        i0 = max(0, int(round(prev_t0 * sr)), i1 - n_tail)
        if i1 - i0 < self._min_n:
            return None

        j0 = max(0, int(round(next_t0 * sr)))
        j1 = min(nmax, int(round(next_t1 * sr)), j0 + n_head)
        if j1 - j0 < self._min_n:
            return None

        tail = self._wav[i0:i1]
        head = self._wav[j0:j1]
        with torch.inference_mode():
            tw = torch.from_numpy(tail).float().unsqueeze(0).unsqueeze(0).to(self._device)
            hw = torch.from_numpy(head).float().unsqueeze(0).unsqueeze(0).to(self._device)
            try:
                et = np.asarray(self._emb(tw)[0], dtype=np.float64)
                eh = np.asarray(self._emb(hw)[0], dtype=np.float64)
            finally:
                del tw, hw
        return _cosine_similarity_1d(et, eh)

    def _cross_boundary_crop_cosines(
        self,
        prev_t0: float,
        prev_t1: float,
        next_t0: float,
        next_t1: float,
    ) -> float | None:
        """Mean cosine over several tail-ish vs head-ish crops (robust to phrase-final prosody)."""
        pd = max(1e-6, float(prev_t1) - float(prev_t0))
        nd = max(1e-6, float(next_t1) - float(next_t0))
        prev_centers = (float(prev_t0) + 0.70 * pd, float(prev_t0) + 0.88 * pd)
        next_centers = (float(next_t0) + 0.12 * nd, float(next_t0) + 0.30 * nd)
        pairs: list[float] = []
        for pc in prev_centers:
            ep = self._embed_centered_sec(pc, prev_t0, prev_t1)
            if ep is None:
                continue
            for nc in next_centers:
                en = self._embed_centered_sec(nc, next_t0, next_t1)
                if en is None:
                    continue
                pairs.append(_cosine_similarity_1d(ep, en))
        if not pairs:
            return None
        return float(sum(pairs) / len(pairs))

    def pause_merge_voice_score(
        self,
        prev_t0: float,
        prev_t1: float,
        next_t0: float,
        next_t1: float,
        speaker: str,
    ) -> float | None:
        """
        Single scalar in [-1,1] (typical 0..1): blend boundary, pair centroids, cross-boundary crops,
        and match of next chunk to precomputed speaker profile.
        """
        b = self.tail_head_cosine(prev_t0, prev_t1, next_t0, next_t1)
        pc = self.chunk_profile_centroid(prev_t0, prev_t1)
        nc = self.chunk_profile_centroid(next_t0, next_t1)
        p = _cosine_similarity_1d(pc, nc) if pc is not None and nc is not None else None
        x = self._cross_boundary_crop_cosines(prev_t0, prev_t1, next_t0, next_t1)
        prof = self._profile_by_speaker.get(speaker)
        q = _cosine_similarity_1d(nc, prof) if nc is not None and prof is not None else None

        w_b, w_p, w_x, w_q = 0.34, 0.28, 0.26, 0.12
        parts: list[tuple[float, float]] = []
        if b is not None:
            parts.append((w_b, float(b)))
        if p is not None:
            parts.append((w_p, float(p)))
        if x is not None:
            parts.append((w_x, float(x)))
        if q is not None:
            parts.append((w_q, float(q)))
        if not parts:
            return None
        ws = sum(w for w, _ in parts)
        if ws < 1e-9:
            return None
        return float(sum(w * s for w, s in parts) / ws)


_VOICE_EMBEDDING_AUTO_ORDER: tuple[str, ...] = (
    "hbredin/wespeaker-voxceleb-resnet34-LM",
    "pyannote/embedding",
)


def _build_voice_comparator(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    embedding_model: str,
    device: torch.device,
    hf_token: str | None,
    model_dir: str | None,
    window_sec: float,
    verbose: bool,
) -> tuple[ChunkVoiceComparator | None, str | None]:
    """
    Load ChunkVoiceComparator. ``embedding_model`` ``auto`` tries WeSpeaker ONNX, then pyannote.
    """
    key = (embedding_model or "auto").strip().lower()
    if key == "auto":
        last_err: Exception | None = None
        for mid in _VOICE_EMBEDDING_AUTO_ORDER:
            try:
                cmp_ = ChunkVoiceComparator(
                    waveform,
                    sample_rate,
                    device=device,
                    hf_token=hf_token,
                    embedding_model=mid,
                    model_dir=model_dir,
                    window_sec=window_sec,
                )
                if verbose:
                    print(f"  Voice embedding: {mid}", flush=True)
                return cmp_, mid
            except Exception as e:
                last_err = e
                if verbose:
                    print(f"  Voice embedding skip {mid}: {type(e).__name__}: {e}", flush=True)
        if verbose:
            print(
                f"  Voice pause gate: no embedding model available (last error: {last_err})",
                flush=True,
            )
        return None, None
    try:
        cmp_ = ChunkVoiceComparator(
            waveform,
            sample_rate,
            device=device,
            hf_token=hf_token,
            embedding_model=embedding_model,
            model_dir=model_dir,
            window_sec=window_sec,
        )
        return cmp_, embedding_model
    except Exception as e:
        print(
            "Voice pause gate disabled (embedding model not loaded). "
            "Use --voice-embedding-model auto, install onnxruntime, accept HF terms for "
            "pyannote/embedding, or pass --no-voice-pause-gate. "
            f"Detail: {e}",
            file=sys.stderr,
            flush=True,
        )
        return None, None


def merge_chunks_smart_same_speaker_pauses(
    chunks: list[tuple[float, float, str, str]],
    *,
    hard_split_gap_sec: float = 3.6,
    max_glue_gap_sec: float = 2.5,
    breath_gap_sec: float = 0.42,
    discourse_gap_sec: float = 1.85,
    period_glue_if_lowercase_sec: float = 1.25,
    voice_compare: ChunkVoiceComparator | None = None,
    voice_gate_above_gap_sec: float | None = None,
    voice_min_cosine: float = 0.55,
    voice_veto_cosine: float = 0.38,
) -> list[tuple[float, float, str, str]]:
    """
    Join adjacent clips that share a speaker when the inter-clip pause looks like a mid-utterance
    breath or turn continuation, not a new scene.

    Uses timing + light text heuristics. Optionally refines candidates with ``ChunkVoiceComparator``
    (boundary line, multi-crop centroids, cross-boundary crops, speaker profile map), including
    full-stop + new capitalized sentence on the same diarization label.
    """
    gate_above = breath_gap_sec if voice_gate_above_gap_sec is None else float(voice_gate_above_gap_sec)
    if not chunks or len(chunks) < 2:
        return chunks

    merged: list[tuple[float, float, str, str]] = []
    for t0, t1, spk, text in chunks:
        if not merged:
            merged.append((t0, t1, spk, text))
            continue
        pt0, pt1, pspk, ptext = merged[-1]
        if spk != pspk:
            merged.append((t0, t1, spk, text))
            continue
        gap = float(t0) - float(pt1)
        if gap < 0:
            merged.append((t0, t1, spk, text))
            continue
        if gap >= hard_split_gap_sec:
            merged.append((t0, t1, spk, text))
            continue

        last_tok = _last_token(ptext)
        glue = False
        if gap <= breath_gap_sec:
            glue = True
        elif _likely_abbreviation_period(last_tok):
            glue = True
        elif gap <= discourse_gap_sec and _discourse_continuation_turn(text):
            glue = True
        elif (
            gap <= period_glue_if_lowercase_sec
            and _probably_full_sentence_terminal(last_tok)
            and _starts_lowercase_continuation(text)
        ):
            glue = True
        elif _probably_full_sentence_terminal(last_tok) and _starts_new_sentence_continuation(text):
            # Full stop + new sentence (e.g. "… said. She …") on the same diarization label.
            # Without embeddings, only merge within the tighter window; with voice gate, up to max_glue.
            cap = max_glue_gap_sec if voice_compare is not None else period_glue_if_lowercase_sec
            if gap <= cap:
                glue = True
        elif gap <= max_glue_gap_sec and not _probably_full_sentence_terminal(last_tok):
            glue = True
        elif gap <= max_glue_gap_sec and not _sentence_ends(last_tok):
            glue = True

        if glue and voice_compare is not None and gap > gate_above:
            b_line = voice_compare.tail_head_cosine(pt0, pt1, t0, t1)
            if b_line is not None and b_line < voice_veto_cosine:
                glue = False
            elif glue:
                score = voice_compare.pause_merge_voice_score(pt0, pt1, t0, t1, pspk)
                sim = score if score is not None else b_line
                if sim is not None:
                    if sim < voice_veto_cosine:
                        glue = False
                    elif sim < voice_min_cosine and not _pause_merge_strong_text_signal(last_tok, text):
                        glue = False

        if glue:
            merged[-1] = (pt0, t1, pspk, f"{ptext} {text}".strip())
        else:
            merged.append((t0, t1, spk, text))

    return merged


def _speaker_label(word: dict) -> str:
    sp = word.get("speaker")
    if sp is None or sp == "":
        return "SPEAKER_UNKNOWN"
    return str(sp)


def _sanitize(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_") or "unk"


def _chunk_start_sort_token_and_ms(t0_sec: float) -> tuple[str, int]:
    """
    Lexicographically sortable file token + integer milliseconds for manifests.
    Caps at 999_999_999 ms (~277 h) so the token stays 9 digits after ``t``.
    """
    ms = int(round(max(0.0, float(t0_sec)) * 1000.0))
    ms = max(0, min(ms, 999_999_999))
    return f"t{ms:09d}", ms


def _collect_aligned_words(result: dict) -> list[dict]:
    words: list[dict] = []
    for seg in result.get("segments") or []:
        for w in seg.get("words") or []:
            if w.get("start") is None:
                continue
            words.append(w)
    words.sort(key=lambda x: float(x["start"]))
    return words


def transcript_for_time_range(words: list[dict], t0: float, t1: float) -> str:
    """
    Join Whisper / alignment word tokens whose timestamps overlap ``[t0, t1]`` (seconds).

    Used when the user nudges clip boundaries in the GUI so the displayed sentence matches
    the words that fall inside the new window.
    """
    t0f, t1f = float(t0), float(t1)
    if t1f <= t0f or not words:
        return ""
    parts: list[tuple[float, str]] = []
    for w in words:
        ws_raw = w.get("start")
        if ws_raw is None:
            continue
        ws = float(ws_raw)
        we_raw = w.get("end")
        we = float(we_raw) if we_raw is not None else ws + 0.05
        if ws < t1f and we > t0f:
            tok = str(w.get("word") or "").strip()
            if tok:
                parts.append((ws, tok))
    parts.sort(key=lambda x: x[0])
    return " ".join(p[1] for p in parts).strip()


def _chunk_end_time(words: list[dict]) -> float:
    last = words[-1]
    t1 = last.get("end")
    if t1 is not None:
        return float(t1)
    return float(last["start"]) + 0.05


def _extend_sentence_end_time(
    t1_align: float,
    *,
    next_bound_sec: float | None,
    audio_duration_sec: float | None,
    tail_sec: float,
    min_gap_before_next_sec: float,
) -> float:
    """
    Add time after the last aligned word so clips do not cut off release, breath, or final prosody.
    Capped by the next word start (if any) and file end.
    """
    if tail_sec <= 0:
        return float(t1_align)
    ext = float(t1_align) + float(tail_sec)
    if next_bound_sec is not None:
        cap = float(next_bound_sec) - float(min_gap_before_next_sec)
        if cap > t1_align:
            ext = min(ext, cap)
    if audio_duration_sec is not None:
        ext = min(ext, float(audio_duration_sec))
    return max(float(t1_align), ext)


def _word_mid_and_duration(w: dict) -> tuple[float, float]:
    s = float(w["start"])
    e_raw = w.get("end")
    e = float(e_raw) if e_raw is not None else s + 0.05
    return (s + e) / 2.0, max(0.02, e - s)


def apply_contextual_speaker_smoothing(
    words: list[dict],
    *,
    half_window_sec: float = 0.45,
    max_island_words: int = 2,
    max_island_sec: float = 0.18,
    preserve_raw: bool = True,
) -> list[dict]:
    """
    Refine per-word speaker labels using local temporal context (logic layer on top of pyannote).

    1) Duration-weighted vote: each word inherits the speaker with the largest weighted
       presence in a time window (robust to single-word diarization glitches).
    2) Island collapse: very short runs (few words + short time) sandwiched between longer
       runs of another speaker are merged into the dominant neighbor (typical boundary flip).
    """
    if not words or half_window_sec <= 0:
        return words

    if preserve_raw:
        for w in words:
            if "speaker_raw" not in w and "speaker" in w:
                w["speaker_raw"] = w["speaker"]

    labels: list[str] = []
    mids: list[float] = []
    durs: list[float] = []
    for w in words:
        m, d = _word_mid_and_duration(w)
        mids.append(m)
        durs.append(d)
        labels.append(_speaker_label(w))

    n = len(words)
    smoothed: list[str] = []
    for i in range(n):
        t0 = mids[i] - half_window_sec
        t1 = mids[i] + half_window_sec
        scores: dict[str, float] = {}
        for j in range(n):
            if t0 <= mids[j] <= t1:
                sp = labels[j]
                scores[sp] = scores.get(sp, 0.0) + durs[j]
        if not scores:
            smoothed.append(labels[i])
        else:
            smoothed.append(max(scores.items(), key=lambda x: x[1])[0])

    for w, sp in zip(words, smoothed):
        w["speaker"] = sp

    _collapse_speaker_islands(
        words,
        max_island_words=max_island_words,
        max_island_sec=max_island_sec,
    )
    return words


def _collapse_speaker_islands(
    words: list[dict],
    *,
    max_island_words: int,
    max_island_sec: float,
) -> None:
    """Merge tiny speaker runs into the surrounding majority (in-place)."""
    if not words or max_island_words < 1:
        return

    n = len(words)
    labs = [_speaker_label(w) for w in words]

    def run_bounds(start: int) -> tuple[int, int, str]:
        sp = labs[start]
        j = start
        while j < n and labs[j] == sp:
            j += 1
        return start, j, sp
    i = 0
    while i < n:
        a, b, sp = run_bounds(i)
        run_dur = sum(_word_mid_and_duration(words[k])[1] for k in range(a, b))
        prev_sp = labs[a - 1] if a > 0 else None
        next_sp = labs[b] if b < n else None
        merge_to: str | None = None
        # Only collapse "sandwiched" glitches (A…B…A). Edge runs at file start/end are kept;
        # merging them into the neighbor without a matching far side caused false speaker swaps.
        if (
            b - a <= max_island_words
            and run_dur <= max_island_sec
            and sp != "SPEAKER_UNKNOWN"
            and prev_sp is not None
            and next_sp is not None
            and prev_sp == next_sp
            and sp != prev_sp
        ):
            merge_to = prev_sp
        if merge_to is not None:
            for k in range(a, b):
                words[k]["speaker"] = merge_to
                labs[k] = merge_to
        i = b


def iter_sentence_speaker_chunks(
    words: list[dict],
    *,
    min_duration: float = 0.08,
    audio_duration_sec: float | None = None,
    sentence_end_tail_sec: float = 0.32,
    min_gap_before_next_sec: float = 0.025,
) -> list[tuple[float, float, str, str]]:
    """
    Returns list of (t0, t1, speaker_id, text_joined) for each exported region.
    Flushes when the speaker changes or when a word ends a sentence.

    For sentence-ending flushes, ``t1`` is extended by ``sentence_end_tail_sec`` (capped by the
    next word and file end) so exports are less likely to clip natural phrase-final decay.
    """
    chunks: list[tuple[float, float, str, str]] = []
    buf: list[dict] = []
    n = len(words)

    def flush(*, sentence_tail: bool, next_word_start: float | None) -> None:
        nonlocal buf
        if not buf:
            return
        t0 = float(buf[0]["start"])
        t1 = _chunk_end_time(buf)
        if sentence_tail and sentence_end_tail_sec > 0:
            t1 = _extend_sentence_end_time(
                t1,
                next_bound_sec=next_word_start,
                audio_duration_sec=audio_duration_sec,
                tail_sec=sentence_end_tail_sec,
                min_gap_before_next_sec=min_gap_before_next_sec,
            )
        if t1 - t0 < min_duration:
            buf = []
            return
        sp = _speaker_label(buf[0])
        # Majority speaker label if drift (rare)
        speakers = {_speaker_label(w) for w in buf}
        if len(speakers) == 1:
            label = sp
        else:
            label = max(speakers, key=lambda s: sum(1 for w in buf if _speaker_label(w) == s))
        text = " ".join((w.get("word") or "").strip() for w in buf).strip()
        chunks.append((t0, t1, label, text))
        buf = []

    for idx, w in enumerate(words):
        sp = _speaker_label(w)
        if buf and sp != _speaker_label(buf[-1]):
            flush(sentence_tail=False, next_word_start=None)
        buf.append(w)
        if _sentence_ends(str(w.get("word") or "")):
            nxt = None
            if idx + 1 < n:
                ns = words[idx + 1].get("start")
                if ns is not None:
                    nxt = float(ns)
            flush(sentence_tail=True, next_word_start=nxt)

    flush(
        sentence_tail=bool(buf) and _sentence_ends(str(buf[-1].get("word") or "")),
        next_word_start=None,
    )
    return chunks


def export_one_wav_clip(
    audio_path: str,
    dest_path: str,
    t0_sec: float,
    t1_sec: float,
    *,
    padding_ms: int = 120,
    speaker: str = "",
    transcript: str = "",
    source_basename: str | None = None,
    embed_metadata: bool = True,
) -> None:
    """
    Write a single padded slice of ``audio_path`` to ``dest_path`` as WAV.

    Uses the same ms window as :func:`export_wav_clips` for the given ``(t0_sec, t1_sec)``.
    """
    audio_path = str(Path(audio_path).resolve())
    dest_path = str(Path(dest_path).resolve())
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    segment = AudioSegment.from_wav(audio_path)
    try:
        total_ms = len(segment)
        pad = max(0, int(padding_ms))
        ms0 = max(0, int(float(t0_sec) * 1000) - pad)
        ms1 = min(total_ms, int(float(t1_sec) * 1000) + pad)
        if ms1 <= ms0:
            raise ValueError("clip window empty after padding")
        segment[ms0:ms1].export(dest_path, format="wav")
        if embed_metadata:
            try:
                _wav_meta.embed_voice_engine_wav_metadata(
                    dest_path,
                    source_audio_basename=source_basename or Path(audio_path).name,
                    speaker=speaker,
                    transcript=transcript,
                    trim_export_start_ms=ms0,
                    trim_export_end_ms=ms1,
                )
            except Exception:
                pass
    finally:
        del segment
        gc.collect()


def export_wav_clips(
    audio_path: str,
    out_dir: str,
    chunks: list[tuple[float, float, str, str]],
    *,
    stem: str | None = None,
    padding_ms: int = 120,
    verbose: bool = False,
    manifest_path: str | Path | None = None,
    return_source_indices: bool = False,
    embed_metadata: bool = True,
) -> list[str] | tuple[list[str], list[int]]:
    """
    Slice ``audio_path`` (WAV) into clips from (t0, t1, speaker, text) chunks.

    If ``return_source_indices`` is True, returns ``(paths, source_indices)`` where
    ``source_indices[k]`` is the index into ``chunks`` for ``paths[k]`` (only exported clips).
    """
    audio_path = str(Path(audio_path).resolve())
    out_dir = str(Path(out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)
    segment = AudioSegment.from_wav(audio_path)
    total_ms = len(segment)
    stem = stem or Path(audio_path).stem
    pad = max(0, int(padding_ms))
    exported: list[str] = []
    source_indices: list[int] = []
    manifest_chunks: list[dict] = []
    out_idx = 0
    try:
        for i, (t0, t1, spk, text) in enumerate(chunks):
            ms0 = max(0, int(t0 * 1000) - pad)
            ms1 = min(total_ms, int(t1 * 1000) + pad)
            if ms1 <= ms0:
                continue
            clip = segment[ms0:ms1]
            sort_tok, start_ms = _chunk_start_sort_token_and_ms(t0)
            fname = f"{stem}_{sort_tok}_auto_{out_idx + 1:04d}_{_sanitize(spk)}.wav"
            dest = os.path.join(out_dir, fname)
            clip.export(dest, format="wav")
            if embed_metadata:
                try:
                    _wav_meta.embed_voice_engine_wav_metadata(
                        dest,
                        source_audio_basename=Path(audio_path).name,
                        speaker=spk,
                        transcript=text,
                        trim_export_start_ms=ms0,
                        trim_export_end_ms=ms1,
                    )
                except Exception:
                    pass
            exported.append(dest)
            source_indices.append(i)
            manifest_chunks.append(
                {
                    "chunk_index": out_idx + 1,
                    "source_chunk_index": i,
                    "file": fname,
                    "start_ms": start_ms,
                    "start_sort": sort_tok,
                    "t0_sec": round(float(t0), 4),
                    "t1_sec": round(float(t1), 4),
                    "t0_export_sec": round(ms0 / 1000.0, 4),
                    "t1_export_sec": round(ms1 / 1000.0, 4),
                    "speaker": spk,
                    "text": text,
                }
            )
            out_idx += 1
            if verbose:
                print(f"  {fname}  [{t0:.2f} - {t1:.2f}] {spk}", flush=True)
    finally:
        del segment
        gc.collect()

    if manifest_path:
        mp = Path(manifest_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "source_audio": audio_path,
            "out_dir": out_dir,
            "padding_ms": pad,
            "chunks": manifest_chunks,
        }
        mp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        if verbose:
            print(f"Wrote manifest {mp}", flush=True)
    if return_source_indices:
        return exported, source_indices
    return exported


def _play_wav_file(path: str) -> None:
    """Play one WAV without keeping Python-side audio buffers (OS handles file)."""
    p = str(Path(path).resolve())
    if not os.path.isfile(p):
        print(f"(missing) {p}", flush=True)
        return
    if sys.platform == "win32":
        try:
            import winsound

            winsound.PlaySound(p, winsound.SND_FILENAME)
            return
        except Exception:
            pass
    try:
        import subprocess

        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"(New-Object Media.SoundPlayer '{p}').PlaySync()"],
            check=False,
            capture_output=True,
            timeout=600,
        )
    except Exception as e:
        print(f"(play failed) {e}", flush=True)


def interactive_preview_trims(
    audio_path: str,
    chunks: list[tuple[float, float, str, str]],
    *,
    padding_ms: int = 120,
    stem: str | None = None,
    preview_play_all: bool = False,
    verbose: bool = False,
    input_fn: Callable[[], str] | None = None,
) -> tuple[list[tuple[float, float, str, str]], bool]:
    """
    Export clips to a temporary directory, let the user list / play / exclude clips, then return
    the chunk list to export (or abort).

    Returns ``(chunks_to_export, proceed)``. If ``proceed`` is False, caller must not write to
    final output. Temporary files are always removed before return.

    ``input_fn`` defaults to ``input``; for tests pass a callable returning scripted lines.
    """
    if not chunks:
        return [], True

    stem = stem or Path(audio_path).stem
    readline = input_fn or input
    tmp_root: str | None = None
    try:
        tmp_root = tempfile.mkdtemp(prefix="pslicer_preview_")
        prev_manifest = os.path.join(tmp_root, "preview_manifest.json")
        result = export_wav_clips(
            audio_path,
            tmp_root,
            chunks,
            stem=stem,
            padding_ms=padding_ms,
            verbose=verbose,
            manifest_path=prev_manifest,
            return_source_indices=True,
        )
        paths, src_idx = result

        n = len(paths)
        if n == 0:
            return [], True

        excluded_preview: set[int] = set()

        def chunk_line(j: int) -> str:
            si = src_idx[j]
            t0, t1, spk, tx = chunks[si]
            snip = (tx[:72] + "...") if len(tx) > 72 else tx
            mark = " (excluded)" if j in excluded_preview else ""
            return f"  [{j + 1:3d}/{n}]  {t0:.2f}-{t1:.2f}  {spk}{mark}  {snip!r}"

        print(f"\nPreview: {n} clip(s) in temp dir (will be deleted after you finish):", flush=True)
        print(tmp_root, flush=True)
        for j in range(n):
            print(chunk_line(j), flush=True)

        if preview_play_all:
            print("\nPlaying each preview clip in order (OS player; no buffers kept in Python)...", flush=True)
            for j in range(n):
                if j in excluded_preview:
                    continue
                print(f"\n--- Playing {j + 1}/{n} ---", flush=True)
                _play_wav_file(paths[j])
                gc.collect()

        print(
            "\nCommands:  l=list  p N=play  x N=exclude  i N=include  a=all  e=export  q=quit (no export)",
            flush=True,
        )
        print("Press Enter to open command mode...", flush=True)
        readline()

        while True:
            try:
                line = readline().strip()
            except EOFError:
                line = "q"
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            if cmd in ("q", "quit", "exit"):
                return [], False
            if cmd in ("e", "export", "go"):
                kept_src = {src_idx[j] for j in range(n) if j not in excluded_preview}
                ordered = [chunks[i] for i in range(len(chunks)) if i in kept_src]
                return ordered, True
            if cmd in ("a", "all"):
                excluded_preview.clear()
                print("Cleared exclusions.", flush=True)
                continue
            if cmd == "l":
                for j in range(n):
                    print(chunk_line(j), flush=True)
                continue
            if cmd == "p" and len(parts) >= 2:
                try:
                    j = int(parts[1]) - 1
                    if 0 <= j < n:
                        _play_wav_file(paths[j])
                        gc.collect()
                    else:
                        print("Index out of range.", flush=True)
                except ValueError:
                    print("Usage: p N  (N is 1-based)", flush=True)
                continue
            if cmd == "x" and len(parts) >= 2:
                try:
                    j = int(parts[1]) - 1
                    if 0 <= j < n:
                        excluded_preview.add(j)
                        print(f"Excluded preview #{j + 1}", flush=True)
                except ValueError:
                    print("Usage: x N", flush=True)
                continue
            if cmd == "i" and len(parts) >= 2:
                try:
                    j = int(parts[1]) - 1
                    if 0 <= j < n:
                        excluded_preview.discard(j)
                        print(f"Included preview #{j + 1}", flush=True)
                except ValueError:
                    print("Usage: i N", flush=True)
                continue
            print("Unknown command. Try l, p 3, x 3, i 3, a, e, q", flush=True)
    finally:
        if tmp_root and os.path.isdir(tmp_root):
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass
        gc.collect()


def compute_auto_trim_chunks(
    audio_path: str,
    *,
    model_name: str = "medium",
    language: str | None = None,
    diarize: bool = True,
    min_speakers: int = 1,
    max_speakers: int = 8,
    batch_size: int = 8,
    hf_token: str | None = None,
    diarize_model: str | None = None,
    model_dir: str | None = None,
    verbose: bool = False,
    smooth_speakers: bool = True,
    smooth_half_window_sec: float = 0.45,
    max_island_words: int = 2,
    max_island_sec: float = 0.18,
    smart_pause_merge: bool = True,
    pause_hard_split_sec: float = 3.6,
    pause_max_glue_sec: float = 2.5,
    pause_breath_sec: float = 0.42,
    pause_discourse_sec: float = 1.85,
    pause_period_lowercase_sec: float = 1.25,
    voice_pause_gate: bool = True,
    voice_embedding_model: str = "auto",
    voice_compare_window_sec: float = 0.45,
    voice_gate_above_gap_sec: float | None = None,
    voice_min_cosine: float = 0.55,
    voice_veto_cosine: float = 0.38,
    voice_on_cpu: bool = False,
    sentence_end_tail_sec: float = 0.32,
    sentence_tail_min_gap_sec: float = 0.025,
    phase_callback: Callable[[str], None] | None = None,
    return_aligned_words: bool = False,
) -> list[tuple[float, float, str, str]] | tuple[list[tuple[float, float, str, str]], list[dict]]:
    """
    Run WhisperX + optional diarization + chunking; does not write output files.

    If ``return_aligned_words`` is True, returns ``(chunks, words)`` where ``words`` is the
    aligned word list (``start`` / ``end`` / ``word`` / ``speaker``) for UI transcript updates.
    """

    def _phase(msg: str) -> None:
        if phase_callback is None:
            return
        try:
            phase_callback(msg)
        except Exception:
            pass

    from whisperx.alignment import align, load_align_model
    from whisperx.asr import load_model
    from whisperx.audio import SAMPLE_RATE, load_audio
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    audio_path = str(Path(audio_path).resolve())
    _phase(f"Reading audio: {Path(audio_path).name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    audio = load_audio(audio_path)
    _phase("Loading Whisper model (first run may download weights)…")
    # Pyannote VAD (default) expects Hub auth like diarization. Without a token and without
    # diarization, use Silero VAD so AI trim / ``--no-diarize`` still runs.
    vad_method = "pyannote"
    vad_options: dict[str, float | int] = {"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363}
    if not diarize and not hf_token:
        vad_method = "silero"
        _phase("No HF token — using Silero VAD (no pyannote VAD)…")
    asr = load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        asr_options=None,
        vad_method=vad_method,
        vad_options=vad_options,
        task="transcribe",
        download_root=model_dir,
        threads=4,
        use_auth_token=hf_token,
    )

    if verbose:
        print("Transcribing...", flush=True)
    _phase("Transcribing speech with WhisperX (often the slowest step)…")
    result = asr.transcribe(audio, batch_size=batch_size, print_progress=verbose)
    lang = result.get("language") or language or "en"

    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print("Aligning...", flush=True)
    _phase("Loading forced-alignment model…")
    align_model, align_meta = load_align_model(
        lang, device, model_name=None, model_dir=model_dir, model_cache_only=False
    )
    if align_model is not None and len(result.get("segments") or []) > 0:
        _phase("Aligning words to the waveform for accurate cut times…")
        result = align(
            result["segments"],
            align_model,
            align_meta,
            audio,
            device,
            interpolate_method="nearest",
            return_char_alignments=False,
            print_progress=verbose,
        )
    result["language"] = lang

    del align_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if diarize:
        if not hf_token:
            raise PslicerUserError(
                "Diarization requires HF_TOKEN (or huggingface-cli login). "
                "Use --no-diarize to export sentence chunks without speaker separation."
            )
        if verbose:
            print("Diarizing (pyannote)...", flush=True)
        _phase("Running speaker diarization (pyannote)…")
        dia = DiarizationPipeline(
            model_name=diarize_model, token=hf_token, device=device, cache_dir=model_dir
        )
        di_segments = dia(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=False,
        )
        assign_word_speakers(di_segments, result, fill_nearest=True)
        _phase("Mapping diarization labels onto each word…")
        del di_segments
        del dia
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    words = _collect_aligned_words(result)
    if not words:
        raise PslicerUserError("No aligned words — cannot auto-trim (try another model or check audio).")

    if smooth_speakers and diarize:
        if verbose:
            print(
                f"Smoothing speakers (+/-{smooth_half_window_sec}s, islands <={max_island_words}w / {max_island_sec}s)...",
                flush=True,
            )
        _phase("Smoothing short spurious speaker flips…")
        apply_contextual_speaker_smoothing(
            words,
            half_window_sec=smooth_half_window_sec,
            max_island_words=max_island_words,
            max_island_sec=max_island_sec,
            preserve_raw=True,
        )

    audio_dur_sec = float(len(audio)) / float(SAMPLE_RATE)
    _phase("Building sentence-level clip boundaries…")
    chunks = iter_sentence_speaker_chunks(
        words,
        audio_duration_sec=audio_dur_sec,
        sentence_end_tail_sec=sentence_end_tail_sec,
        min_gap_before_next_sec=sentence_tail_min_gap_sec,
    )
    if smart_pause_merge and diarize:
        n_before = len(chunks)
        voice_cmp: ChunkVoiceComparator | None = None
        if voice_pause_gate:
            emb_device = torch.device("cpu" if voice_on_cpu else device)
            if verbose:
                print(
                    f"Voice pause gate ({voice_embedding_model}, window={voice_compare_window_sec}s, "
                    f"device={emb_device.type})...",
                    flush=True,
                )
            _phase(f"Loading speaker embeddings for pause merge ({voice_embedding_model})…")
            voice_cmp, _voice_used = _build_voice_comparator(
                audio,
                SAMPLE_RATE,
                embedding_model=voice_embedding_model,
                device=emb_device,
                hf_token=hf_token,
                model_dir=model_dir,
                window_sec=voice_compare_window_sec,
                verbose=verbose,
            )
            if voice_cmp is not None:
                if verbose:
                    print("  Speaker voice profile map (per diarization label)...", flush=True)
                _phase("Computing per-speaker voice profiles…")
                voice_cmp.precompute_speaker_profiles(chunks)
            del audio
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            _phase("Merging same-speaker clips across short pauses (smart gate)…")
            chunks = merge_chunks_smart_same_speaker_pauses(
                chunks,
                hard_split_gap_sec=pause_hard_split_sec,
                max_glue_gap_sec=pause_max_glue_sec,
                breath_gap_sec=pause_breath_sec,
                discourse_gap_sec=pause_discourse_sec,
                period_glue_if_lowercase_sec=pause_period_lowercase_sec,
                voice_compare=voice_cmp,
                voice_gate_above_gap_sec=voice_gate_above_gap_sec,
                voice_min_cosine=voice_min_cosine,
                voice_veto_cosine=voice_veto_cosine,
            )
        finally:
            if voice_cmp is not None:
                del voice_cmp
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if verbose:
            print(f"Smart pause merge: {n_before} -> {len(chunks)} clips", flush=True)
    elif smart_pause_merge and not diarize and verbose:
        print(
            "Smart pause merge skipped (needs diarization to know same-speaker pauses).",
            flush=True,
        )

    _phase(f"Ready: {len(chunks)} clip(s). Opening preview…")
    if return_aligned_words:
        return chunks, words
    return chunks


def run_auto_trim(
    audio_path: str,
    out_dir: str,
    *,
    model_name: str = "medium",
    language: str | None = None,
    diarize: bool = True,
    min_speakers: int = 1,
    max_speakers: int = 8,
    batch_size: int = 8,
    padding_ms: int = 120,
    hf_token: str | None = None,
    diarize_model: str | None = None,
    model_dir: str | None = None,
    verbose: bool = False,
    smooth_speakers: bool = True,
    smooth_half_window_sec: float = 0.45,
    max_island_words: int = 2,
    max_island_sec: float = 0.18,
    manifest_path: str | Path | None = None,
    smart_pause_merge: bool = True,
    pause_hard_split_sec: float = 3.6,
    pause_max_glue_sec: float = 2.5,
    pause_breath_sec: float = 0.42,
    pause_discourse_sec: float = 1.85,
    pause_period_lowercase_sec: float = 1.25,
    voice_pause_gate: bool = True,
    voice_embedding_model: str = "auto",
    voice_compare_window_sec: float = 0.45,
    voice_gate_above_gap_sec: float | None = None,
    voice_min_cosine: float = 0.55,
    voice_veto_cosine: float = 0.38,
    voice_on_cpu: bool = False,
    sentence_end_tail_sec: float = 0.32,
    sentence_tail_min_gap_sec: float = 0.025,
    preview_before_export: bool = False,
    preview_play_all: bool = False,
    preview_input: Callable[[], str] | None = None,
    phase_callback: Callable[[str], None] | None = None,
) -> list[str]:
    out_dir = str(Path(out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)

    chunks = compute_auto_trim_chunks(
        audio_path,
        model_name=model_name,
        language=language,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        batch_size=batch_size,
        hf_token=hf_token,
        diarize_model=diarize_model,
        model_dir=model_dir,
        verbose=verbose,
        smooth_speakers=smooth_speakers,
        smooth_half_window_sec=smooth_half_window_sec,
        max_island_words=max_island_words,
        max_island_sec=max_island_sec,
        smart_pause_merge=smart_pause_merge,
        pause_hard_split_sec=pause_hard_split_sec,
        pause_max_glue_sec=pause_max_glue_sec,
        pause_breath_sec=pause_breath_sec,
        pause_discourse_sec=pause_discourse_sec,
        pause_period_lowercase_sec=pause_period_lowercase_sec,
        voice_pause_gate=voice_pause_gate,
        voice_embedding_model=voice_embedding_model,
        voice_compare_window_sec=voice_compare_window_sec,
        voice_gate_above_gap_sec=voice_gate_above_gap_sec,
        voice_min_cosine=voice_min_cosine,
        voice_veto_cosine=voice_veto_cosine,
        voice_on_cpu=voice_on_cpu,
        sentence_end_tail_sec=sentence_end_tail_sec,
        sentence_tail_min_gap_sec=sentence_tail_min_gap_sec,
        phase_callback=phase_callback,
    )

    if preview_before_export:
        chunks, ok = interactive_preview_trims(
            audio_path,
            chunks,
            padding_ms=padding_ms,
            stem=Path(audio_path).stem,
            preview_play_all=preview_play_all,
            verbose=verbose,
            input_fn=preview_input,
        )
        if not ok:
            if verbose:
                print("Preview aborted; no files written to output directory.", flush=True)
            return []
        if not chunks:
            if verbose:
                print("No clips left after preview; nothing exported.", flush=True)
            return []

    if verbose:
        print(f"Exporting {len(chunks)} clips...", flush=True)

    return export_wav_clips(
        audio_path,
        out_dir,
        chunks,
        stem=Path(audio_path).stem,
        padding_ms=padding_ms,
        verbose=verbose,
        manifest_path=manifest_path,
    )


def main() -> int:
    if os.name == "nt":
        register_windows_torchcodec_dll_paths()
    _preconfigure_cuda_for_pyannote()

    p = argparse.ArgumentParser(
        description="Auto-trim WAV at sentences + speakers (WhisperX).",
        epilog=(
            "Terminal preview: python pslicer.py file.wav --preview (optional --preview-play).\n"
            "GUI preview + waveform: run python slicer.py and use the AI trim… button.\n"
            "Without --preview, clips are written to --out immediately after processing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Input .wav path (required unless you only pass -h)",
    )
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "pslicer_out"),
        help="Output directory for trimmed WAVs",
    )
    p.add_argument("--model", default="medium", help="WhisperX model name (e.g. medium, large-v3-turbo)")
    p.add_argument("--language", default=None, help="ISO language code (default: auto)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--padding-ms", type=int, default=120, help="Lead/trail padding per clip")
    p.add_argument(
        "--sentence-tail-sec",
        type=float,
        default=0.32,
        help="Extra time after sentence-final aligned word (capped by next word / EOF)",
    )
    p.add_argument(
        "--sentence-tail-min-gap",
        type=float,
        default=0.025,
        help="Minimum gap left before the next word when extending sentence tail",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Interactive preview in a temp dir (list/play/exclude) before writing --out",
    )
    p.add_argument(
        "--preview-play",
        action="store_true",
        help="With --preview, play each clip once via OS player before command mode",
    )
    p.add_argument("--min-speakers", type=int, default=1)
    p.add_argument("--max-speakers", type=int, default=8)
    p.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip pyannote (no speaker separation; all chunks SPEAKER_UNKNOWN)",
    )
    p.add_argument("--diarize-model", default=None, help="HF model id for DiarizationPipeline")
    p.add_argument("--model-dir", default=None, help="Download cache directory for models")
    p.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable contextual speaker smoothing (raw pyannote labels)",
    )
    p.add_argument(
        "--smooth-window",
        type=float,
        default=0.45,
        help="Half-width (seconds) for duration-weighted speaker vote",
    )
    p.add_argument(
        "--island-words",
        type=int,
        default=2,
        help="Max words in a spurious speaker island to merge into neighbors",
    )
    p.add_argument(
        "--island-sec",
        type=float,
        default=0.18,
        help="Max duration (seconds) for a spurious island",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="Write JSON manifest of chunk timestamps and speakers (for benchmarks)",
    )
    p.add_argument(
        "--no-smart-pause-merge",
        action="store_true",
        help="Disable merging same-speaker clips across short pauses (sentence split only)",
    )
    p.add_argument(
        "--pause-hard-split",
        type=float,
        default=3.6,
        help="Gap (sec) at or above: never merge across pause (scene break)",
    )
    p.add_argument(
        "--pause-max-glue",
        type=float,
        default=2.5,
        help="Max gap (sec) to glue when prior chunk lacks strong sentence end",
    )
    p.add_argument(
        "--pause-breath",
        type=float,
        default=0.42,
        help="Gaps (sec) up to this: treat as breath, merge same speaker",
    )
    p.add_argument(
        "--pause-discourse",
        type=float,
        default=1.85,
        help="Max gap (sec) to merge when next chunk starts with discourse continuation",
    )
    p.add_argument(
        "--pause-period-lowercase",
        type=float,
        default=1.25,
        help="Max gap (sec) to merge full stop + lowercase continuation",
    )
    p.add_argument(
        "--no-voice-pause-gate",
        action="store_true",
        help="Skip speaker-embedding tail/head check on pause merges (faster, text/timing only)",
    )
    p.add_argument(
        "--voice-embedding-model",
        default="auto",
        help=(
            "Speaker embedding: auto = WeSpeaker ONNX then pyannote/embedding; "
            "or set one model id explicitly"
        ),
    )
    p.add_argument(
        "--voice-on-cpu",
        action="store_true",
        help="Run speaker embedding on CPU (slower, less GPU VRAM during pause merge)",
    )
    p.add_argument(
        "--voice-compare-window",
        type=float,
        default=0.45,
        help="Seconds of audio at chunk boundary to embed (tail vs head)",
    )
    p.add_argument(
        "--voice-gate-above-gap",
        type=float,
        default=None,
        help="Apply embedding gate only when gap exceeds this (sec); default: same as --pause-breath",
    )
    p.add_argument(
        "--voice-min-cosine",
        type=float,
        default=0.55,
        help=(
            "Min blended voice score to allow merge (boundary + chunk centroids + cross crops + "
            "speaker profile); strong text cues can still allow marginal merges"
        ),
    )
    p.add_argument(
        "--voice-veto-cosine",
        type=float,
        default=0.38,
        help="Below this cosine, block merge across a gated gap",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if not args.audio:
        print(
            "No input file given. pslicer is a command-line tool: pass a WAV path, then optional flags.\n"
            '\n  Review trims in this console before export:\n'
            f'    python "{Path(__file__).name}" your_audio.wav --preview\n'
            "\n  Same, and play each clip once automatically:\n"
            f'    python "{Path(__file__).name}" your_audio.wav --preview --preview-play\n'
            "\n  See all options:\n"
            f'    python "{Path(__file__).name}" -h\n',
            file=sys.stderr,
        )
        return 2

    if not os.path.isfile(args.audio):
        print("Input not found:", args.audio, file=sys.stderr)
        return 1

    hf_token, hf_src = (None, None)
    if not args.no_diarize:
        hf_token, hf_src = _resolve_hf_token()
        if not hf_token:
            print(
                "No Hugging Face token found. Diarization needs one:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-community-1 (accept terms)\n"
                "  set HF_TOKEN=hf_...   or   venv\\Scripts\\hf.exe auth login\n"
                "Or run with --no-diarize.\n",
                file=sys.stderr,
            )
            return 1
        if args.verbose:
            print(f"Hugging Face token: {hf_src}", flush=True)

    try:
        paths = run_auto_trim(
            args.audio,
            args.out,
            model_name=args.model,
            language=args.language,
            diarize=not args.no_diarize,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            batch_size=args.batch_size,
            padding_ms=args.padding_ms,
            hf_token=hf_token,
            diarize_model=args.diarize_model,
            model_dir=args.model_dir,
            verbose=args.verbose,
            smooth_speakers=not args.no_smooth,
            smooth_half_window_sec=args.smooth_window,
            max_island_words=args.island_words,
            max_island_sec=args.island_sec,
            manifest_path=args.manifest,
            smart_pause_merge=not args.no_smart_pause_merge,
            pause_hard_split_sec=args.pause_hard_split,
            pause_max_glue_sec=args.pause_max_glue,
            pause_breath_sec=args.pause_breath,
            pause_discourse_sec=args.pause_discourse,
            pause_period_lowercase_sec=args.pause_period_lowercase,
            voice_pause_gate=not args.no_voice_pause_gate,
            voice_embedding_model=args.voice_embedding_model,
            voice_compare_window_sec=args.voice_compare_window,
            voice_gate_above_gap_sec=args.voice_gate_above_gap,
            voice_min_cosine=args.voice_min_cosine,
            voice_veto_cosine=args.voice_veto_cosine,
            voice_on_cpu=args.voice_on_cpu,
            sentence_end_tail_sec=args.sentence_tail_sec,
            sentence_tail_min_gap_sec=args.sentence_tail_min_gap,
            preview_before_export=args.preview,
            preview_play_all=args.preview_play,
        )
    except PslicerUserError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"Done. {len(paths)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
