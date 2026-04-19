"""
Assemble a master reference WAV from many clips, and build Kokoro voice tensors.

Kokoro-82M ships decoder weights only (no reference/style encoder in the public
checkpoint). Voice packs are precomputed FloatTensor files shaped [510, 1, 256]
(Hugging Face) or the same arrays inside an .npz (kokoro-onnx voices*.bin).

This script cannot derive a voice pack from raw WAV alone. It can:
  - Concatenate / trim / resample your harvested speech to a single reference WAV.
  - Blend existing hexgrad/Kokoro-82M voices (same idea as KPipeline's comma blend)
    and save torch .pt or a one-voice .npz for kokoro-onnx.

Usage examples:
  python kokoro_voice_pack.py --blend bm_george:0.5,bf_emma:0.5 --output jarvis.pt
  python kokoro_voice_pack.py --blend am_michael:1 --output jarvis.npz --npz-key jarvis
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from pydub import AudioSegment


DEFAULT_INPUT_DIR = r"C:\AI\SanctumCore\voice_assets\Pure_Jarvis_Audio"
DEFAULT_MASTER_WAV = r"C:\AI\SanctumCore\voice_assets\jarvis_reference.wav"
DEFAULT_REPO = "hexgrad/Kokoro-82M"
VOICE_RANK = 3  # [510, 1, 256]


def _parse_blend(spec: str) -> list[tuple[str, float]]:
    """Parse 'af_heart:0.3,bm_george:0.7' into weighted voice names."""
    out: list[tuple[str, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Blend part must be name:weight, got {part!r}")
        name, w = part.rsplit(":", 1)
        out.append((name.strip(), float(w)))
    if not out:
        raise ValueError("Empty --blend")
    s = sum(w for _, w in out)
    if s <= 0:
        raise ValueError("Blend weights must sum to a positive value")
    return [(n, w / s) for n, w in out]


def _load_voice_tensor(repo_id: str, voice: str) -> torch.Tensor:
    if voice.endswith(".pt"):
        path = voice
    else:
        path = hf_hub_download(repo_id=repo_id, filename=f"voices/{voice}.pt")
    t = torch.load(path, map_location="cpu", weights_only=True)
    if t.ndim != VOICE_RANK or t.shape[1] != 1 or t.shape[2] != 256:
        raise ValueError(f"Unexpected voice tensor shape for {voice}: {tuple(t.shape)}")
    return t


def _blend_voices(repo_id: str, weighted: Sequence[tuple[str, float]]) -> torch.Tensor:
    acc: torch.Tensor | None = None
    for name, w in weighted:
        t = _load_voice_tensor(repo_id, name)
        acc = t * w if acc is None else acc + t * w
    assert acc is not None
    return acc


def _build_master_wav(input_dir: str, master_wav: str, max_ms: int) -> float:
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        raise SystemExit(
            f"The folder {input_dir} did not exist; it was created empty. "
            "Add isolated .wav clips, then run again."
        )

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".wav"))
    if not files:
        raise SystemExit(f"No .wav reagents found in {input_dir}.")

    combined = AudioSegment.empty()
    for f in files:
        print(f" -> Injecting: {f}")
        combined += AudioSegment.from_wav(os.path.join(input_dir, f))

    if len(combined) > max_ms:
        print(
            f"\nAudio exceeds {max_ms/1000:.0f} seconds. "
            f"Truncating to the first {max_ms/1000:.0f}s window..."
        )
        combined = combined[:max_ms]

    if combined.channels > 1:
        combined = combined.set_channels(1)
    if combined.frame_rate != 24_000:
        print(f"Resampling reference clip {combined.frame_rate} Hz -> 24000 Hz for Kokoro alignment...")
        combined = combined.set_frame_rate(24_000)

    os.makedirs(os.path.dirname(os.path.abspath(master_wav)) or ".", exist_ok=True)
    combined.export(master_wav, format="wav")
    duration_s = len(combined) / 1000.0
    print(f"Master reference saved: {master_wav} ({duration_s:.2f} seconds)")
    return duration_s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--master-wav", default=DEFAULT_MASTER_WAV)
    p.add_argument("--max-ms", type=int, default=30_000)
    p.add_argument("--skip-wav", action="store_true", help="Do not rebuild master WAV.")
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument(
        "--blend",
        required=True,
        help="Weighted Kokoro voices, e.g. bm_george:0.6,bf_emma:0.4 (names from VOICES.md).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to .pt (torch) or .npz / .bin (kokoro-onnx voices file format).",
    )
    p.add_argument(
        "--npz-key",
        default="custom",
        help="Array name inside .npz when saving for kokoro-onnx (default: custom).",
    )
    args = p.parse_args()

    print("Step 1: Assembling the biological acoustic record...")
    if not args.skip_wav:
        _build_master_wav(args.input_dir, args.master_wav, args.max_ms)
    else:
        print("Skipping WAV export (--skip-wav).")

    print(
        "\nStep 2: Forging Kokoro voice tensor (blended from published voice packs)...\n"
        "Note: Kokoro's public weights do not include a reference encoder; "
        "harvested WAV is exported for your archive or for other TTS stacks."
    )

    weighted = _parse_blend(args.blend)
    for n, w in weighted:
        print(f" -> Voice {n} (weight {w:.4f})")
    pack = _blend_voices(args.repo_id, weighted)

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    ext = os.path.splitext(out)[1].lower()
    if ext == ".pt":
        torch.save(pack, out)
    elif ext in (".npz", ".bin"):
        arr = pack.detach().cpu().numpy().astype(np.float32)
        key = re.sub(r"[^0-9a-zA-Z_]", "_", args.npz_key)
        np.savez_compressed(out, **{key: arr})
    else:
        raise SystemExit("Use --output ending in .pt, .npz, or .bin (single-voice kokoro-onnx bundle).")

    print(f"\nSUCCESS! Voice pack written: {out}")
    print("Load in Python Kokoro:  voice_tensor = torch.load(path, weights_only=True)")
    key = re.sub(r"[^0-9a-zA-Z_]", "_", args.npz_key)
    print(
        "Load in kokoro-onnx:  d = numpy.load(path); "
        f"style = d[{key!r}]  # or pass that array as voice= to Kokoro.create"
    )


if __name__ == "__main__":
    main()
