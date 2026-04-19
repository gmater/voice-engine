"""Build a short two-voice WAV (Edge TTS) for pslicer end-to-end demos."""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

import edge_tts
import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None


async def _save_line(text: str, voice: str, path: Path) -> None:
    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=_REPO / "pslicer_demo_mixed.wav",
    )
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    if librosa is None:
        print("librosa required for mp3 decode", file=sys.stderr)
        return 1

    lines = [
        ("Hello. I am the first speaker in this mixed recording.", "en-US-JennyNeural"),
        ("Thanks. I am the second voice. We alternate sentences.", "en-US-GuyNeural"),
        ("Good. That should give the diarizer two clear speakers.", "en-US-JennyNeural"),
        ("Agreed. End of our short test dialogue.", "en-US-GuyNeural"),
    ]

    tmp = Path(tempfile.mkdtemp(prefix="pslicer_demo_"))
    try:
        mp3s: list[Path] = []
        for i, (text, voice) in enumerate(lines):
            p = tmp / f"line_{i}.mp3"
            asyncio.run(_save_line(text, voice, p))
            mp3s.append(p)

        chunks: list[np.ndarray] = []
        gap = np.zeros(int(0.35 * args.sr), dtype=np.float32)
        for p in mp3s:
            y, _ = librosa.load(str(p), sr=args.sr, mono=True)
            y = y.astype(np.float32)
            mx = np.max(np.abs(y)) or 1.0
            y = (y / mx * 0.95).astype(np.float32)
            if chunks:
                chunks.append(gap.copy())
            chunks.append(y)
        mix = np.concatenate(chunks)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.out), mix, args.sr, subtype="PCM_16")
        print("Wrote", args.out, f"({len(mix)/args.sr:.1f}s)")
        return 0
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
