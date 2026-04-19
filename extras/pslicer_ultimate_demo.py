"""
Ultimate pslicer demo: mixed two-voice WAV → (optional) diarized clips → plot + play.

Requires HF_TOKEN / huggingface-cli login for real speaker separation (pyannote).
Without a token: still builds/plays the mixed demo and shows what to configure.

Usage (from ``voice_engine`` repo root):

  set HF_TOKEN=hf_...
  venv\\Scripts\\python.exe extras\\pslicer_ultimate_demo.py

  venv\\Scripts\\python.exe extras\\pslicer_ultimate_demo.py --model base --skip-play
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_EXTRA = Path(__file__).resolve().parent
os.chdir(_REPO)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pslicer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import winsound
except ImportError:
    winsound = None


def _wav_to_mono_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _play_wav(path: Path) -> None:
    if winsound is None:
        print("(winsound unavailable)", path)
        return
    print("Playing:", path.name, flush=True)
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def _open_in_explorer(dir_path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(dir_path)  # noqa: S606
    else:
        subprocess.Popen(["xdg-open", str(dir_path)])  # noqa: S603,S607


def _plot_overview(
    original: Path,
    clip_paths: list[Path],
    out_png: Path,
    titles: list[str] | None = None,
) -> None:
    if plt is None:
        print("matplotlib not installed; skipping plot. pip install matplotlib", flush=True)
        return
    n = 1 + len(clip_paths)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    y0, sr0 = _wav_to_mono_float(original)
    t0 = np.arange(len(y0)) / sr0
    axes[0].plot(t0, y0, color="#38bdf8", linewidth=0.35)
    axes[0].set_title(f"Original (mixed): {original.name}")
    axes[0].set_ylabel("amp")
    axes[0].set_xlim(0, t0[-1] if len(t0) else 1)

    for i, p in enumerate(clip_paths):
        ax = axes[i + 1]
        y, sr = _wav_to_mono_float(p)
        t = np.arange(len(y)) / sr
        ax.plot(t, y, color="#34d399", linewidth=0.35)
        title = titles[i] if titles and i < len(titles) else p.name
        ax.set_title(title)
        ax.set_ylabel("amp")
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, t[-1] if len(t) else 1)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print("Saved overview:", out_png, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pslicer ultimate demo (mixed voices → separated clips).")
    ap.add_argument("--mixed", type=Path, default=_REPO / "pslicer_demo_mixed.wav")
    ap.add_argument("--out", type=Path, default=_REPO / "pslicer_demo_out")
    ap.add_argument("--model", default="base", help="WhisperX model (base is faster for demos)")
    ap.add_argument("--skip-synth", action="store_true", help="Do not run Edge TTS if mixed wav missing")
    ap.add_argument("--skip-play", action="store_true")
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument("--open-folder", action="store_true", help="Open output folder in Explorer when done")
    args = ap.parse_args()

    if not args.mixed.is_file():
        if args.skip_synth:
            print("Missing mixed WAV:", args.mixed, file=sys.stderr)
            return 1
        print("Synthesizing two-voice demo (Edge TTS)…", flush=True)
        r = subprocess.run([sys.executable, str(_EXTRA / "pslicer_demo_prepare.py"), "-o", str(args.mixed)])
        if r.returncode != 0:
            return r.returncode

    hf, src = pslicer._resolve_hf_token()
    if not hf:
        print(
            "\n*** No Hugging Face token — cannot run pyannote diarization in this session. ***\n"
            "  1) Accept: https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  2) Token: https://huggingface.co/settings/tokens\n"
            "  3) cmd:  set HF_TOKEN=hf_...\n"
            "     or:   venv\\Scripts\\hf.exe auth login\n"
            "  Then re-run:  python pslicer_ultimate_demo.py\n",
            flush=True,
        )
        if not args.skip_plot and plt is not None:
            _plot_overview(args.mixed, [], _REPO / "pslicer_demo_overview.png")
        if not args.skip_play and winsound:
            print("Playing original (mixed) only…\n", flush=True)
            _play_wav(args.mixed)
        if args.open_folder:
            _open_in_explorer(args.mixed.parent)
        return 2

    print(f"Hugging Face: {src}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    print("Running pslicer (transcribe + align + diarize + smooth + export)…", flush=True)
    exported = pslicer.run_auto_trim(
        str(args.mixed),
        str(args.out),
        model_name=args.model,
        language="en",
        diarize=True,
        min_speakers=1,
        max_speakers=4,
        batch_size=8,
        padding_ms=120,
        hf_token=hf,
        verbose=True,
        smooth_speakers=True,
    )
    clips = [Path(p) for p in exported]
    print(f"\nExported {len(clips)} clip(s) to {args.out}", flush=True)
    for p in clips:
        print(" ", p.name)

    titles = [f"{p.name}" for p in clips]
    if not args.skip_plot:
        _plot_overview(args.mixed, clips, _REPO / "pslicer_demo_overview.png", titles=titles)
        try:
            os.startfile(_REPO / "pslicer_demo_overview.png")  # noqa: S606
        except OSError:
            pass

    if not args.skip_play and winsound:
        print("\n--- Playback: original (mixed) ---", flush=True)
        _play_wav(args.mixed)
        y_mix, sr_mix = _wav_to_mono_float(args.mixed)
        time.sleep(min(len(y_mix) / sr_mix + 0.5, 120))
        for p in clips:
            print(f"\n--- Clip: {p.name} ---", flush=True)
            _play_wav(p)
            y, sr = _wav_to_mono_float(p)
            time.sleep(len(y) / sr + 0.35)

    if args.open_folder:
        _open_in_explorer(args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
