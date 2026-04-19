"""
Windows: TorchCodec needs extra DLL search paths before WhisperX imports pyannote.

CUDA PyTorch DLLs live under torch\\lib; FFmpeg *shared* builds ship avcodec-*.dll next to
ffmpeg.exe. os.add_dll_directory is required (PATH alone is not enough on Python 3.8+).

Optional: set FFMPEG_SHARED_BIN to the directory containing ffmpeg.exe and avcodec-*.dll.
Install example: winget install BtbN.FFmpeg.GPL.Shared.7.1
"""

from __future__ import annotations

import os
import sys


def _dir_has_ffmpeg_shared_libs(dir_path: str) -> bool:
    try:
        names = os.listdir(dir_path)
    except OSError:
        return False
    if not any(n.lower() == "ffmpeg.exe" for n in names):
        return False
    return any(n.startswith("avcodec") and n.endswith(".dll") for n in names)


def register_windows_torchcodec_dll_paths() -> bool:
    """Return True if FFmpeg shared bin was found and registered."""
    if os.name != "nt":
        return True

    import torch

    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

    candidates: list[str] = []
    env = os.environ.get("FFMPEG_SHARED_BIN")
    if env:
        candidates.append(os.path.abspath(env.strip()))

    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        candidates.append(os.path.dirname(os.path.abspath(exe)))

    seen: set[str] = set()
    for dir_path in candidates:
        if not dir_path or dir_path in seen or not os.path.isdir(dir_path):
            continue
        seen.add(dir_path)
        if _dir_has_ffmpeg_shared_libs(dir_path):
            os.add_dll_directory(dir_path)
            return True

    for dir_path in os.environ.get("PATH", "").split(os.pathsep):
        dir_path = dir_path.strip('"')
        if not dir_path or dir_path in seen:
            continue
        seen.add(dir_path)
        if _dir_has_ffmpeg_shared_libs(dir_path):
            os.add_dll_directory(dir_path)
            return True

    return False


def _preconfigure_cuda_for_pyannote() -> None:
    """Match pyannote's preferred TF32 state before it runs fix_reproducibility (avoids ReproducibilityWarning)."""
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main() -> None:
    if os.name == "nt" and not register_windows_torchcodec_dll_paths():
        print(
            "TorchCodec: no FFmpeg *shared* bin on PATH (folder with ffmpeg.exe + avcodec-*.dll). "
            "PyAnnote may warn. Install e.g.  winget install BtbN.FFmpeg.GPL.Shared.7.1  "
            "or set FFMPEG_SHARED_BIN to that bin directory.",
            file=sys.stderr,
        )

    _preconfigure_cuda_for_pyannote()

    sys.argv = ["whisperx"] + sys.argv[1:]
    from whisperx.__main__ import cli

    cli()


if __name__ == "__main__":
    main()
