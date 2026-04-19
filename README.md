# Voice Engine — desktop Tk trimmer & WhisperX pslicer

Python toolkit for **waveform trimming** (desktop Tk app) and **WhisperX-based auto-trim** with optional **speaker diarization**, export to WAV, and GUI preview.

This layout is suitable for **publishing as its own repository** (copy this folder or use it as a submodule). Paths below assume you are in the repo root (`voice_engine/`).

## Core layout

| Path | Role |
|------|------|
| `slicer.py` | **Voice Engine** — main Tk + Matplotlib trimmer, optional **AI trim…** (pslicer preview). |
| `pslicer.py` | **Auto-trim library** — WhisperX transcribe/align/diarize, sentence chunks, smart pause merge, CLI + export. |
| `whisperx_windows_entry.py` | Windows DLL / CUDA helpers for pyannote / WhisperX (imported by the stack). |
| `test_*.py` | Unit / stress tests (no GPU required for most). |
| `extras/` | **Optional** scripts — alternate UIs, demos, benchmarks, workspace utilities. See `extras/README.md`. |

## Quick start

1. **Python 3.10+** (3.12 used in development), **64-bit**, Windows or Linux.
2. Create a venv and install dependencies for the main GUI: `tkinter` (stdlib), `matplotlib`, `librosa`, `pydub`, `pygame`, `numpy`, etc.
3. **AI trim** (optional): in the **same venv**, install WhisperX and its stack — `pip install -r requirements-pslicer.txt`. If `torch` fails to resolve, install a CPU/CUDA wheel from [PyTorch Get Started](https://pytorch.org/get-started/locally/) first, then rerun that pip command. Always launch `slicer.py` with that venv’s `python` so `import torch` and `import pslicer` succeed.
4. **Hugging Face token** for **speaker diarization**: use **Settings…** in the app to paste and save a token (stored under your user profile, e.g. `%LOCALAPPDATA%\VoiceEngine\settings.json` on Windows — applied immediately to `HF_TOKEN` for this process). Alternatively set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, or `huggingface-cli login`, and accept the **pyannote** model terms (e.g. [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)). Environment variables take precedence over the saved file on startup. Without any token, **AI trim** can still run if you choose **Continue without diarization** (sentence cuts + Silero VAD; no multi-speaker labels).

### Voice Engine (GUI)

```text
python slicer.py
python slicer.py "C:\path\to\file.wav"
```

If the legacy `SanctumCore/voice_assets/…` folders exist on your machine, **Open…** and the initial export folder use those paths; otherwise the app defaults to the repo directory and creates **`exports/`** here for WAV output (you can still pick any folder in the UI).

### pslicer (CLI auto-trim)

```text
python pslicer.py recording.wav --out .\exports --model medium
```

Use `python pslicer.py --help` for flags (diarization, pause merge, voice gate, preview, etc.).

## Tests

From repo root, with venv activated:

```text
python test_pslicer_separation.py
python test_pslicer_preview.py
python test_pslicer_diagnostics.py
python test_pslicer_swallow_neighbors.py
python test_slicer_pslicer_stress.py
python test_voice_engine_settings.py
```

## Publishing to a new repository

This directory is a **standalone git repo** (`requirements.txt` for core deps, `.gitignore` for venv and artifacts).

**Create `voice-engine` on GitHub and push** (after [GitHub CLI](https://cli.github.com/) login):

```text
cd voice_engine
gh auth login
gh repo create voice-engine --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if you prefer. If the GitHub repo already exists (empty), add the remote and push:

```text
git remote add origin https://github.com/gmater/voice-engine.git
git push -u origin main
```

Set **`git config user.email`** / **`user.name`** in this repo (or globally) before amending if you do not want the placeholder author on the first commit.

Do **not** commit secrets (HF tokens, etc.). Optional: trim **`extras/`** before publishing if you only want the main app + pslicer.

## License / attribution

Add your own `LICENSE` when publishing; attribute **WhisperX**, **pyannote**, **faster-whisper**, and other upstream licenses as required by your dependency set.
