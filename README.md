# Voice Engine

Voice Engine is a **desktop waveform trimmer** for Windows and Linux: open a WAV, adjust a time range on the waveform, preview, and export trimmed clips. It includes an optional **AI trim** workflow powered by **WhisperX** and the bundled **pslicer** library—automatic sentence-boundary cuts, optional **speaker diarization**, and a preview window before you commit changes.

The same repository ships **pslicer** as a **command-line** tool for batch auto-trim without the GUI.

---

## Features

- **Manual trim** — Matplotlib span selection, typed start/end times, zoom, playback (pygame).
- **Export** — WAV to a folder you choose; sensible default filenames (speaker/transcript hints when using AI trim preview).
- **AI trim (optional)** — Transcribe, align, optional diarization, chunk on sentences/speakers; edit in a list + waveform preview, then apply or export from the dialog.
- **Settings** — Persisted options (e.g. Hugging Face token path for diarization) via the in-app **Settings…** dialog.

---

## System requirements

| Item | Notes |
|------|--------|
| **OS** | **Windows 10/11** (64-bit) or **Linux** (64-bit). macOS is not a first-class target for this codebase but may work with enough manual dependency work. |
| **Python** | **3.10 or newer** (3.12 is used in development). **64-bit** interpreter required. |
| **RAM** | **4 GB** minimum for small WAVs and manual trim only; **8 GB+** recommended. **AI trim** benefits from **16 GB+** when using larger Whisper models. |
| **Disk** | **Core app** (no PyTorch): typically **hundreds of MB** for the venv after `requirements.txt`. **AI trim** adds **PyTorch + WhisperX** — often **~1–4 GB** with CPU PyTorch, **several GB** with CUDA builds. |
| **GPU** | **Optional**. AI trim is faster with a **CUDA-capable NVIDIA GPU** and matching PyTorch CUDA wheels; CPU-only installs work but are slower. |
| **Display** | Any reasonable resolution; layout adapts for narrow or remote-desktop style windows. |
| **Tkinter** | Usually bundled with the official **python.org** Windows installer (choose *tcl/tk*). On some Linux distros, install the distro **`python3-tk`** (or equivalent) package. |
| **ffmpeg** | **Not required for pure WAV** load/export in typical use. **pydub** uses **ffmpeg** for many non-WAV formats; install ffmpeg and ensure it is on your **PATH** if you rely on those formats. |
| **Network** | Required for **first-time model downloads** (WhisperX, pyannote, etc.) and Hugging Face access when using diarization. |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gmater/voice-engine.git
cd voice-engine
```

(If your checkout folder is named differently, use that path below instead of `voice-engine`.)

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip wheel
```

**Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip wheel
```

You can also use the helper script **`scripts/recreate_core_venv.ps1`** on Windows to create a venv and install only the core stack.

### 3. Install dependencies

**Core Voice Engine (manual trim + waveform + export)** — recommended baseline:

```bash
python -m pip install --no-cache-dir -r requirements.txt
```

**Optional: AI trim (WhisperX / pslicer)** — large install; pick **one** strategy.

- **CPU PyTorch (smaller disk, slower inference):**

  ```bash
  python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  python -m pip install --no-cache-dir -r requirements-pslicer.txt
  ```

- **CUDA PyTorch (GPU):** install the matching **`torch` / `torchvision` / `torchaudio`** wheels for your driver from [PyTorch — Get Started](https://pytorch.org/get-started/locally/), then:

  ```bash
  python -m pip install --no-cache-dir -r requirements-pslicer.txt
  ```

See the comment block at the top of **`requirements-pslicer.txt`** for disk-saving tips (single CUDA build, `pip cache purge`, optional second venv for AI only).

### 4. Run the application

```bash
python slicer.py
```

Optional: open a file on startup:

```bash
python slicer.py "C:\path\to\file.wav"
python slicer.py /home/you/audio/file.wav
```

---

## How to use

### Voice Engine (GUI)

1. **Start** the app with `python slicer.py`.
2. **Open** a WAV via the UI (**Open…** or drag/drop if your build supports it).
3. **Select** the region to keep using the **highlighted span** on the waveform (drag handles or use the time fields, depending on build).
4. **Preview** playback if needed.
5. **Export** — choose output folder and filename, then export; overwritten files may get a `.bak` sibling depending on behavior in your version.

**AI trim…** (optional): requires a Python environment where **`import torch`** and **`import pslicer`** succeed. Configure a **Hugging Face** token under **Settings…** (or set **`HF_TOKEN`** / **`HUGGING_FACE_HUB_TOKEN`**) and accept model terms on the Hub if you want **speaker diarization**; you can still continue **without** diarization when prompted, for sentence-style cuts without multi-speaker labels. See [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) for terms links used in typical setups.

**Stress / automated UI test** (developers):

```bash
python slicer.py --stress
```

### pslicer (CLI)

From the repo root with the same venv (including AI dependencies if you use diarization features):

```bash
python pslicer.py recording.wav --out ./exports --model medium
python pslicer.py --help
```

---

## Basic troubleshooting

| Problem | Things to try |
|---------|----------------|
| **`No module named 'tkinter'`** (Linux) | Install your distro’s Tk package, e.g. `python3-tk` (Debian/Ubuntu) or `tk` on Fedora. |
| **`No module named 'torch'`** when using **AI trim** | Install PyTorch (CPU or CUDA) **before** `requirements-pslicer.txt`, then install WhisperX stack; restart the app using **that** venv’s `python`. |
| **`import pslicer` / WhisperX errors** | Ensure you used **`pip install -r requirements-pslicer.txt`** in the **same** environment; on Windows, **`whisperx_windows_entry.py`** is part of this repo for DLL/CUDA helpers—run from the checkout, not a partial copy. |
| **ffmpeg errors** when loading non-WAV | Install [ffmpeg](https://ffmpeg.org/) and add it to **PATH**, or convert inputs to WAV first. |
| **Hugging Face / pyannote access** | Set token in **Settings…** or env vars; accept Hub model conditions; without a token you may still run **without diarization** when the app offers it. |
| **CUDA out of memory** | Use a **smaller** Whisper model, close other GPU apps, or switch to **CPU** PyTorch for AI trim (slower, less VRAM). |
| **Huge venv size** | You likely installed **CUDA PyTorch** or multiple stacks. Use **only** `requirements.txt` for manual-only work, or **CPU** torch + one CUDA build in a **dedicated** venv; run **`python -m pip cache purge`** after correcting installs. |
| **Windows path / spaces** | Prefer quoting paths: `python slicer.py "D:\My Audio\file.wav"`. |

---

## Legal notice

**Copyright and restricted content.** Voice Engine and pslicer are **tools**. What you record, load, transcribe, trim, or export—especially **copyrighted** or otherwise **restricted** material—is **your responsibility**. You must comply with applicable **copyright**, **contract**, **privacy**, and **platform terms** (e.g. streaming services). **The authors do not provide legal advice**; when in doubt, obtain permission or consult a qualified professional.

**Third-party models.** AI features rely on third-party weights and libraries (e.g. **WhisperX**, **pyannote**, **faster-whisper**). Their **licenses and acceptable-use rules** apply in addition to this notice.

---

## Repository layout

| Path | Role |
|------|------|
| `slicer.py` | Main **Voice Engine** GUI. |
| `pslicer.py` | **Auto-trim** library + CLI. |
| `whisperx_windows_entry.py` | Windows helpers for WhisperX / pyannote integration. |
| `requirements.txt` | Core dependencies (no PyTorch). |
| `requirements-pslicer.txt` | Optional **AI trim** stack (`whisperx`). |
| `scripts/` | Helper scripts (e.g. small venv setup on Windows). |
| `test_*.py` | Tests. |
| `extras/` | Optional scripts and demos; see `extras/README.md`. |

---

## Tests

With the venv activated:

```bash
python test_pslicer_separation.py
python test_pslicer_preview.py
python test_pslicer_diagnostics.py
python test_pslicer_swallow_neighbors.py
python test_slicer_pslicer_stress.py
python test_voice_engine_settings.py
```

---

## Publishing / remotes

This tree is intended as a **standalone git repository**. Example remote:

```bash
git remote add origin https://github.com/gmater/voice-engine.git
git push -u origin main
```

With [GitHub CLI](https://cli.github.com/): `gh repo create voice-engine --public --source=. --remote=origin --push` (after `gh auth login`). Do **not** commit secrets (tokens, API keys).

---

## License / attribution

Add a **`LICENSE`** file suitable for your distribution. Respect upstream licenses for **WhisperX**, **pyannote**, **faster-whisper**, **PyTorch**, and other dependencies you ship or install.
