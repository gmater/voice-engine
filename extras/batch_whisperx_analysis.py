import os
import subprocess
import sys

import torch

# WhisperX calls the `ffmpeg` binary; running via `venv\Scripts\python.exe` without
# activating the venv leaves Scripts off PATH, which causes WinError 2.
_bindir = os.path.dirname(os.path.abspath(sys.executable))
os.environ["PATH"] = _bindir + os.pathsep + os.environ.get("PATH", "")

clean_dir = r"C:\AI\SanctumCore\voice_assets\raw_source\clean"
analysis_dir = r"C:\AI\SanctumCore\voice_assets\analysis"

os.makedirs(analysis_dir, exist_ok=True)


def _resolve_hf_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t, "HF_TOKEN / HUGGING_FACE_HUB_TOKEN"
    try:
        from huggingface_hub import get_token

        t = get_token()
        if t:
            return t, "huggingface-cli login cache"
    except Exception:
        pass
    return None, None


hf_token, hf_token_source = _resolve_hf_token()
if not hf_token:
    print(
        "Diarization needs a Hugging Face token (gated pyannote model).\n"
        "  1) Accept terms (while logged in): "
        "https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        "  2) Create a read token: https://huggingface.co/settings/tokens\n"
        "  3) cmd.exe:  set HF_TOKEN=hf_...\n"
        "     Then:     venv\\Scripts\\python.exe batch_whisperx_analysis.py\n"
        "     Or run:   venv\\Scripts\\hf.exe auth login\n"
        "\n"
        "If you already set a token but see 401 on diarization: use the same HF account "
        "that accepted the model terms, and a token with Read access (not expired)."
    )
    sys.exit(1)

hf_token = hf_token.strip()
if not hf_token:
    print("HF_TOKEN is set but empty after trimming whitespace.")
    sys.exit(1)

if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    print(
        "Note: HF_TOKEN / HUGGING_FACE_HUB_TOKEN overrides `hf auth login`. "
        "A bad value there causes 401 even after a successful login."
    )

print(f"Hugging Face: using token from {hf_token_source}.")

try:
    from huggingface_hub import HfApi

    who = HfApi(token=hf_token).whoami()
    uname = who.get("name") or who.get("fullname") or "unknown"
    print(f"Hugging Face API: token OK (identity: {uname}).")
except Exception as e:
    print(
        "Hugging Face rejected this token — same as `hf auth login` → Invalid user token.\n"
        f"API error: {e}\n\n"
        "Fix:\n"
        "  • Create a new token: https://huggingface.co/settings/tokens "
        "(classic token, Read is enough).\n"
        "  • cmd: no spaces around `=` — use  set HF_TOKEN=hf_...\n"
        "  • Clear a bad env token:  set HF_TOKEN=\n"
        "    then  venv\\Scripts\\hf.exe auth login\n"
        "  • When the token validates, accept pyannote terms: "
        "https://huggingface.co/pyannote/speaker-diarization-community-1"
    )
    sys.exit(1)

compute_type = "float16" if torch.cuda.is_available() else "float32"
device_note = "CUDA" if torch.cuda.is_available() else "CPU (no CUDA — using float32)"
audio_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
print(f"Found {len(audio_files)} files. Device: {device_note}. WhisperX diarization...")

_whisperx_entry = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisperx_windows_entry.py")

for file in audio_files:
    input_path = os.path.join(clean_dir, file)
    print(f"\n--- Analyzing Isotope: {file} ---")
    subprocess.run(
        [
            sys.executable,
            _whisperx_entry,
            input_path,
            "--model",
            "large-v3-turbo",
            "--diarize",
            "--min_speakers",
            "1",
            "--max_speakers",
            "3",
            "--compute_type",
            compute_type,
            "--hf_token",
            hf_token,
            "--output_dir",
            analysis_dir,
            "--output_format",
            "json",
        ]
    )

print("\nBatch analysis complete. JSON blueprints generated.")
