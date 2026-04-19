# Extras — optional scripts

These files are **not required** for the core **Voice Engine** (`slicer.py`) + **pslicer** (`pslicer.py`) workflow. They are kept for demos, alternate UIs, benchmarks, and one-off workspace automation.

Run everything **from the `voice_engine` repository root** so imports resolve (`pslicer`, etc.).

## Contents

| Script | Purpose |
|--------|---------|
| `02_pro_splicer.py` | Thin launcher for `pro_slicer.main()`. |
| `pro_slicer.py` | Alternate “pro” Tk splicer UI. |
| `manual_splicer.py` | Manual splicing helper. |
| `drag_slicer.py` | Audacity-style region slicer (Tk). |
| `streamlit_slicer.py` | Streamlit web trimmer (separate from `slicer.py`). |
| `pslicer_demo_prepare.py` | Edge TTS → short two-voice demo WAV (for demos). |
| `pslicer_ultimate_demo.py` | End-to-end pslicer demo + optional plot/play. |
| `pslicer_benchmark_suite.py` | Synthetic multi-speaker benchmark vs pslicer. |
| `batch_whisperx_analysis.py` | Batch WhisperX-style analysis (paths may need editing for your tree). |
| `harvest_speakers_from_whisperx.py` | Harvest sentence WAVs from analysis JSON + clean sources. |
| `jarvis_audition.py` | Tk GUI to audition/move WAVs (paths configurable in file). |
| `kokoro_voice_pack.py` | Kokoro / voice-pack utilities. |

## Examples

```text
venv\Scripts\python.exe extras\pro_slicer.py
venv\Scripts\python.exe extras\pslicer_ultimate_demo.py --model base --skip-play
streamlit run extras/streamlit_slicer.py
```

Benchmark:

```text
venv\Scripts\python.exe extras\pslicer_benchmark_suite.py --synth-only
```

## Note on paths

Several utilities still contain **example absolute paths** (e.g. under `SanctumCore/voice_assets/…`). Before publishing a clean repo, search `extras/*.py` for `C:\\` or `voice_assets` and replace with environment variables or CLI arguments.
