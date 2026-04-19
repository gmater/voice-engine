"""
Voice Engine — Streamlit web trimmer (local PC; optional companion to ``slicer.py``).

Run from ``voice_engine`` repo root::

  streamlit run extras/streamlit_slicer.py

Requires: streamlit, librosa, matplotlib, pydub, numpy
(slicer.py is unchanged; this is a separate UI.)
"""

from __future__ import annotations

import io
import os
import re
from glob import glob

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pydub import AudioSegment

DEFAULT_IN = r"C:\AI\SanctumCore\voice_assets\raw_source\clean"
DEFAULT_OUT = r"C:\AI\SanctumCore\voice_assets\Pure_Jarvis_Audio"

os.makedirs(DEFAULT_OUT, exist_ok=True)

BG = "#000000"
ACCENT = "#00FBFF"
WAVEFORM = "#00FF88"
MUTED = "#7DD3CE"
TEXT = "#E8FFF8"


def list_wav_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    paths = sorted(glob(os.path.join(folder, "*.wav")))
    return [os.path.basename(p) for p in paths]


def full_path_in(folder: str, basename: str) -> str:
    return os.path.normpath(os.path.join(folder, basename))


def increment_output_name(name: str) -> str:
    """Match slicer.py _increment_name behavior."""
    base = name[:-4] if name.lower().endswith(".wav") else name
    m = re.search(r"(\d+)$", base)
    if m:
        n = int(m.group(1)) + 1
        nb = base[: m.start()] + str(n).zfill(len(m.group(1)))
        return nb + (".wav" if name.lower().endswith(".wav") else "")
    return name


def fig_waveform(y: np.ndarray, sr: int, t0: float, t1: float, duration: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100, facecolor=BG)
    ax.set_facecolor(BG)
    librosa.display.waveshow(y, sr=sr, ax=ax, color=WAVEFORM, alpha=0.88, lw=0.45)
    ax.axvspan(t0, t1, color=ACCENT, alpha=0.35, linewidth=0)
    ax.set_xlim(0.0, max(duration, 0.01))
    ax.set_xlabel("Time (s)", color=MUTED, fontsize=13)
    ax.set_ylabel("Amplitude", color=MUTED, fontsize=13)
    ax.tick_params(colors=MUTED, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.set_title("Trim range (highlighted)", color=TEXT, fontsize=14)
    plt.tight_layout()
    return fig


def _clear_manual_time_keys() -> None:
    """Reset manual time fields when the range slider moves."""
    st.session_state.pop("manual_s", None)
    st.session_state.pop("manual_e", None)


def main():
    st.set_page_config(
        page_title="Voice Engine (Web)",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    if st.session_state.pop("_save_ok", False):
        st.success(st.session_state.pop("_save_ok_msg", "Saved."))

    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: {BG}; }}
            h1, h2, h3, label, .stMarkdown p, span {{ color: {TEXT} !important; }}
            div[data-testid="stVerticalBlock"] > div {{ padding-top: 0.25rem; }}
            .stSlider label {{ font-size: 1.05rem !important; }}
            .stButton > button {{
                min-height: 3.25rem;
                font-size: 1.15rem !important;
                font-weight: 600;
                padding: 0.75rem 1.25rem;
                border-radius: 12px;
            }}
            .stSelectbox label {{ font-size: 1.05rem !important; }}
            .stTextInput label {{ font-size: 1.05rem !important; }}
            .stTextInput input {{ font-size: 1.1rem !important; min-height: 2.75rem; }}
            .stNumberInput input {{ font-size: 1.1rem !important; min-height: 2.5rem; }}
            audio {{ width: 100% !important; min-height: 48px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Voice Engine — Web trimmer")
    st.caption("Chunky controls for desktop & mobile browsers · audio stays on this PC")

    if "out_name" not in st.session_state:
        st.session_state.out_name = "jarvis_001.wav"

    wav_names = list_wav_files(DEFAULT_IN)
    if not wav_names:
        st.error(f"No `.wav` files in:\n`{DEFAULT_IN}`")
        st.stop()

    choice = st.selectbox(
        "Source file (clean)",
        options=wav_names,
        index=0,
        help="Files from raw_source/clean",
    )
    src_path = full_path_in(DEFAULT_IN, choice)

    if (
        "loaded_src" not in st.session_state
        or st.session_state.loaded_src != src_path
        or st.session_state.get("reload") is True
    ):
        with st.spinner("Loading audio…"):
            y, sr = librosa.load(src_path, sr=22050, mono=True)
        st.session_state.loaded_src = src_path
        st.session_state.y = y
        st.session_state.sr = sr
        st.session_state.duration = float(len(y)) / float(sr)
        st.session_state.reload = False
        for _k in ("trim_slider", "manual_s", "manual_e"):
            st.session_state.pop(_k, None)

    y = st.session_state.y
    sr = st.session_state.sr
    duration = st.session_state.duration

    st.markdown("### Trim range (seconds)")
    _default_hi = min(duration, max(duration * 0.25, 0.1))
    t0, t1 = st.slider(
        "Drag both handles (touch-friendly)",
        min_value=0.0,
        max_value=max(duration, 0.01),
        value=(0.0, _default_hi),
        step=0.01,
        format="%.2f",
        key="trim_slider",
        on_change=_clear_manual_time_keys,
    )

    with st.expander("Type exact start / end (seconds)", expanded=False):
        st.caption("Enter times in seconds, then tap **Apply typed times** (end may be before start — we swap).")
        mc1, mc2 = st.columns(2)
        with mc1:
            mt0 = st.number_input(
                "Start (s)",
                min_value=0.0,
                max_value=float(max(duration, 0.01)),
                value=float(t0),
                step=0.001,
                format="%.4f",
                key="manual_s",
            )
        with mc2:
            mt1 = st.number_input(
                "End (s)",
                min_value=0.0,
                max_value=float(max(duration, 0.01)),
                value=float(t1),
                step=0.001,
                format="%.4f",
                key="manual_e",
            )
        if st.button("Apply typed times", use_container_width=True, key="apply_manual_times"):
            lo, hi = sorted((float(mt0), float(mt1)))
            lo = max(0.0, lo)
            hi = min(float(duration), hi)
            if hi - lo < 0.001:
                st.error("End must be after start by at least 0.001 s.")
            else:
                st.session_state.trim_slider = (lo, hi)
                st.session_state.pop("manual_s", None)
                st.session_state.pop("manual_e", None)
                st.rerun()

    if t1 - t0 < 0.02:
        st.warning("Selection is very short; increase the range for a usable trim.")

    fig = fig_waveform(y, sr, t0, t1, duration)
    st.pyplot(fig)
    plt.close(fig)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        play = st.button("Play selection", type="primary", use_container_width=True)
    with col_b:
        save = st.button("Save trim", type="secondary", use_container_width=True)

    start_ms = int(round(t0 * 1000))
    end_ms = int(round(t1 * 1000))

    if play:
        try:
            seg = AudioSegment.from_wav(src_path)
            dur_ms = len(seg)
            a = max(0, min(start_ms, dur_ms))
            b = max(a + 1, min(end_ms, dur_ms))
            clip = seg[a:b]
            buf = io.BytesIO()
            clip.export(buf, format="wav")
            buf.seek(0)
            st.audio(buf.read(), format="audio/wav", start_time=0)
        except Exception as e:
            st.error(f"Could not build preview: {e}")

    st.text_input(
        "Output filename",
        key="out_name",
        help="Saved under Pure_Jarvis_Audio · auto-increments after Save",
    )

    if save:
        name = (st.session_state.out_name or "").strip()
        if not name:
            st.warning("Enter a filename.")
        elif not name.lower().endswith(".wav"):
            name += ".wav"
        try:
            seg = AudioSegment.from_wav(src_path)
            dur_ms = len(seg)
            a = int(round(t0 * 1000))
            b = int(round(t1 * 1000))
            a = max(0, min(a, dur_ms))
            b = max(a + 1, min(b, dur_ms))
            if b - a < 1:
                st.warning("Selection too small to export.")
            else:
                out_path = os.path.join(DEFAULT_OUT, name)
                seg[a:b].export(out_path, format="wav")
                st.session_state.out_name = increment_output_name(name)
                st.session_state._save_ok = True
                st.session_state._save_ok_msg = f"Saved:\n`{out_path}`"
                st.rerun()
        except Exception as e:
            st.error(f"Export failed: {e}")

    st.divider()
    st.markdown(
        f"**Input:** `{DEFAULT_IN}`  \n**Output:** `{DEFAULT_OUT}`  \n"
        f"**Duration:** {duration:.2f}s"
    )


if __name__ == "__main__":
    main()
