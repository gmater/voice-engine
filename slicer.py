"""
Voice Engine — production waveform trimmer (Tk + Matplotlib).

Stack: Tkinter, Matplotlib (SpanSelector), librosa waveshow, pygame, pydub.
Optional WhisperX auto-trim (pslicer): toolbar “AI trim…” opens a preview window (waveform, timestamps, include/exclude) then exports to the chosen output folder. Hugging Face token: **Settings…** (persisted) or ``HF_TOKEN`` / CLI login.
Adapts layout for desktop vs touch/remote. Export filename follows the newest .wav in the output
folder; Output folder picks the directory (path not shown on chrome). Each save backs up any overwritten
file as .bak and refreshes Sanctum_moving_backup.wav in that folder.
Run: python slicer.py  (see requirements.txt for a small venv; AI trim is optional.)
Optional: python slicer.py "C:\\path\\to\\file.wav"  (load that WAV on startup)
Stress: python slicer.py --stress  (runs desktop + touch sessions)
"""

from __future__ import annotations

import os
import sys

# Per-monitor DPI — before Tk creates windows (sharp rendering on phone RDP / high-DPI).
try:
    import ctypes

    # 2 = PROCESS_PER_MONITOR_DPI_AWARE (Win 8.1+); falls back to system-aware.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
except Exception:
    pass

_SM_REMOTESESSION = 0x1000


def _is_remote_desktop_session() -> bool:
    """True when running inside RDP (e.g. phone client → Windows)."""
    if os.environ.get("SANCTUM_REMOTE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if sys.platform != "win32":
        return False
    sn = os.environ.get("SESSIONNAME", "").strip().upper()
    if sn.startswith("RDP") or "RDP-TCP" in sn or "ICA-TCP" in sn:
        return True
    try:
        import ctypes as _ctypes

        return int(_ctypes.windll.user32.GetSystemMetrics(_SM_REMOTESESSION)) != 0
    except Exception:
        return False


def _parse_work_area(
    profile_geometry: str, screen_w: int, screen_h: int, remote: bool
) -> tuple[int, int, int, int]:
    """
    Returns (avail_w, avail_h, gw0, gh0) using one margin for width/height so minsize and
    initial geometry stay consistent (avoids minwidth > usable width on narrow displays).
    """
    marg_w = 28
    # Extra bottom margin on RDP so taskbar / phone nav does not cover export controls.
    marg_h = 100 if remote else 44
    avail_w = max(200, screen_w - marg_w)
    avail_h = max(200, screen_h - marg_h)
    try:
        part = profile_geometry.lower().replace(" ", "").split("x", 1)
        gw0, gh0 = int(part[0]), int(part[1])
    except (ValueError, IndexError):
        gw0, gh0 = 1200, 800
    return avail_w, avail_h, gw0, gh0

import gc
import importlib.util
import json
import queue
import re
import shutil
import tempfile
import time
import threading
import webbrowser
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from pydub import AudioSegment

_REPO_ROOT = Path(__file__).resolve().parent
_LEGACY_IN = Path(r"C:\AI\SanctumCore\voice_assets\raw_source\clean")
_LEGACY_OUT = Path(r"C:\AI\SanctumCore\voice_assets\Pure_Jarvis_Audio")


def _default_open_dir() -> str:
    try:
        if _LEGACY_IN.is_dir():
            return str(_LEGACY_IN)
    except OSError:
        pass
    return str(_REPO_ROOT)


def _default_export_dir() -> str:
    """Prefer legacy export folder if that tree exists; else ``<repo>/exports``. Never raises."""
    candidates: list[Path] = []
    try:
        if _LEGACY_OUT.parent.is_dir():
            candidates.append(_LEGACY_OUT)
    except OSError:
        pass
    candidates.append(_REPO_ROOT / "exports")
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            return str(d.resolve())
        except OSError:
            continue
    return str(_REPO_ROOT)


DEFAULT_IN = _default_open_dir()
DEFAULT_OUT = _default_export_dir()


def _pslicer_requirements_file() -> Path:
    return _REPO_ROOT / "requirements-pslicer.txt"


def _ai_trim_missing_stack_message(*, detail: str) -> str:
    """User-facing text when pslicer / torch / WhisperX is not installed."""
    exe = sys.executable
    req = _pslicer_requirements_file()
    return (
        "AI trim needs PyTorch, WhisperX, and related packages in the same Python environment "
        "that runs Voice Engine.\n\n"
        f"{detail}\n\n"
        "Install (use the venv that launches this app). For less disk use, install **CPU** "
        "PyTorch first (unless you need CUDA), then WhisperX:\n"
        "  https://pytorch.org/get-started/locally/  (choose CPU)\n"
        "  or: pip install torch torchvision torchaudio --index-url "
        "https://download.pytorch.org/whl/cpu\n\n"
        f'  "{exe}" -m pip install -r "{req}"\n\n'
        "Day-to-day trimming only needs requirements.txt (no torch). "
        "You can use a second venv for AI trim if you prefer.\n\n"
        "Then start Voice Engine with that interpreter, for example:\n"
        f'  "{exe}" "{_REPO_ROOT / "slicer.py"}"'
    )


def _ai_trim_import_error_message(exc: ImportError) -> str:
    return _ai_trim_missing_stack_message(detail=str(exc))


def _resolve_hf_token_probe() -> str | None:
    """Lightweight HF token check (same rules as ``pslicer.resolve_hf_token``) without importing pslicer."""
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t and str(t).strip():
        return str(t).strip()
    try:
        from huggingface_hub import get_token

        t2 = get_token()
        if t2 and str(t2).strip():
            return str(t2).strip()
    except Exception:
        pass
    return None


VOICE_ENGINE_SETTINGS_PATH_ENV = "VOICE_ENGINE_SETTINGS_PATH"


def voice_engine_settings_path() -> Path:
    """User-local JSON for Voice Engine settings (tests may set VOICE_ENGINE_SETTINGS_PATH)."""
    override = (os.environ.get(VOICE_ENGINE_SETTINGS_PATH_ENV) or "").strip()
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override))).resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return (base / "VoiceEngine" / "settings.json").resolve()
    return (Path.home() / ".config" / "voice_engine" / "settings.json").resolve()


def load_voice_engine_settings_into_environ() -> None:
    """If no HF token is already in the environment, load ``hf_token`` from disk into ``HF_TOKEN``."""
    if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
        return
    p = voice_engine_settings_path()
    if not p.is_file():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return
    if not isinstance(data, dict):
        return
    tok = (data.get("hf_token") or "").strip()
    if tok:
        os.environ["HF_TOKEN"] = tok


def save_voice_engine_hf_token(hf_token: str | None) -> None:
    """Persist HF token to disk and apply to this process (``HF_TOKEN`` / clear)."""
    p = voice_engine_settings_path()
    tok = (hf_token or "").strip()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        if tok:
            p.write_text(json.dumps({"hf_token": tok}, indent=2), encoding="utf-8")
            os.environ["HF_TOKEN"] = tok
        else:
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
            os.environ.pop("HF_TOKEN", None)
    except OSError:
        raise


# Modern dark UI (slate base + sky accent; avoids pure black / neon halation on OLED & RDP)
BG = "#0f172a"
PANEL = "#1e293b"
ACCENT = "#38bdf8"
ACCENT_HOVER = "#7dd3fc"
ACCENT_ON_PRIMARY = "#0f172a"
WAVEFORM_COLOR = "#34d399"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
BTN_BG = "#334155"
BTN_ACTIVE = "#475569"
BTN_TRIM_START_BG = "#14532d"
BTN_TRIM_START_ACTIVE = "#166534"
BTN_TRIM_END_BG = "#7f1d1d"
BTN_TRIM_END_ACTIVE = "#991b1b"
BTN_PLAY_BG = "#1e3a5f"
BTN_PLAY_ACTIVE = "#2e5078"
SPAN_FACE = "#38bdf8"
SPAN_EDGE = "#7dd3fc"

SEEK_STEP_MS = 5000

# Overwritten on every successful export (rolling safety copy in the output folder).
ROLLING_BACKUP_BASENAME = "Sanctum_moving_backup.wav"


def _is_reserved_export_wav(filename: str) -> bool:
    n = filename.lower()
    if n == ROLLING_BACKUP_BASENAME.lower():
        return True
    if n.endswith(".bak"):
        return True
    return False


def suggest_next_export_filename(export_dir: str, default: str = "jarvis_001.wav") -> str:
    """Next filename after the most recently modified non-reserved .wav in export_dir."""
    if not os.path.isdir(export_dir):
        return default
    newest_name: str | None = None
    newest_t = -1.0
    try:
        for fn in os.listdir(export_dir):
            if not fn.lower().endswith(".wav"):
                continue
            if _is_reserved_export_wav(fn):
                continue
            fp = os.path.join(export_dir, fn)
            try:
                t = os.path.getmtime(fp)
            except OSError:
                continue
            if t > newest_t:
                newest_t = t
                newest_name = fn
    except OSError:
        return default
    if not newest_name:
        return default
    m = re.match(r"^(.+?)(\d+)\.wav$", newest_name, re.I)
    if m:
        pfx, ds = m.group(1), m.group(2)
        n = int(ds) + 1
        return f"{pfx}{str(n).zfill(len(ds))}.wav"
    stem, _ = os.path.splitext(newest_name)
    return bump_export_filename_if_exists(export_dir, f"{stem}_001.wav")


def _sanitize_export_filename_text(s: str, max_len: int) -> str:
    """Strip control chars and Windows-forbidden filename symbols; collapse whitespace."""
    s = (s or "").strip()
    s = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len].rstrip()


def build_export_filename_from_hints(
    speaker: str,
    transcript: str,
    t0_sec: float,
    *,
    max_stem_chars: int = 180,
) -> str:
    """
    ``Speaker - Transcribed text at {offset_ms}ms.wav`` using start time in the original file.

    ``t0_sec`` is the left edge of the current trim (seconds). Stem is capped for Windows paths.
    """
    sp = _sanitize_export_filename_text(speaker, 48) or "Unknown"
    tx0 = _sanitize_export_filename_text(transcript, 120) or "clip"
    ms = int(max(0, round(float(t0_sec) * 1000)))
    suffix = f" at {ms}ms.wav"
    sep = " - "
    budget = max_stem_chars - len(sp) - len(sep) - len(suffix)
    if budget < 1:
        tx = tx0[:1]
    else:
        tx = tx0[:budget].rstrip() or "clip"
    return f"{sp}{sep}{tx}{suffix}"


def bump_export_filename_if_exists(export_dir: str, name: str) -> str:
    """If name already exists in export_dir, increment trailing digits (preserve width) until free."""
    name = name.strip()
    if not name.lower().endswith(".wav"):
        name += ".wav"
    path = os.path.join(export_dir, name)
    if not os.path.exists(path):
        return name
    base = name[:-4]
    m = re.search(r"(\d+)$", base)
    if m:
        pfx, ds = base[: m.start()], m.group(1)
        w = len(ds)
        n = int(ds)
        for _ in range(10000):
            n += 1
            cand = f"{pfx}{str(n).zfill(w)}.wav"
            if not os.path.exists(os.path.join(export_dir, cand)):
                return cand
    for i in range(1, 100000):
        cand = f"{base}_{i:04d}.wav"
        if not os.path.exists(os.path.join(export_dir, cand)):
            return cand
    return name


@dataclass(frozen=True)
class UiProfile:
    name: str
    font_label: tuple
    font_btn: tuple
    font_btn_bold: tuple
    font_toggle: tuple
    font_mono: tuple
    font_header: tuple
    pad_x: int
    pad_y: int
    pad_x_tight: int
    pad_y_tight: int
    pad_toggle_x: int
    pad_toggle_y: int
    header_h: int
    minsize_w: int
    minsize_h: int
    geometry: str
    force_zoomed: bool
    tk_scale_min: float
    hint: str
    hint_wrap: int
    plot_fig_w: float
    plot_fig_h: float
    mpl_tick: int
    mpl_axis: int
    mpl_title: int
    btn_ipadx: int
    btn_ipady: int
    trim_ipadx: int
    trim_ipady: int
    entry_w: int
    entry_ipady: int
    min_trim_sec: float
    span_min_sec: float
    span_grab: int
    span_lw: float
    span_handle_lw: float
    waveform_lw: float
    browse_ipadx: int
    browse_ipady: int
    save_ipadx: int
    save_ipady: int


def resolve_ui_profile(root: tk.Tk) -> str:
    """desktop = mouse-first; touch = fat-finger / phone RDP. Override with SANCTUM_UI."""
    raw = os.environ.get("SANCTUM_UI", "auto").strip().lower()
    if raw in ("touch", "phone", "mobile", "remote", "rdp"):
        return "touch"
    if raw in ("desktop", "pc", "mouse"):
        return "desktop"
    if raw == "auto" and _is_remote_desktop_session():
        return "touch"
    try:
        sw = int(root.winfo_screenwidth())
        sh = int(root.winfo_screenheight())
    except tk.TclError:
        return "desktop"
    short, long_ = min(sw, sh), max(sw, sh)
    try:
        scale = float(root.tk.call("tk", "scaling"))
    except tk.TclError:
        scale = 1.0
    if short < 1000 or long_ < 1280:
        return "touch"
    if scale >= 1.45:
        return "touch"
    return "desktop"


def ui_profile_for(name: str) -> UiProfile:
    if name == "touch":
        return UiProfile(
            name="touch",
            font_label=("Segoe UI", 14),
            font_btn=("Segoe UI", 12),
            font_btn_bold=("Segoe UI", 12, "bold"),
            font_toggle=("Segoe UI", 16, "bold"),
            font_mono=("Consolas", 14),
            font_header=("Segoe UI", 16, "bold"),
            pad_x=14,
            pad_y=12,
            pad_x_tight=10,
            pad_y_tight=8,
            pad_toggle_x=18,
            pad_toggle_y=16,
            header_h=56,
            minsize_w=1024,
            minsize_h=720,
            geometry="1200x820",
            force_zoomed=True,
            tk_scale_min=1.35,
            hint="Center: large toggle play · Space = play/pause · Esc stop · ←/→ seek · tap waveform · typed start/end (s) below",
            hint_wrap=1100,
            plot_fig_w=11.0,
            plot_fig_h=5.5,
            mpl_tick=12,
            mpl_axis=14,
            mpl_title=14,
            btn_ipadx=10,
            btn_ipady=12,
            trim_ipadx=10,
            trim_ipady=12,
            entry_w=36,
            entry_ipady=10,
            min_trim_sec=0.03,
            span_min_sec=0.04,
            span_grab=32,
            span_lw=4.0,
            span_handle_lw=5.0,
            waveform_lw=0.45,
            browse_ipadx=12,
            browse_ipady=10,
            save_ipadx=14,
            save_ipady=12,
        )
    return UiProfile(
        name="desktop",
        font_label=("Segoe UI", 11),
        font_btn=("Segoe UI", 10),
        font_btn_bold=("Segoe UI", 10, "bold"),
        font_toggle=("Segoe UI", 13, "bold"),
        font_mono=("Consolas", 11),
        font_header=("Segoe UI", 13, "bold"),
        pad_x=8,
        pad_y=8,
        pad_x_tight=6,
        pad_y_tight=6,
        pad_toggle_x=10,
        pad_toggle_y=10,
        header_h=44,
        minsize_w=960,
        minsize_h=640,
        geometry="1200x780",
        force_zoomed=False,
        tk_scale_min=1.0,
        hint="Center: play/pause · Space · Esc · ←/→ seek · click waveform · typed start/end (s) below",
        hint_wrap=920,
        plot_fig_w=10.0,
        plot_fig_h=4.0,
        mpl_tick=10,
        mpl_axis=11,
        mpl_title=11,
        btn_ipadx=6,
        btn_ipady=8,
        trim_ipadx=6,
        trim_ipady=8,
        entry_w=42,
        entry_ipady=6,
        min_trim_sec=0.02,
        span_min_sec=0.012,
        span_grab=20,
        span_lw=2.5,
        span_handle_lw=3.5,
        waveform_lw=0.32,
        browse_ipadx=8,
        browse_ipady=8,
        save_ipadx=10,
        save_ipady=8,
    )


def _attach_tk_hover_tooltip(widget: tk.Widget, text: str, *, wraplength: int = 360) -> None:
    """Small hover tooltip (destroyed on leave). Safe if ``widget`` is destroyed while hidden."""
    tip_ref: list[tk.Toplevel | None] = [None]

    def _destroy_tip() -> None:
        tw = tip_ref[0]
        if tw is not None:
            try:
                tw.destroy()
            except tk.TclError:
                pass
            tip_ref[0] = None

    def _show_tip(_e: object | None = None) -> None:
        if tip_ref[0] is not None:
            return
        s = (text or "").strip()
        if not s:
            return
        try:
            x = int(widget.winfo_rootx()) + 10
            y = int(widget.winfo_rooty()) + int(widget.winfo_height()) + 4
        except tk.TclError:
            return
        tw = tk.Toplevel(widget)
        tip_ref[0] = tw
        tw.wm_overrideredirect(True)
        try:
            tw.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=s,
            justify=tk.LEFT,
            bg="#1e293b",
            fg="#e2e8f0",
            relief=tk.SOLID,
            bd=1,
            padx=10,
            pady=8,
            wraplength=wraplength,
            font=("Segoe UI", 9),
        ).pack()
        try:
            tw.update_idletasks()
            sw = int(tw.winfo_screenwidth())
            sh = int(tw.winfo_screenheight())
            tw_w = int(tw.winfo_width())
            tw_h = int(tw.winfo_height())
            if x + tw_w > sw - 8:
                x = max(8, sw - tw_w - 8)
            if y + tw_h > sh - 8:
                y = max(8, int(widget.winfo_rooty()) - tw_h - 6)
            tw.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            pass

    def _hide_tip(_e: object | None = None) -> None:
        _destroy_tip()

    widget.bind("<Enter>", _show_tip)
    widget.bind("<Leave>", _hide_tip)
    widget.bind("<Destroy>", lambda _e: _destroy_tip())


def _pslicer_clip_interval_subset(
    inner: tuple[float, float, object, object],
    t_outer0: float,
    t_outer1: float,
    *,
    eps: float = 1e-4,
) -> bool:
    """True if ``[inner[0], inner[1]]`` lies fully inside ``[t_outer0, t_outer1]`` (inclusive, eps)."""
    a, b = float(inner[0]), float(inner[1])
    return t_outer0 <= a + eps and b <= t_outer1 + eps


def _pslicer_chunk_indices_fully_swallowed(
    chunks: list[tuple[float, float, str, str]],
    si: int,
    t0: float,
    t1: float,
    *,
    eps: float = 1e-4,
) -> list[int]:
    """
    After editing chunk ``si`` to ``[t0, t1]``, return every *other* chunk index whose interval
    lies fully inside ``[t0, t1]`` (not only immediate temporal neighbors). Those clips are removed
    from the AI trim preview list.
    """
    if not chunks or si < 0 or si >= len(chunks):
        return []
    out: list[int] = []
    for k in range(len(chunks)):
        if k == si:
            continue
        if _pslicer_clip_interval_subset(chunks[k], t0, t1, eps=eps):
            out.append(int(k))
    return out


class PslicerTrimPreviewDialog:
    """Temporary preview WAVs + waveform / timestamps for pslicer chunks; cleans up on close."""

    def __init__(
        self,
        app: "SanctumSurgicalV3",
        audio_path: str,
        chunks: list[tuple[float, float, str, str]],
        *,
        padding_ms: int = 120,
        aligned_words: list[dict] | None = None,
    ) -> None:
        self.app = app
        self.audio_path = str(Path(audio_path).resolve())
        self.chunks = list(chunks)
        self._aligned_words: list[dict] = list(aligned_words) if aligned_words else []
        self.padding_ms = max(0, int(padding_ms))
        self._temp_dir = tempfile.mkdtemp(prefix="sanctum_pslicer_preview_")
        self._closing = False
        self._fig_pslicer = None
        self._canvas_pslicer = None
        self._lbl_times: tk.Label | None = None
        self._lbl_text: tk.Label | None = None
        self._list: tk.Listbox | None = None
        self._var_include = tk.IntVar(value=1)
        self.top: tk.Toplevel | None = None
        self._trim_t0_var = tk.StringVar(value="0.000")
        self._trim_t1_var = tk.StringVar(value="0.000")
        self._wave_context_var = tk.StringVar(value="0.35")
        self._chunks_ai: tuple[tuple[float, float, str, str], ...] = ()

        import pslicer as _ps

        stem = Path(self.audio_path).stem
        res = _ps.export_wav_clips(
            self.audio_path,
            self._temp_dir,
            self.chunks,
            stem=stem,
            padding_ms=self.padding_ms,
            return_source_indices=True,
        )
        self.paths, self.src_idx = res
        self._included = [True] * len(self.paths)
        self._chunks_ai = tuple(tuple(row) for row in self.chunks)
        self._unique_speakers = sorted({self.chunks[self.src_idx[j]][2] for j in range(len(self.paths))})
        self._visible_j: list[int] = []
        self._view_filter_var: tk.StringVar | None = None
        self._export_preset_var: tk.StringVar | None = None
        self._export_preset_choice_list: tuple[str, ...] = ()

        if os.path.normcase(os.path.abspath(app.audio_path)) == os.path.normcase(
            self.audio_path
        ) and app.pydub_audio is not None:
            self._parent_total_ms = len(app.pydub_audio)
        else:
            self._parent_total_ms = len(AudioSegment.from_wav(self.audio_path))

        if not self.paths:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except OSError:
                pass
            messagebox.showwarning(
                "AI trim preview",
                "No clips could be exported (check audio and alignment).",
                parent=app.root,
            )
            return

        self._build_ui()

    def _build_ui(self) -> None:
        app = self.app
        u = app.ui
        top = tk.Toplevel(app.root)
        self.top = top
        top.title("AI trim — preview & export")
        top.configure(bg=BG)
        portrait = bool(getattr(app, "_portrait_chrome", False))
        top.minsize(300 if portrait else 720, 500 if portrait else 580)
        top.transient(app.root)
        try:
            top.grab_set()
        except tk.TclError:
            pass
        top.protocol("WM_DELETE_WINDOW", self._on_user_close)

        hint_main = (
            "Show clips: browse one speaker at a time. Export preset: quick all / none / one-speaker "
            "rules; use Include per clip to mix (shows as Custom). Adjust Start/End on the source file "
            "and wave preview ± to fine-tune each clip before export; the transcript updates from WhisperX "
            "words in the new window when you apply trim. Save this clip as… writes the selected clip after "
            "you pick a path; valid Start/End values are applied then (same rules as Apply trim)."
        )
        hint_trim = (
            "Dashed lines = AI sentence window; shaded = export (includes padding). Preview loads more context "
            "from the original WAV when ± is larger. Apply trim re-slices the sentence from WhisperX words in that window."
        )
        hint_bar = tk.Frame(top, bg=BG)
        hint_bar.pack(fill=tk.X, padx=u.pad_x, pady=(8, 4))
        tk.Label(
            hint_bar,
            text="AI trim — hover ? for overview" if portrait else "AI trim — hover ? for full help",
            fg=MUTED,
            bg=BG,
            font=u.font_label,
        ).pack(side=tk.LEFT, anchor=tk.W)
        q_main = tk.Label(
            hint_bar,
            text="?",
            fg=ACCENT,
            bg=BG,
            cursor="hand2",
            font=(u.font_mono[0], max(12, int(u.font_mono[1]) + 2), "bold"),
        )
        q_main.pack(side=tk.RIGHT, padx=(6, 0))
        _attach_tk_hover_tooltip(
            q_main,
            hint_main,
            wraplength=min(420 if portrait else 520, max(260, getattr(app, "_narrow_px", 520))),
        )

        body_pad = u.pad_x if not portrait else max(4, int(u.pad_x) - 2)
        body = tk.Frame(top, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=body_pad, pady=(0, 8))

        left_w = 260 if portrait else 280
        left = tk.Frame(body, bg=BG, width=left_w)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        filter_fr = tk.Frame(left, bg=BG)
        filter_fr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(filter_fr, text="Show clips", fg=MUTED, bg=BG, font=u.font_label).pack(anchor=tk.W)
        self._view_filter_var = tk.StringVar(value="All speakers")
        view_choices = ("All speakers", *self._unique_speakers)
        view_row = tk.Frame(filter_fr, bg=BG)
        view_row.pack(fill=tk.X, pady=(2, 0))
        view_menu = tk.OptionMenu(
            view_row,
            self._view_filter_var,
            *view_choices,
            command=lambda *_: self._populate_list_for_current_filter(),
        )
        view_menu.config(bg=BTN_BG, fg=TEXT, activebackground=BTN_ACTIVE, activeforeground=TEXT, highlightthickness=0)
        view_menu["menu"].config(bg=BTN_BG, fg=TEXT)
        view_menu.pack(fill=tk.X)

        export_fr = tk.Frame(left, bg=BG)
        export_fr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(export_fr, text="Export preset", fg=MUTED, bg=BG, font=u.font_label).pack(anchor=tk.W)
        self._export_preset_choice_list = (
            "All speakers (include)",
            "None (exclude all)",
            *(f"Only · {spk}" for spk in self._unique_speakers),
            "Custom (mixed)",
        )
        self._export_preset_var = tk.StringVar(value="All speakers (include)")
        export_row = tk.Frame(export_fr, bg=BG)
        export_row.pack(fill=tk.X, pady=(2, 0))
        export_menu = tk.OptionMenu(
            export_row,
            self._export_preset_var,
            *self._export_preset_choice_list,
            command=lambda *_: self._apply_export_preset(self._export_preset_var.get()),
        )
        export_menu.config(bg=BTN_BG, fg=TEXT, activebackground=BTN_ACTIVE, activeforeground=TEXT, highlightthickness=0)
        export_menu["menu"].config(bg=BTN_BG, fg=TEXT)
        export_menu.pack(fill=tk.X)

        list_fr = tk.Frame(left, bg=BG)
        list_fr.pack(fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(list_fr, orient=tk.VERTICAL)
        self._list = tk.Listbox(
            list_fr,
            height=14 if portrait else 16,
            width=26 if portrait else 36,
            font=u.font_mono,
            bg=BTN_BG,
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground=ACCENT_ON_PRIMARY,
            activestyle="none",
            yscrollcommand=sb.set,
            exportselection=False,
        )
        sb.config(command=self._list.yview)
        self._list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        right_pad = (6, 0) if portrait else (12, 0)
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=right_pad)

        wrap_r = min(520, max(160, getattr(app, "_narrow_px", 520) - (56 if portrait else 32)))
        self._lbl_times = tk.Label(right, text="", fg=ACCENT, bg=BG, font=u.font_mono, anchor=tk.W)
        self._lbl_times.pack(fill=tk.X, pady=(0, 4))
        self._lbl_text = tk.Label(
            right, text="", fg=MUTED, bg=BG, font=u.font_label, anchor=tk.W, justify=tk.LEFT, wraplength=wrap_r
        )
        self._lbl_text.pack(fill=tk.X, pady=(0, 6))

        trim_fr = tk.Frame(right, bg=BG, bd=1, relief=tk.GROOVE, highlightthickness=0)
        trim_fr.pack(fill=tk.X, pady=(0, 8))
        trim_head = tk.Frame(trim_fr, bg=BG)
        trim_head.pack(fill=tk.X, padx=6, pady=(6, 2))
        tk.Label(
            trim_head,
            text="Trim on source file (seconds)",
            fg=MUTED,
            bg=BG,
            font=u.font_label,
        ).pack(side=tk.LEFT, anchor=tk.W)
        q_trim = tk.Label(
            trim_head,
            text="?",
            fg=ACCENT,
            bg=BG,
            cursor="hand2",
            font=(u.font_mono[0], max(12, int(u.font_mono[1]) + 2), "bold"),
        )
        q_trim.pack(side=tk.RIGHT, padx=(4, 2))
        _attach_tk_hover_tooltip(q_trim, hint_trim, wraplength=min(360, wrap_r + 40))

        trim_grid = tk.Frame(trim_fr, bg=BG)
        trim_grid.pack(fill=tk.X, padx=8, pady=6)
        ent_w = 10 if portrait else 12
        if portrait:
            tk.Label(trim_grid, text="Start", fg=MUTED, bg=BG, font=u.font_label).grid(row=0, column=0, sticky=tk.W)
            e0 = tk.Entry(
                trim_grid,
                textvariable=self._trim_t0_var,
                width=ent_w,
                font=u.font_mono,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=TEXT,
            )
            e0.grid(row=0, column=1, padx=(6, 0), sticky=tk.EW)
            tk.Label(trim_grid, text="End", fg=MUTED, bg=BG, font=u.font_label).grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
            e1 = tk.Entry(
                trim_grid,
                textvariable=self._trim_t1_var,
                width=ent_w,
                font=u.font_mono,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=TEXT,
            )
            e1.grid(row=1, column=1, padx=(6, 0), sticky=tk.EW, pady=(6, 0))
            tk.Label(trim_grid, text="Wave ± (s)", fg=MUTED, bg=BG, font=u.font_label).grid(
                row=2, column=0, sticky=tk.W, pady=(6, 0)
            )
            ec = tk.Entry(
                trim_grid,
                textvariable=self._wave_context_var,
                width=ent_w,
                font=u.font_mono,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=TEXT,
            )
            ec.grid(row=2, column=1, padx=(6, 0), sticky=tk.EW, pady=(6, 0))
            trim_grid.columnconfigure(1, weight=1)
        else:
            tk.Label(trim_grid, text="Start", fg=MUTED, bg=BG, font=u.font_label).grid(row=0, column=0, sticky=tk.W)
            e0 = tk.Entry(
                trim_grid,
                textvariable=self._trim_t0_var,
                width=ent_w,
                font=u.font_mono,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=TEXT,
            )
            e0.grid(row=0, column=1, padx=(6, 16), sticky=tk.W)
            tk.Label(trim_grid, text="End", fg=MUTED, bg=BG, font=u.font_label).grid(row=0, column=2, sticky=tk.W)
            e1 = tk.Entry(
                trim_grid,
                textvariable=self._trim_t1_var,
                width=ent_w,
                font=u.font_mono,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=TEXT,
            )
            e1.grid(row=0, column=3, padx=(6, 0), sticky=tk.W)
            tk.Label(trim_grid, text="Wave preview ± (s)", fg=MUTED, bg=BG, font=u.font_label).grid(
                row=1, column=0, columnspan=2, sticky=tk.W, pady=(8, 0)
            )
            ec = tk.Entry(
                trim_grid,
                textvariable=self._wave_context_var,
                width=ent_w,
                font=u.font_mono,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=TEXT,
            )
            ec.grid(row=1, column=2, columnspan=2, padx=(6, 0), sticky=tk.W, pady=(8, 0))
        e0.bind("<Return>", lambda _e: self._apply_pslicer_trim_edits())
        e1.bind("<Return>", lambda _e: self._apply_pslicer_trim_edits())
        ec.bind("<Return>", lambda _e: self._refresh_pslicer_wave_preview())
        trim_btns = tk.Frame(trim_fr, bg=BG)
        trim_btns.pack(fill=tk.X, padx=8, pady=(0, 8))
        if portrait:
            btn_apply = tk.Button(
                trim_btns,
                text="Apply trim",
                command=self._apply_pslicer_trim_edits,
                bg=ACCENT,
                fg=ACCENT_ON_PRIMARY,
                font=u.font_btn,
                activebackground=ACCENT_HOVER,
                activeforeground=ACCENT_ON_PRIMARY,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            )
            btn_apply.pack(fill=tk.X, pady=(0, 4), ipadx=6, ipady=5)
            row_tb = tk.Frame(trim_btns, bg=BG)
            row_tb.pack(fill=tk.X)
            tk.Button(
                row_tb,
                text="Reset clip to AI",
                command=self._reset_pslicer_clip_to_ai,
                bg=BTN_BG,
                fg=TEXT,
                font=u.font_btn,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3), ipadx=4, ipady=5)
            tk.Button(
                row_tb,
                text="Update wave",
                command=self._refresh_pslicer_wave_preview,
                bg=BTN_BG,
                fg=TEXT,
                font=u.font_btn,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0), ipadx=4, ipady=5)
        else:
            tk.Button(
                trim_btns,
                text="Apply trim",
                command=self._apply_pslicer_trim_edits,
                bg=ACCENT,
                fg=ACCENT_ON_PRIMARY,
                font=u.font_btn,
                activebackground=ACCENT_HOVER,
                activeforeground=ACCENT_ON_PRIMARY,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, ipadx=8, ipady=4)
            tk.Button(
                trim_btns,
                text="Reset clip to AI",
                command=self._reset_pslicer_clip_to_ai,
                bg=BTN_BG,
                fg=TEXT,
                font=u.font_btn,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(10, 0), ipadx=8, ipady=4)
            tk.Button(
                trim_btns,
                text="Update wave",
                command=self._refresh_pslicer_wave_preview,
                bg=BTN_BG,
                fg=TEXT,
                font=u.font_btn,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(10, 0), ipadx=8, ipady=4)

        ctrl = tk.Frame(right, bg=BG)
        ctrl.pack(fill=tk.X, pady=(0, 6))
        if portrait:
            row_c0 = tk.Frame(ctrl, bg=BG)
            row_c0.pack(fill=tk.X)
            tk.Checkbutton(
                row_c0,
                text="Include in export",
                variable=self._var_include,
                fg=TEXT,
                bg=BG,
                selectcolor=BTN_BG,
                activebackground=BG,
                activeforeground=TEXT,
                font=u.font_label,
                command=self._on_include_toggle,
            ).pack(side=tk.LEFT, anchor=tk.W)
            row_c1 = tk.Frame(ctrl, bg=BG)
            row_c1.pack(fill=tk.X, pady=(4, 0))
            tk.Button(
                row_c1,
                text="Play clip",
                command=self._play_current_clip,
                bg=BTN_PLAY_BG,
                fg=ACCENT,
                activebackground=BTN_PLAY_ACTIVE,
                activeforeground=ACCENT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3), ipadx=4, ipady=5)
            tk.Button(
                row_c1,
                text="Save clip…",
                command=self._save_current_pslicer_clip_as,
                bg=BTN_BG,
                fg=TEXT,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0), ipadx=4, ipady=5)
            row_c2 = tk.Frame(ctrl, bg=BG)
            row_c2.pack(fill=tk.X, pady=(4, 0))
            tk.Button(
                row_c2,
                text="All on",
                command=self._include_all,
                bg=BTN_BG,
                fg=TEXT,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3), ipadx=4, ipady=5)
            tk.Button(
                row_c2,
                text="All off",
                command=self._exclude_all,
                bg=BTN_BG,
                fg=TEXT,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0), ipadx=4, ipady=5)
        else:
            tk.Checkbutton(
                ctrl,
                text="Include in export",
                variable=self._var_include,
                fg=TEXT,
                bg=BG,
                selectcolor=BTN_BG,
                activebackground=BG,
                activeforeground=TEXT,
                font=u.font_label,
                command=self._on_include_toggle,
            ).pack(side=tk.LEFT)
            tk.Button(
                ctrl,
                text="Play clip",
                command=self._play_current_clip,
                bg=BTN_PLAY_BG,
                fg=ACCENT,
                activebackground=BTN_PLAY_ACTIVE,
                activeforeground=ACCENT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(12, 0))
            tk.Button(
                ctrl,
                text="Save this clip as…",
                command=self._save_current_pslicer_clip_as,
                bg=BTN_BG,
                fg=TEXT,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(12, 0))
            tk.Button(
                ctrl,
                text="All on",
                command=self._include_all,
                bg=BTN_BG,
                fg=TEXT,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(12, 0))
            tk.Button(
                ctrl,
                text="All off",
                command=self._exclude_all,
                bg=BTN_BG,
                fg=TEXT,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(6, 0))

        _fig_w, _fig_h = (5.2, 1.85) if portrait else (6.5, 2.2)
        self._fig_pslicer, self._ax_pslicer = plt.subplots(figsize=(_fig_w, _fig_h), facecolor=BG, dpi=100)
        self._fig_pslicer.patch.set_facecolor(BG)
        self._ax_pslicer.set_facecolor(BG)
        self._ax_pslicer.tick_params(colors=MUTED, labelsize=u.mpl_tick)
        self._ax_pslicer.set_xlabel("Time (s)", color=MUTED, fontsize=u.mpl_axis)
        self._ax_pslicer.set_ylabel("Amplitude", color=MUTED, fontsize=u.mpl_axis)
        self._canvas_pslicer = FigureCanvasTkAgg(self._fig_pslicer, master=right)
        self._canvas_pslicer.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bottom = tk.Frame(top, bg=PANEL)
        bottom.pack(fill=tk.X, side=tk.BOTTOM, padx=u.pad_x, pady=(0, 10))
        _tmp_line = self._temp_dir
        if portrait and len(_tmp_line) > 54:
            _tmp_line = "…" + _tmp_line[-50:]
        lbl_tmpdir = tk.Label(
            bottom,
            text=f"Temp preview: {_tmp_line}",
            fg=MUTED,
            bg=PANEL,
            font=(u.font_mono[0], max(8, int(u.font_mono[1]) - 1)),
            anchor=tk.W,
            cursor="hand2" if portrait else "arrow",
        )
        lbl_tmpdir.pack(fill=tk.X, pady=(8, 6))
        _attach_tk_hover_tooltip(lbl_tmpdir, f"Full temp path:\n{self._temp_dir}", wraplength=420)
        row = tk.Frame(bottom, bg=PANEL)
        row.pack(fill=tk.X, pady=(0, 8))
        if portrait:
            tk.Button(
                row,
                text="Export to output folder",
                command=self._export_final,
                bg=ACCENT,
                fg=ACCENT_ON_PRIMARY,
                font=u.font_btn_bold,
                activebackground=ACCENT_HOVER,
                activeforeground=ACCENT_ON_PRIMARY,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(fill=tk.X, ipady=8)
            tk.Button(
                row,
                text="Cancel",
                command=self._on_user_close,
                bg=BTN_BG,
                fg=TEXT,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(fill=tk.X, pady=(8, 0), ipady=8)
        else:
            tk.Button(
                row,
                text="Export to output folder",
                command=self._export_final,
                bg=ACCENT,
                fg=ACCENT_ON_PRIMARY,
                font=u.font_btn_bold,
                activebackground=ACCENT_HOVER,
                activeforeground=ACCENT_ON_PRIMARY,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, ipadx=10, ipady=6)
            tk.Button(
                row,
                text="Cancel",
                command=self._on_user_close,
                bg=BTN_BG,
                fg=TEXT,
                activebackground=BTN_ACTIVE,
                activeforeground=TEXT,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(12, 0), ipadx=10, ipady=6)

        self._list.bind("<<ListboxSelect>>", lambda _e: self._on_list_select())
        self._populate_list_for_current_filter()

    def _clear_preview_plot(self, message: str) -> None:
        if self._ax_pslicer is None or self._canvas_pslicer is None:
            return
        self._ax_pslicer.clear()
        self._ax_pslicer.set_facecolor(BG)
        self._ax_pslicer.text(
            0.5,
            0.5,
            message,
            transform=self._ax_pslicer.transAxes,
            ha="center",
            va="center",
            color=MUTED,
        )
        self._canvas_pslicer.draw_idle()

    def _parent_duration_sec(self) -> float:
        return max(1e-9, float(self._parent_total_ms) / 1000.0)

    def _parse_float_field(self, raw: str, *, field: str) -> float | None:
        s = (raw or "").strip().replace(",", ".")
        if not s:
            messagebox.showwarning("AI trim", f"{field} is empty.", parent=self.top)
            return None
        try:
            return float(s)
        except ValueError:
            messagebox.showwarning("AI trim", f"Invalid number for {field}: {raw!r}", parent=self.top)
            return None

    def _wave_context_sec(self) -> float:
        s = (self._wave_context_var.get() or "").strip().replace(",", ".")
        try:
            v = float(s) if s else 0.35
        except ValueError:
            v = 0.35
        return float(max(0.02, min(120.0, v)))

    def _sync_trim_fields_from_chunk(self, j: int) -> None:
        if j < 0 or j >= len(self.paths):
            return
        si = self.src_idx[j]
        t0, t1, _, _ = self.chunks[si]
        self._trim_t0_var.set(f"{float(t0):.3f}")
        self._trim_t1_var.set(f"{float(t1):.3f}")

    def _push_export_name_hints_for_row(self, j: int) -> None:
        """Mirror selected clip speaker + transcript into the main window export filename hints."""
        if j < 0 or j >= len(self.paths):
            return
        si = self.src_idx[j]
        if si < 0 or si >= len(self.chunks):
            return
        _t0, _t1, spk, tx = self.chunks[si]
        try:
            self.app._set_export_name_hints(spk, tx)
        except (tk.TclError, AttributeError):
            pass

    def _rewrite_preview_clip(self, j: int) -> None:
        """Rewrite temp preview WAV for clip ``j`` from ``self.audio_path`` using current chunk times."""
        if j < 0 or j >= len(self.paths):
            return
        si = self.src_idx[j]
        t0, t1, spk, _text = self.chunks[si]
        dest = self.paths[j]
        segment = AudioSegment.from_wav(self.audio_path)
        total_ms = len(segment)
        pad = max(0, int(self.padding_ms))
        ms0 = max(0, int(float(t0) * 1000) - pad)
        ms1 = min(total_ms, int(float(t1) * 1000) + pad)
        if ms1 <= ms0:
            del segment
            return
        clip = segment[ms0:ms1]
        clip.export(dest, format="wav")
        del segment, clip
        gc.collect()

    def _refresh_chunk_transcript(self, si: int) -> None:
        """Set chunk text from aligned Whisper words overlapping the current (t0, t1)."""
        if not self._aligned_words or si < 0 or si >= len(self.chunks):
            return
        import pslicer as _ps

        t0, t1, spk, prev = self.chunks[si]
        tx = _ps.transcript_for_time_range(self._aligned_words, t0, t1)
        if not tx.strip():
            if self._chunks_ai and si < len(self._chunks_ai):
                tx = self._chunks_ai[si][3]
            else:
                tx = prev
        self.chunks[si] = (t0, t1, spk, tx)

    def _update_listbox_row_for_clip_j(self, j: int) -> None:
        if self._list is None or j not in self._visible_j:
            return
        row = self._visible_j.index(j)
        si = self.src_idx[j]
        t0, t1, spk, tx = self.chunks[si]
        snip = (tx[:40] + "…") if len(tx) > 40 else tx
        line = f"{j + 1:3d}  {t0:6.2f}–{t1:6.2f}s  {spk}  {snip}"
        self._list.delete(row)
        self._list.insert(row, line)
        self._list.selection_set(row)

    def _purge_preview_chunk_at_index(self, k: int, anchor_j: list[int]) -> None:
        """Remove chunk ``k`` from ``chunks`` / ``_chunks_ai`` and delete matching preview WAV rows."""
        for jj in range(len(self.paths) - 1, -1, -1):
            if self.src_idx[jj] == k:
                try:
                    os.remove(self.paths[jj])
                except OSError:
                    pass
                del self.paths[jj], self.src_idx[jj], self._included[jj]
                if jj < anchor_j[0]:
                    anchor_j[0] -= 1
        for jj in range(len(self.paths)):
            if self.src_idx[jj] > k:
                self.src_idx[jj] -= 1
        del self.chunks[k]
        if self._chunks_ai:
            ca = list(self._chunks_ai)
            if k < len(ca):
                ca.pop(k)
            self._chunks_ai = tuple(tuple(x) for x in ca)

    def _finalize_trim_edit_with_swallow(self, j: int, si0: int, t0: float, t1: float) -> int:
        """Refresh transcript, remove any other clip fully inside ``[t0,t1]``, update list/preview/plot."""
        self._refresh_chunk_transcript(si0)
        swallowed = _pslicer_chunk_indices_fully_swallowed(self.chunks, si0, float(t0), float(t1))
        anchor_j = [j]
        si_track = si0
        for k in sorted(swallowed, reverse=True):
            self._purge_preview_chunk_at_index(k, anchor_j)
            if k < si_track:
                si_track -= 1
        jn = anchor_j[0]
        if swallowed:
            if self.paths:
                self._unique_speakers = sorted(
                    {self.chunks[self.src_idx[jj]][2] for jj in range(len(self.paths))}
                )
            self._populate_list_for_current_filter()
        else:
            self._update_listbox_row_for_clip_j(jn)
        try:
            if self.paths and 0 <= jn < len(self.paths):
                self._rewrite_preview_clip(jn)
                self._sync_trim_fields_from_chunk(jn)
                self._refresh_meta_and_waveform(jn)
        except (IndexError, tk.TclError, RuntimeError):
            pass
        if self.paths and 0 <= jn < len(self.paths):
            self._push_export_name_hints_for_row(jn)
        return jn

    def _commit_pslicer_trim_times(self, j: int, t0: float, t1: float) -> int:
        """Apply (t0, t1) to the chunk for clip ``j``; may remove swallowed clips. Returns list row index."""
        si0 = self.src_idx[j]
        _o0, _o1, spk, text = self.chunks[si0]
        self.chunks[si0] = (float(t0), float(t1), spk, text)
        return self._finalize_trim_edit_with_swallow(j, si0, float(t0), float(t1))

    def _apply_pslicer_trim_edits(self) -> None:
        j = self._current_clip_index()
        if j is None or self.top is None:
            return
        t0 = self._parse_float_field(self._trim_t0_var.get(), field="Start (s)")
        if t0 is None:
            return
        t1 = self._parse_float_field(self._trim_t1_var.get(), field="End (s)")
        if t1 is None:
            return
        tot = self._parent_duration_sec()
        min_dur = 0.05
        if t0 < 0.0 or t1 > tot or t1 - t0 < min_dur:
            messagebox.showwarning(
                "AI trim",
                f"Need 0 ≤ start < end ≤ {tot:.3f} s and at least {min_dur:.2f} s long.",
                parent=self.top,
            )
            return
        if t0 >= t1:
            messagebox.showwarning("AI trim", "Start must be less than end.", parent=self.top)
            return
        self._commit_pslicer_trim_times(j, float(t0), float(t1))

    def _save_current_pslicer_clip_as(self) -> None:
        """Save the selected clip to a user-chosen WAV path using current Start/End (validated like Apply)."""
        import pslicer as _ps

        j = self._current_clip_index()
        if j is None or self.top is None:
            messagebox.showwarning("AI trim", "Select a clip in the list first.", parent=self.top)
            return
        t0 = self._parse_float_field(self._trim_t0_var.get(), field="Start (s)")
        if t0 is None:
            return
        t1 = self._parse_float_field(self._trim_t1_var.get(), field="End (s)")
        if t1 is None:
            return
        tot = self._parent_duration_sec()
        min_dur = 0.05
        if t0 < 0.0 or t1 > tot or t1 - t0 < min_dur:
            messagebox.showwarning(
                "AI trim",
                f"Need 0 ≤ start < end ≤ {tot:.3f} s and at least {min_dur:.2f} s long.",
                parent=self.top,
            )
            return
        if t0 >= t1:
            messagebox.showwarning("AI trim", "Start must be less than end.", parent=self.top)
            return
        si = self.src_idx[j]
        stem = Path(self.audio_path).stem
        sort_tok, _ms = _ps._chunk_start_sort_token_and_ms(float(t0))
        initial = f"{stem}_{sort_tok}_clip_{j + 1:04d}_{_ps._sanitize(self.chunks[si][2])}.wav"
        out = filedialog.asksaveasfilename(
            parent=self.top,
            title="Save this AI trim clip",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            initialdir=os.path.abspath(self.app.export_dir),
            initialfile=initial,
        )
        if not out:
            return
        j = self._commit_pslicer_trim_times(j, float(t0), float(t1))
        si = self.src_idx[j]
        t0s, t1s, _, _ = self.chunks[si]
        try:
            _ps.export_one_wav_clip(
                self.audio_path,
                str(out),
                t0s,
                t1s,
                padding_ms=self.padding_ms,
            )
        except Exception as e:
            messagebox.showerror("AI trim", f"Could not save clip:\n{e}", parent=self.top)
            return
        messagebox.showinfo("AI trim", f"Saved clip to:\n{out}", parent=self.top)

    def _reset_pslicer_clip_to_ai(self) -> None:
        j = self._current_clip_index()
        if j is None or not self._chunks_ai:
            return
        si0 = self.src_idx[j]
        if si0 >= len(self._chunks_ai):
            return
        t0, t1, spk, text = self._chunks_ai[si0]
        self.chunks[si0] = (t0, t1, spk, text)
        self._finalize_trim_edit_with_swallow(j, si0, float(t0), float(t1))

    def _refresh_pslicer_wave_preview(self) -> None:
        """Redraw waveform from the parent file using current ± context (no time change)."""
        j = self._current_clip_index()
        if j is None:
            return
        self._refresh_meta_and_waveform(j)

    def _populate_list_for_current_filter(self) -> None:
        if self._list is None or self._view_filter_var is None:
            return
        prev_j: int | None = None
        sel = self._list.curselection()
        if sel and self._visible_j and 0 <= int(sel[0]) < len(self._visible_j):
            prev_j = self._visible_j[int(sel[0])]
        vf = self._view_filter_var.get()
        if vf == "All speakers":
            self._visible_j = list(range(len(self.paths)))
        else:
            self._visible_j = [
                j for j in range(len(self.paths)) if self.chunks[self.src_idx[j]][2] == vf
            ]
        self._list.delete(0, tk.END)
        for j in self._visible_j:
            si = self.src_idx[j]
            t0, t1, spk, tx = self.chunks[si]
            snip = (tx[:40] + "…") if len(tx) > 40 else tx
            self._list.insert(tk.END, f"{j + 1:3d}  {t0:6.2f}–{t1:6.2f}s  {spk}  {snip}")
        if not self._visible_j:
            if self._lbl_times is not None:
                self._lbl_times.config(text="")
            if self._lbl_text is not None:
                self._lbl_text.config(text="")
            self._clear_preview_plot("No clips for this speaker filter.")
            return
        want_row = 0
        if prev_j is not None and prev_j in self._visible_j:
            want_row = self._visible_j.index(prev_j)
        self._list.selection_set(want_row)
        self._list.see(want_row)
        self._on_list_select()

    def _apply_export_preset(self, label: str) -> None:
        if label == "Custom (mixed)":
            return
        n = len(self.paths)
        if label == "All speakers (include)":
            self._included = [True] * n
        elif label == "None (exclude all)":
            self._included = [False] * n
        elif label.startswith("Only · "):
            spk = label[len("Only · ") :]
            self._included = [self.chunks[self.src_idx[j]][2] == spk for j in range(n)]
        else:
            return
        if self._export_preset_var is not None:
            self._export_preset_var.set(label)
        j = self._current_clip_index()
        if j is not None:
            self._var_include.set(1 if self._included[j] else 0)

    def _sync_export_preset_ui(self) -> None:
        if self._export_preset_var is None or not self.paths:
            return
        if all(self._included):
            self._export_preset_var.set("All speakers (include)")
            return
        if not any(self._included):
            self._export_preset_var.set("None (exclude all)")
            return
        for spk in self._unique_speakers:
            ok = True
            for j in range(len(self.paths)):
                if self._included[j] != (self.chunks[self.src_idx[j]][2] == spk):
                    ok = False
                    break
            if ok:
                self._export_preset_var.set(f"Only · {spk}")
                return
        self._export_preset_var.set("Custom (mixed)")

    def _current_clip_index(self) -> int | None:
        if self._list is None or not self._visible_j:
            return None
        sel = self._list.curselection()
        if not sel:
            return None
        row = int(sel[0])
        if row < 0 or row >= len(self._visible_j):
            return None
        return self._visible_j[row]

    def _on_list_select(self) -> None:
        j = self._current_clip_index()
        if j is None:
            return
        self._var_include.set(1 if self._included[j] else 0)
        self._sync_trim_fields_from_chunk(j)
        self._refresh_meta_and_waveform(j)
        self._push_export_name_hints_for_row(j)

    def _on_include_toggle(self) -> None:
        j = self._current_clip_index()
        if j is None:
            return
        self._included[j] = bool(self._var_include.get())
        self._sync_export_preset_ui()

    def _include_all(self) -> None:
        self._included = [True] * len(self.paths)
        if self._export_preset_var is not None:
            self._export_preset_var.set("All speakers (include)")
        j = self._current_clip_index()
        if j is not None:
            self._var_include.set(1)

    def _exclude_all(self) -> None:
        self._included = [False] * len(self.paths)
        if self._export_preset_var is not None:
            self._export_preset_var.set("None (exclude all)")
        j = self._current_clip_index()
        if j is not None:
            self._var_include.set(0)

    def _refresh_meta_and_waveform(self, j: int) -> None:
        if self._lbl_times is None or self._lbl_text is None or self._ax_pslicer is None:
            return
        if self._canvas_pslicer is None:
            return
        if j < 0 or j >= len(self.paths):
            return
        si = self.src_idx[j]
        t0, t1, spk, text = self.chunks[si]
        ms0 = max(0, int(t0 * 1000) - self.padding_ms)
        ms1 = min(self._parent_total_ms, int(t1 * 1000) + self.padding_ms)
        try:
            row_vis = self._visible_j.index(j) + 1
            n_vis = len(self._visible_j)
        except ValueError:
            row_vis, n_vis = 0, len(self._visible_j)
        ctx = self._wave_context_sec()
        self._lbl_times.config(
            text=(
                f"Shown {row_vis}/{n_vis} · clip #{j + 1} of {len(self.paths)} · "
                f"{t0:.3f}–{t1:.3f} s · export {ms0 / 1000:.3f}–{ms1 / 1000:.3f} s · "
                f"preview ±{ctx:.2f}s · {spk}"
            )
        )
        self._lbl_text.config(text=(text if len(text) <= 300 else text[:297] + "…"))

        tot_sec = self._parent_duration_sec()
        win0 = max(0.0, float(t0) - ctx)
        win1 = min(tot_sec, float(t1) + ctx)
        dur_win = max(win1 - win0, 0.001)
        try:
            y_full, sr = librosa.load(
                self.audio_path,
                sr=None,
                mono=True,
                offset=win0,
                duration=dur_win,
            )
        except Exception as e:
            self._ax_pslicer.clear()
            self._ax_pslicer.set_facecolor(BG)
            self._ax_pslicer.text(0.5, 0.5, f"Load error: {e}", transform=self._ax_pslicer.transAxes, ha="center", va="center", color=MUTED)
            self._canvas_pslicer.draw_idle()
            return

        sr_f = float(sr) if sr else 1.0
        total_dur = float(len(y_full)) / sr_f
        cap = 200_000
        if len(y_full) > cap:
            step = max(1, len(y_full) // cap)
            y = y_full[::step]
            t = win0 + np.linspace(0.0, total_dur, num=len(y), endpoint=False)
        else:
            y = y_full
            t = win0 + np.arange(len(y), dtype=np.float64) / sr_f
        self._ax_pslicer.clear()
        self._ax_pslicer.set_facecolor(BG)
        exp0 = ms0 / 1000.0
        exp1 = ms1 / 1000.0
        self._ax_pslicer.axvspan(exp0, exp1, alpha=0.22, color=ACCENT, zorder=1)
        self._ax_pslicer.plot(t, y, color=WAVEFORM_COLOR, lw=0.35, zorder=2)
        self._ax_pslicer.axvline(float(t0), color=MUTED, ls="--", lw=1.0, zorder=3)
        self._ax_pslicer.axvline(float(t1), color=MUTED, ls="--", lw=1.0, zorder=3)
        self._ax_pslicer.set_xlim(win0, max(win1, win0 + 1e-6))
        self._ax_pslicer.tick_params(colors=MUTED)
        self._canvas_pslicer.draw_idle()

    def _play_current_clip(self) -> None:
        j = self._current_clip_index()
        if j is None:
            return
        path = self.paths[j]
        try:
            self.app.stop_all()
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass
        try:
            import subprocess

            p = str(Path(path).resolve())
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", f"(New-Object Media.SoundPlayer '{p}').PlaySync()"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            messagebox.showwarning("Play", "Could not play this clip.", parent=self.top)

    def _export_final(self) -> None:
        import pslicer as _ps

        kept = {self.src_idx[j] for j in range(len(self.paths)) if self._included[j]}
        ordered = [self.chunks[i] for i in range(len(self.chunks)) if i in kept]
        if not ordered:
            messagebox.showwarning("AI trim", "No clips selected for export.", parent=self.top)
            return
        out_dir = os.path.abspath(self.app.export_dir)
        os.makedirs(out_dir, exist_ok=True)
        stem = Path(self.audio_path).stem
        n = len(
            _ps.export_wav_clips(
                self.audio_path,
                out_dir,
                ordered,
                stem=stem,
                padding_ms=self.padding_ms,
                verbose=False,
            )
        )
        messagebox.showinfo("AI trim", f"Exported {n} file(s) to:\n{out_dir}", parent=self.top)
        self._cleanup()

    def _on_user_close(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            self.app.stop_all()
        except Exception:
            pass
        tmp = self._temp_dir
        try:
            if self._fig_pslicer is not None:
                plt.close(self._fig_pslicer)
        except Exception:
            pass
        self._fig_pslicer = None
        self._canvas_pslicer = None
        self.paths = []
        self.src_idx = []
        if self.top is not None:
            try:
                self.top.destroy()
            except tk.TclError:
                pass
            self.top = None
        gc.collect()
        if tmp and os.path.isdir(tmp):
            for _ in range(12):
                try:
                    shutil.rmtree(tmp)
                    break
                except OSError:
                    time.sleep(0.04)
                    gc.collect()
            else:
                shutil.rmtree(tmp, ignore_errors=True)
        gc.collect()


class SanctumSurgicalV3:
    def __init__(self, root: tk.Tk):
        self.root = root
        load_voice_engine_settings_into_environ()
        try:
            root.update_idletasks()
        except tk.TclError:
            pass
        self.ui_profile = resolve_ui_profile(root)
        self.ui = ui_profile_for(self.ui_profile)

        self._remote_session = _is_remote_desktop_session()
        try:
            self._screen_w = int(root.winfo_screenwidth())
            self._screen_h = int(root.winfo_screenheight())
        except tk.TclError:
            self._screen_w, self._screen_h = 1280, 800

        self.root.title("Voice Engine")
        avail_w, avail_h, gw0, gh0 = _parse_work_area(
            self.ui.geometry, self._screen_w, self._screen_h, self._remote_session
        )
        self._narrow_width = self._screen_w < 1320
        self._low_height = avail_h < 940
        # Portrait / narrow width: stacked export, two tool rows, split transport (saves horizontal space).
        # Wide landscape RDP must NOT use this — it steals vertical space from the waveform.
        self._portrait_chrome = self._narrow_width
        # Slimmer padding + use full work-area height (remote, narrow, or short landscape).
        self._slim_chrome = self._remote_session or self._narrow_width or self._low_height
        self._narrow_px = max(280, min(self._screen_w, self._screen_h))
        # Touch UI is tall; on short work areas (common in RDP) the hint eats the waveform.
        self._skip_hint = self.ui_profile == "touch" and avail_h < 1280
        min_w = max(280, min(self.ui.minsize_w, avail_w))
        min_h = max(320, min(self.ui.minsize_h, avail_h))
        gw = max(min_w, min(gw0, avail_w))
        gh = max(min_h, min(gh0, avail_h))
        # Narrow + touch-style chrome stacks two tool rows and a tall hint; default profile
        # height can exceed the window and Tk collapses the plot/transport to 1px.
        if self._slim_chrome:
            # Tall enough for chrome + plot; cap so mis-reported session sizes do not
            # create a multi-thousand-pixel window on one monitor.
            gh = max(gh, min(avail_h, 1600))
        self.root.geometry(f"{gw}x{gh}")
        self.root.minsize(min_w, min_h)
        self.root.configure(bg=BG)
        if self.ui.force_zoomed:
            try:
                self.root.state("zoomed")
            except tk.TclError:
                pass

        try:
            s = float(self.root.tk.call("tk", "scaling"))
            if s < self.ui.tk_scale_min:
                self.root.tk.call("tk", "scaling", max(s, self.ui.tk_scale_min))
        except tk.TclError:
            pass

        self.pydub_audio: AudioSegment | None = None
        self.audio_path = ""
        self.duration_sec = 0.0
        self._y = None
        self._sr = None

        self.span: SpanSelector | None = None
        self._play_kind: str | None = None
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._playback_temp: str | None = None
        self._selection_stop_after = None
        self._preview_after = None
        self._preview_bump_after = None
        self._clock_after = None
        self._closing = False
        self._export_busy = False
        self._export_speaker_hint: str = ""
        self._export_transcript_hint: str = ""
        self._export_name_after: str | None = None
        self._pslicer_busy = False
        self._pslicer_phase_queue: queue.Queue[str] | None = None
        self._pslicer_load_win: tk.Toplevel | None = None
        self._pslicer_load_lbl: tk.Label | None = None
        self._pslicer_load_pb: ttk.Progressbar | None = None
        self._pslicer_load_poll_id: str | None = None
        self._pslicer_done_after_id: str | None = None
        self._settings_win: tk.Toplevel | None = None
        self._zoom_debounce_after = None
        self._plot_resize_after = None
        self._viewport_sync_after: str | None = None
        self._live_portrait: bool | None = None
        self._hint_label: tk.Label | None = None
        self._topmost_clear_after: str | None = None
        self.var_auto_zoom = tk.IntVar(value=1)
        self._last_clock_text: str | None = None
        self._last_play_end_ms: int | None = None
        self._last_saved_trim_end_ms: int | None = None
        self._ylim_adjust_depth = 0
        self._xlim_cid: int | None = None
        self.export_dir: str = os.path.abspath(DEFAULT_OUT)

        self._mixer_initialized = False
        try:
            pygame.mixer.init()
            self._mixer_initialized = True
        except pygame.error:
            pass

        u = self.ui
        # Pack bottom chrome first so remaining space goes to the matplotlib pane (Tk pack + TOP/BOTTOM).
        exp = tk.Frame(root, bg=PANEL)
        self.fr_exp = exp
        exp_pad_bottom = (
            10 + (32 if self._remote_session else 0) + (36 if self._slim_chrome else 0)
        )
        self._exp_pad_bottom = exp_pad_bottom
        exp.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, exp_pad_bottom))
        out_pick = tk.Frame(exp, bg=PANEL)
        out_pick.pack(
            fill=tk.X,
            padx=u.pad_x,
            pady=(6, 0) if self._portrait_chrome else (14, 0),
        )
        self.var_loop = tk.IntVar(value=1)
        self.lbl_export_dir = None  # path only via folder dialog

        if self._portrait_chrome:
            tk.Button(
                out_pick,
                text="Output folder",
                command=self.choose_export_directory,
                bg=BTN_BG,
                fg=TEXT,
                activeforeground=TEXT,
                activebackground=BTN_ACTIVE,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(
                fill=tk.X,
                ipadx=min(10, u.browse_ipadx),
                ipady=4,
                pady=(0, 0),
            )
        else:
            tk.Button(
                out_pick,
                text="Output folder",
                command=self.choose_export_directory,
                bg=BTN_BG,
                fg=TEXT,
                activeforeground=TEXT,
                activebackground=BTN_ACTIVE,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(
                side=tk.LEFT,
                ipadx=u.browse_ipadx,
                ipady=u.browse_ipady,
                padx=0,
                pady=u.pad_y_tight,
            )

        inner = tk.Frame(exp, bg=PANEL)
        inner.pack(
            fill=tk.X,
            padx=u.pad_x,
            pady=(6, 8) if self._portrait_chrome else u.pad_x,
        )
        if self._portrait_chrome:
            _exp_lbl = (u.font_label[0], max(9, int(u.font_label[1]) - 1))
            tk.Label(inner, text="Filename", fg=TEXT, bg=PANEL, font=_exp_lbl).pack(
                anchor=tk.W, fill=tk.X, pady=(0, 2)
            )
            self.ent_name = tk.Entry(
                inner,
                width=1,
                font=_exp_lbl,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=ACCENT,
            )
            self.ent_name.pack(fill=tk.X, pady=(0, 2), ipady=4)
            self._apply_suggested_export_name()
            tk.Checkbutton(
                inner,
                text="Loop",
                variable=self.var_loop,
                fg=TEXT,
                bg=PANEL,
                selectcolor=BTN_BG,
                activebackground=PANEL,
                activeforeground=TEXT,
                font=_exp_lbl,
                command=self._on_loop_toggle,
            ).pack(anchor=tk.W, pady=(0, 2))
            self.btn_save = tk.Button(
                inner,
                text="Export",
                command=self.save_trim,
                bg=ACCENT,
                fg=ACCENT_ON_PRIMARY,
                font=u.font_btn_bold,
                activeforeground=ACCENT_ON_PRIMARY,
                activebackground=ACCENT_HOVER,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            )
            self.btn_save.pack(
                fill=tk.X,
                ipadx=min(12, u.save_ipadx),
                ipady=6,
                pady=(0, 0),
            )
        else:
            tk.Label(inner, text="Filename", fg=TEXT, bg=PANEL, font=u.font_label).pack(
                side=tk.LEFT
            )
            self.ent_name = tk.Entry(
                inner,
                width=u.entry_w,
                font=u.font_label,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=ACCENT,
            )
            self.ent_name.pack(side=tk.LEFT, padx=u.pad_x, pady=u.pad_y_tight, ipady=u.entry_ipady)
            self._apply_suggested_export_name()
            tk.Checkbutton(
                inner,
                text="Loop",
                variable=self.var_loop,
                fg=TEXT,
                bg=PANEL,
                selectcolor=BTN_BG,
                activebackground=PANEL,
                activeforeground=TEXT,
                font=u.font_label,
                command=self._on_loop_toggle,
            ).pack(side=tk.LEFT, padx=u.pad_x, pady=u.pad_y_tight)
            self.btn_save = tk.Button(
                inner,
                text="Export",
                command=self.save_trim,
                bg=ACCENT,
                fg=ACCENT_ON_PRIMARY,
                font=u.font_btn_bold,
                activeforeground=ACCENT_ON_PRIMARY,
                activebackground=ACCENT_HOVER,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            )
            self.btn_save.pack(
                side=tk.RIGHT,
                padx=u.pad_x_tight,
                pady=u.pad_y_tight,
                ipadx=u.save_ipadx,
                ipady=u.save_ipady,
            )

        # --- Chrome ---
        _hdr_h = u.header_h
        if self._low_height and not self._portrait_chrome:
            _hdr_h = max(36, u.header_h - 10)
        elif self.ui_profile == "touch" and self._portrait_chrome and avail_h < 1320:
            _hdr_h = max(44, u.header_h - 10)
        header = tk.Frame(root, bg=PANEL, height=_hdr_h)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Voice Engine",
            fg=ACCENT,
            bg=PANEL,
            font=u.font_header,
        ).pack(side=tk.LEFT, padx=u.pad_x, pady=u.pad_y_tight)
        tk.Button(
            header,
            text="Settings…",
            command=self._open_settings_dialog,
            bg=BTN_BG,
            fg=TEXT,
            activeforeground=TEXT,
            activebackground=BTN_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(side=tk.RIGHT, padx=u.pad_x, pady=u.pad_y_tight)

        top = tk.Frame(root, bg=BG)
        self.fr_top = top
        top.pack(side=tk.TOP, fill=tk.X, pady=(u.pad_y, u.pad_y_tight))
        top_load = tk.Frame(top, bg=BG)
        self.fr_top_load = top_load
        top_load.pack(fill=tk.X)
        tk.Button(
            top_load,
            text="Open…",
            command=self.load_file,
            bg=BTN_BG,
            fg=TEXT,
            activeforeground=TEXT,
            activebackground=BTN_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(
            side=tk.LEFT,
            padx=u.pad_x,
            pady=u.pad_y_tight,
            ipadx=u.btn_ipadx,
            ipady=u.btn_ipady,
        )
        self.lbl_file = tk.Label(top, text="No file loaded", fg=ACCENT, bg=BG, font=u.font_label)
        if self._portrait_chrome:
            self.lbl_file.pack(anchor=tk.W, fill=tk.X, padx=u.pad_x, pady=(4, 0))
        else:
            self.lbl_file.pack(side=tk.LEFT, padx=u.pad_x, pady=u.pad_y_tight, in_=top_load)

        # --- Trim / view tools (two rows only when width is tight; landscape uses one row) ---
        tools = tk.Frame(root, bg=BG)
        self.fr_tools = tools
        tools_toolbar_pady = (2, 4) if self._slim_chrome else (4, 6)
        self._tools_toolbar_pady = tools_toolbar_pady
        tools.pack(fill=tk.X, padx=u.pad_x, pady=tools_toolbar_pady)
        if self._portrait_chrome:
            tools_r1 = tk.Frame(tools, bg=BG)
            tools_r1.pack(fill=tk.X)
            tools_r2 = tk.Frame(tools, bg=BG)
            tools_r2.pack(fill=tk.X, pady=(6, 0))
            tools_trim_row, tools_view_row = tools_r1, tools_r2
        else:
            tools_trim_row = tools_view_row = tools

        tk.Button(
            tools_trim_row,
            text="Set start",
            command=self.mark_trim_start,
            bg=BTN_TRIM_START_BG,
            fg=TEXT,
            activeforeground=TEXT,
            activebackground=BTN_TRIM_START_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(
            side=tk.LEFT,
            padx=u.pad_x,
            pady=u.pad_y_tight,
            ipadx=u.trim_ipadx,
            ipady=u.trim_ipady,
        )
        tk.Button(
            tools_trim_row,
            text="Set end",
            command=self.mark_trim_end,
            bg=BTN_TRIM_END_BG,
            fg=TEXT,
            activeforeground=TEXT,
            activebackground=BTN_TRIM_END_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(
            side=tk.LEFT,
            padx=u.pad_x,
            pady=u.pad_y_tight,
            ipadx=u.trim_ipadx,
            ipady=u.trim_ipady,
        )
        self.btn_pslicer = tk.Button(
            tools_trim_row,
            text="AI trim…",
            command=self.start_pslicer_auto_trim,
            bg=BTN_BG,
            fg=ACCENT,
            activeforeground=ACCENT,
            activebackground=BTN_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        self.btn_pslicer.pack(
            side=tk.LEFT,
            padx=u.pad_x,
            pady=u.pad_y_tight,
            ipadx=u.btn_ipadx,
            ipady=u.btn_ipady,
        )
        self._btn(tools_trim_row, "All", self.select_all_waveform)
        self._btn(tools_view_row, "Zoom", self.zoom_to_selection)
        self._btn(tools_view_row, "Fit", self.reset_view)
        tk.Checkbutton(
            tools_view_row,
            text="Auto-zoom",
            variable=self.var_auto_zoom,
            fg=TEXT,
            bg=BG,
            selectcolor=BTN_BG,
            activebackground=BG,
            activeforeground=TEXT,
            font=u.font_label,
            indicatoron=1,
        ).pack(side=tk.LEFT, padx=(u.pad_x * 2, u.pad_x), pady=u.pad_y_tight)

        _wrap_basis = self._narrow_px if (self._portrait_chrome or self._low_height) else self._screen_w
        hint_wrap = min(u.hint_wrap, max(240, _wrap_basis - 2 * u.pad_x))
        if self._low_height and not self._portrait_chrome:
            hint_text = "Space play/pause · ←/→ seek · Esc · click waveform to focus"
            _fl = u.font_label
            _sz = max(8, int(_fl[1]) - 2)
            hint_font = (_fl[0], _sz, *_fl[2:]) if len(_fl) > 2 else (_fl[0], _sz)
        elif self._portrait_chrome or self._low_height:
            hint_text = u.hint
            _fl = u.font_label
            _sz = max(9, int(_fl[1]) - 2)
            hint_font = (_fl[0], _sz, *_fl[2:]) if len(_fl) > 2 else (_fl[0], _sz)
        else:
            hint_text = u.hint
            hint_font = u.font_label
        if not self._skip_hint:
            self._hint_label = tk.Label(
                root,
                text=hint_text,
                fg=MUTED,
                bg=BG,
                font=hint_font,
                wraplength=hint_wrap,
                justify=tk.LEFT,
            )
            self._hint_label.pack(fill=tk.X, padx=u.pad_x, pady=(0, 2 if self._slim_chrome else 4))

        # --- Typed trim times (seconds) ---
        manual = tk.Frame(root, bg=BG)
        self.fr_manual = manual
        manual.pack(fill=tk.X, padx=u.pad_x, pady=(2, 4) if self._slim_chrome else (2, 6))
        manual_row = tk.Frame(manual, bg=BG)
        manual_row.pack(fill=tk.X)
        tk.Label(manual_row, text="Start s", fg=MUTED, bg=BG, font=u.font_label).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self.ent_trim_start = tk.Entry(
            manual_row,
            width=12,
            font=u.font_mono,
            bg=BTN_BG,
            fg=TEXT,
            insertbackground=ACCENT,
        )
        self.ent_trim_start.pack(side=tk.LEFT, padx=(0, u.pad_x), ipady=6)
        tk.Label(manual_row, text="End s", fg=MUTED, bg=BG, font=u.font_label).pack(
            side=tk.LEFT, padx=(4, 6)
        )
        self.ent_trim_end = tk.Entry(
            manual_row,
            width=12,
            font=u.font_mono,
            bg=BTN_BG,
            fg=TEXT,
            insertbackground=ACCENT,
        )
        self.ent_trim_end.pack(side=tk.LEFT, padx=(0, u.pad_x), ipady=6)
        _apply_btn_kw = dict(
            text="Apply",
            command=self.apply_typed_trim_times,
            bg=BTN_BG,
            fg=ACCENT,
            activeforeground=ACCENT,
            activebackground=BTN_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        if self._portrait_chrome:
            tk.Button(manual, **_apply_btn_kw).pack(
                fill=tk.X, pady=(8, 0), ipadx=u.btn_ipadx, ipady=8
            )
        else:
            tk.Button(manual_row, **_apply_btn_kw).pack(
                side=tk.LEFT, padx=u.pad_x, ipadx=u.btn_ipadx, ipady=8
            )

        # --- Transport: seek | Play | seek + time ---
        transport = tk.Frame(root, bg=BG)
        self.fr_transport = transport
        transport.pack(
            fill=tk.X,
            padx=u.pad_x,
            pady=(4, 6) if self._slim_chrome else (8, 10),
        )
        if self._portrait_chrome:
            t_row1 = tk.Frame(transport, bg=BG)
            t_row1.pack(fill=tk.X)
            t_row1.grid_columnconfigure(0, weight=1)
            t_row1.grid_columnconfigure(1, weight=0)
            t_row1.grid_columnconfigure(2, weight=1)
            t_left = tk.Frame(t_row1, bg=BG)
            t_left.grid(row=0, column=0, sticky="e", padx=(0, u.pad_x))
            t_mid = tk.Frame(t_row1, bg=BG)
            t_mid.grid(row=0, column=1, sticky="")
            t_right = tk.Frame(t_row1, bg=BG)
            t_right.grid(row=0, column=2, sticky="w", padx=(u.pad_x, 0))
            t_bottom = tk.Frame(transport, bg=BG)
            t_bottom.pack(fill=tk.X, pady=(8, 0))
        else:
            transport.grid_columnconfigure(0, weight=1)
            transport.grid_columnconfigure(1, weight=0)
            transport.grid_columnconfigure(2, weight=1)
            t_left = tk.Frame(transport, bg=BG)
            t_left.grid(row=0, column=0, sticky="w", padx=(0, u.pad_x))
            t_mid = tk.Frame(transport, bg=BG)
            t_mid.grid(row=0, column=1, sticky="")
            t_right = tk.Frame(transport, bg=BG)
            t_right.grid(row=0, column=2, sticky="e", padx=(u.pad_x, 0))
            t_bottom = t_right

        self._btn(t_left, "−5s", lambda: self.seek_ms(-SEEK_STEP_MS), w=6)

        self.btn_play = tk.Button(
            t_mid,
            text="Play",
            command=self._on_play_button,
            bg=BTN_PLAY_BG,
            fg=ACCENT,
            activebackground=BTN_PLAY_ACTIVE,
            activeforeground=ACCENT,
            font=u.font_toggle,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        _pt_ipadx, _pt_ipady = u.pad_toggle_x, u.pad_toggle_y
        _pt_pady = u.pad_y
        if self._low_height and not self._portrait_chrome:
            _pt_ipadx = max(8, u.pad_toggle_x - 4)
            _pt_ipady = max(8, u.pad_toggle_y - 4)
            _pt_pady = max(4, u.pad_y - 4)
        self.btn_play.pack(ipadx=_pt_ipadx, ipady=_pt_ipady, padx=u.pad_x, pady=_pt_pady)
        self.btn_toggle_play = self.btn_play

        self._btn(t_right, "+5s", lambda: self.seek_ms(SEEK_STEP_MS), w=6)
        if self._portrait_chrome:
            self._btn(t_bottom, "Continue", self.play_from_last_end)
            self.lbl_time = tk.Label(t_bottom, text="", fg=ACCENT, bg=BG, font=u.font_mono)
            self.lbl_time.pack(side=tk.LEFT, padx=(u.pad_x * 2, u.pad_x), pady=u.pad_y_tight)
        else:
            self._btn(t_right, "Continue", self.play_from_last_end)
            self.lbl_time = tk.Label(t_right, text="", fg=ACCENT, bg=BG, font=u.font_mono)
            self.lbl_time.pack(side=tk.LEFT, padx=(u.pad_x * 2, u.pad_x), pady=u.pad_y)

        # --- Matplotlib figure (widget packed after bottom chrome so the plot area gets real height) ---
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(u.plot_fig_w, u.plot_fig_h), facecolor=BG, dpi=100)
        self.fig.patch.set_facecolor(BG)
        self.ax.set_facecolor(BG)
        self.ax.tick_params(colors=MUTED, labelsize=u.mpl_tick)
        self.ax.set_xlabel("Time (s)", color=MUTED, fontsize=u.mpl_axis)
        self.ax.set_ylabel("Amplitude", color=MUTED, fontsize=u.mpl_axis)
        self.ax.set_title(
            "Drag selection · markers follow playhead",
            color=MUTED,
            fontsize=u.mpl_title,
        )

        self.plot_wrap = tk.Frame(root, bg=BG)
        _plot_edge = 4 if (self._low_height and not self._portrait_chrome) else 6
        self.plot_wrap.pack(fill=tk.BOTH, expand=True, padx=u.pad_x_tight, pady=_plot_edge)
        self.plot_wrap.bind("<Configure>", self._on_plot_wrap_configure)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_wrap)
        self._plot_tk = self.canvas.get_tk_widget()
        self._plot_tk.pack(fill=tk.BOTH, expand=True)
        self._plot_tk.bind("<Button-1>", self._on_plot_click_focus, add="+")
        self.canvas.mpl_connect("key_press_event", self._on_mpl_key)

        self._bind_global_keys()
        self._empty_plot()
        self.root.bind("<Map>", self._on_root_map, add="+")
        self.root.bind("<Configure>", self._schedule_viewport_sync, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_clock()
        try:
            self.root.after_idle(self._resize_figure_to_plot_wrap)
        except tk.TclError:
            pass
        try:
            self.root.after(120, self._apply_viewport_sync)
        except tk.TclError:
            pass

    def _schedule_viewport_sync(self, event=None):
        if self._closing:
            return
        if event is not None and getattr(event, "widget", None) is not self.root:
            return
        if self._viewport_sync_after is not None:
            try:
                self.root.after_cancel(self._viewport_sync_after)
            except (tk.TclError, ValueError):
                pass
        self._viewport_sync_after = self.root.after(280, self._apply_viewport_sync)

    def _apply_viewport_sync(self):
        self._viewport_sync_after = None
        if self._closing:
            return
        try:
            rw = int(self.root.winfo_width())
            rh = int(self.root.winfo_height())
        except tk.TclError:
            return
        if rw < 160 or rh < 160:
            return
        self._remote_session = _is_remote_desktop_session()
        try:
            self._screen_w = int(self.root.winfo_screenwidth())
            self._screen_h = int(self.root.winfo_screenheight())
        except tk.TclError:
            pass
        self._narrow_px = max(280, min(self._screen_w, self._screen_h))
        portrait = rw < 1320
        self._low_height = rh < 940
        self._slim_chrome = self._remote_session or portrait or self._low_height

        if self._live_portrait is None:
            if portrait != self._portrait_chrome:
                self._portrait_chrome = portrait
                if not self._export_busy:
                    self._rebuild_responsive_chrome()
            self._live_portrait = portrait
            try:
                self.fr_tools.pack_configure(
                    pady=(2, 4) if self._slim_chrome else (4, 6),
                )
            except tk.TclError:
                pass
        elif portrait != self._live_portrait:
            self._live_portrait = portrait
            self._portrait_chrome = portrait
            if not self._export_busy:
                self._rebuild_responsive_chrome()
            try:
                self.fr_tools.pack_configure(
                    pady=(2, 4) if self._slim_chrome else (4, 6),
                )
            except tk.TclError:
                pass

        self._update_hint_for_viewport(rw)
        try:
            self._resize_figure_to_plot_wrap()
        except Exception:
            pass

    def _update_hint_for_viewport(self, rw: int):
        if self._hint_label is None or self._closing:
            return
        try:
            u = self.ui
            wrap = min(u.hint_wrap, max(200, rw - 2 * u.pad_x))
            self._hint_label.config(wraplength=wrap)
        except tk.TclError:
            pass

    def _rebuild_responsive_chrome(self):
        """Rebuild export / tools / transport / manual / file row for narrow vs wide width."""
        if self._closing:
            return
        u = self.ui
        portrait = self._portrait_chrome
        try:
            name_val = self.ent_name.get().strip()
        except tk.TclError:
            name_val = ""
        if not name_val:
            name_val = suggest_next_export_filename(self.export_dir)
        try:
            t0 = self.ent_trim_start.get()
            t1 = self.ent_trim_end.get()
        except tk.TclError:
            t0, t1 = "", ""

        for fr in (self.fr_exp, self.fr_tools, self.fr_transport, self.fr_manual):
            for w in list(fr.winfo_children()):
                w.destroy()

        # --- Export ---
        out_pick = tk.Frame(self.fr_exp, bg=PANEL)
        out_pick.pack(
            fill=tk.X,
            padx=u.pad_x,
            pady=(6, 0) if portrait else (14, 0),
        )
        self.lbl_export_dir = None
        if portrait:
            tk.Button(
                out_pick,
                text="Output folder",
                command=self.choose_export_directory,
                bg=BTN_BG,
                fg=TEXT,
                activeforeground=TEXT,
                activebackground=BTN_ACTIVE,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(fill=tk.X, ipadx=min(10, u.browse_ipadx), ipady=4, pady=(0, 0))
        else:
            tk.Button(
                out_pick,
                text="Output folder",
                command=self.choose_export_directory,
                bg=BTN_BG,
                fg=TEXT,
                activeforeground=TEXT,
                activebackground=BTN_ACTIVE,
                font=u.font_btn,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            ).pack(
                side=tk.LEFT,
                ipadx=u.browse_ipadx,
                ipady=u.browse_ipady,
                padx=0,
                pady=u.pad_y_tight,
            )

        inner = tk.Frame(self.fr_exp, bg=PANEL)
        inner.pack(fill=tk.X, padx=u.pad_x, pady=(6, 8) if portrait else u.pad_x)
        if portrait:
            _exp_lbl = (u.font_label[0], max(9, int(u.font_label[1]) - 1))
            tk.Label(inner, text="Filename", fg=TEXT, bg=PANEL, font=_exp_lbl).pack(
                anchor=tk.W, fill=tk.X, pady=(0, 2)
            )
            self.ent_name = tk.Entry(
                inner, width=1, font=_exp_lbl, bg=BTN_BG, fg=TEXT, insertbackground=ACCENT
            )
            self.ent_name.insert(0, name_val)
            self.ent_name.pack(fill=tk.X, pady=(0, 2), ipady=4)
            tk.Checkbutton(
                inner,
                text="Loop",
                variable=self.var_loop,
                fg=TEXT,
                bg=PANEL,
                selectcolor=BTN_BG,
                activebackground=PANEL,
                activeforeground=TEXT,
                font=_exp_lbl,
                command=self._on_loop_toggle,
            ).pack(anchor=tk.W, pady=(0, 2))
            self.btn_save = tk.Button(
                inner,
                text="Export",
                command=self.save_trim,
                bg=ACCENT,
                fg=ACCENT_ON_PRIMARY,
                font=u.font_btn_bold,
                activeforeground=ACCENT_ON_PRIMARY,
                activebackground=ACCENT_HOVER,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            )
            self.btn_save.pack(
                fill=tk.X, ipadx=min(12, u.save_ipadx), ipady=6, pady=(0, 0)
            )
        else:
            tk.Label(inner, text="Filename", fg=TEXT, bg=PANEL, font=u.font_label).pack(
                side=tk.LEFT
            )
            self.ent_name = tk.Entry(
                inner,
                width=u.entry_w,
                font=u.font_label,
                bg=BTN_BG,
                fg=TEXT,
                insertbackground=ACCENT,
            )
            self.ent_name.insert(0, name_val)
            self.ent_name.pack(side=tk.LEFT, padx=u.pad_x, pady=u.pad_y_tight, ipady=u.entry_ipady)
            tk.Checkbutton(
                inner,
                text="Loop",
                variable=self.var_loop,
                fg=TEXT,
                bg=PANEL,
                selectcolor=BTN_BG,
                activebackground=PANEL,
                activeforeground=TEXT,
                font=u.font_label,
                command=self._on_loop_toggle,
            ).pack(side=tk.LEFT, padx=u.pad_x, pady=u.pad_y_tight)
            self.btn_save = tk.Button(
                inner,
                text="Export",
                command=self.save_trim,
                bg=ACCENT,
                fg=ACCENT_ON_PRIMARY,
                font=u.font_btn_bold,
                activeforeground=ACCENT_ON_PRIMARY,
                activebackground=ACCENT_HOVER,
                relief=tk.FLAT,
                borderwidth=0,
                highlightthickness=0,
            )
            self.btn_save.pack(
                side=tk.RIGHT,
                padx=u.pad_x_tight,
                pady=u.pad_y_tight,
                ipadx=u.save_ipadx,
                ipady=u.save_ipady,
            )

        # --- Tools ---
        if portrait:
            tools_r1 = tk.Frame(self.fr_tools, bg=BG)
            tools_r1.pack(fill=tk.X)
            tools_r2 = tk.Frame(self.fr_tools, bg=BG)
            tools_r2.pack(fill=tk.X, pady=(6, 0))
            tools_trim_row, tools_view_row = tools_r1, tools_r2
        else:
            tools_trim_row = tools_view_row = self.fr_tools

        tk.Button(
            tools_trim_row,
            text="Set start",
            command=self.mark_trim_start,
            bg=BTN_TRIM_START_BG,
            fg=TEXT,
            activeforeground=TEXT,
            activebackground=BTN_TRIM_START_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(
            side=tk.LEFT,
            padx=u.pad_x,
            pady=u.pad_y_tight,
            ipadx=u.trim_ipadx,
            ipady=u.trim_ipady,
        )
        tk.Button(
            tools_trim_row,
            text="Set end",
            command=self.mark_trim_end,
            bg=BTN_TRIM_END_BG,
            fg=TEXT,
            activeforeground=TEXT,
            activebackground=BTN_TRIM_END_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(
            side=tk.LEFT,
            padx=u.pad_x,
            pady=u.pad_y_tight,
            ipadx=u.trim_ipadx,
            ipady=u.trim_ipady,
        )
        self.btn_pslicer = tk.Button(
            tools_trim_row,
            text="AI trim…",
            command=self.start_pslicer_auto_trim,
            bg=BTN_BG,
            fg=ACCENT,
            activeforeground=ACCENT,
            activebackground=BTN_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        self.btn_pslicer.pack(
            side=tk.LEFT,
            padx=u.pad_x,
            pady=u.pad_y_tight,
            ipadx=u.btn_ipadx,
            ipady=u.btn_ipady,
        )
        self._btn(tools_trim_row, "All", self.select_all_waveform)
        self._btn(tools_view_row, "Zoom", self.zoom_to_selection)
        self._btn(tools_view_row, "Fit", self.reset_view)
        tk.Checkbutton(
            tools_view_row,
            text="Auto-zoom",
            variable=self.var_auto_zoom,
            fg=TEXT,
            bg=BG,
            selectcolor=BTN_BG,
            activebackground=BG,
            activeforeground=TEXT,
            font=u.font_label,
            indicatoron=1,
        ).pack(side=tk.LEFT, padx=(u.pad_x * 2, u.pad_x), pady=u.pad_y_tight)

        # --- Manual ---
        manual_row = tk.Frame(self.fr_manual, bg=BG)
        manual_row.pack(fill=tk.X)
        tk.Label(manual_row, text="Start s", fg=MUTED, bg=BG, font=u.font_label).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        self.ent_trim_start = tk.Entry(
            manual_row,
            width=12,
            font=u.font_mono,
            bg=BTN_BG,
            fg=TEXT,
            insertbackground=ACCENT,
        )
        self.ent_trim_start.pack(side=tk.LEFT, padx=(0, u.pad_x), ipady=6)
        tk.Label(manual_row, text="End s", fg=MUTED, bg=BG, font=u.font_label).pack(
            side=tk.LEFT, padx=(4, 6)
        )
        self.ent_trim_end = tk.Entry(
            manual_row,
            width=12,
            font=u.font_mono,
            bg=BTN_BG,
            fg=TEXT,
            insertbackground=ACCENT,
        )
        self.ent_trim_end.pack(side=tk.LEFT, padx=(0, u.pad_x), ipady=6)
        _apply_btn_kw = dict(
            text="Apply",
            command=self.apply_typed_trim_times,
            bg=BTN_BG,
            fg=ACCENT,
            activeforeground=ACCENT,
            activebackground=BTN_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        if portrait:
            tk.Button(self.fr_manual, **_apply_btn_kw).pack(
                fill=tk.X, pady=(8, 0), ipadx=u.btn_ipadx, ipady=8
            )
        else:
            tk.Button(manual_row, **_apply_btn_kw).pack(
                side=tk.LEFT, padx=u.pad_x, ipadx=u.btn_ipadx, ipady=8
            )
        try:
            self.ent_trim_start.delete(0, tk.END)
            self.ent_trim_start.insert(0, t0)
            self.ent_trim_end.delete(0, tk.END)
            self.ent_trim_end.insert(0, t1)
        except tk.TclError:
            pass

        # --- Transport ---
        self.fr_transport.pack_configure(
            pady=(4, 6) if self._slim_chrome else (8, 10),
        )
        if portrait:
            t_row1 = tk.Frame(self.fr_transport, bg=BG)
            t_row1.pack(fill=tk.X)
            t_row1.grid_columnconfigure(0, weight=1)
            t_row1.grid_columnconfigure(1, weight=0)
            t_row1.grid_columnconfigure(2, weight=1)
            t_left = tk.Frame(t_row1, bg=BG)
            t_left.grid(row=0, column=0, sticky="e", padx=(0, u.pad_x))
            t_mid = tk.Frame(t_row1, bg=BG)
            t_mid.grid(row=0, column=1, sticky="")
            t_right = tk.Frame(t_row1, bg=BG)
            t_right.grid(row=0, column=2, sticky="w", padx=(u.pad_x, 0))
            t_bottom = tk.Frame(self.fr_transport, bg=BG)
            t_bottom.pack(fill=tk.X, pady=(8, 0))
        else:
            self.fr_transport.grid_columnconfigure(0, weight=1)
            self.fr_transport.grid_columnconfigure(1, weight=0)
            self.fr_transport.grid_columnconfigure(2, weight=1)
            t_left = tk.Frame(self.fr_transport, bg=BG)
            t_left.grid(row=0, column=0, sticky="w", padx=(0, u.pad_x))
            t_mid = tk.Frame(self.fr_transport, bg=BG)
            t_mid.grid(row=0, column=1, sticky="")
            t_right = tk.Frame(self.fr_transport, bg=BG)
            t_right.grid(row=0, column=2, sticky="e", padx=(u.pad_x, 0))
            t_bottom = t_right

        self._btn(t_left, "−5s", lambda: self.seek_ms(-SEEK_STEP_MS), w=6)
        self.btn_play = tk.Button(
            t_mid,
            text="Play",
            command=self._on_play_button,
            bg=BTN_PLAY_BG,
            fg=ACCENT,
            activebackground=BTN_PLAY_ACTIVE,
            activeforeground=ACCENT,
            font=u.font_toggle,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        _pt_ipadx, _pt_ipady = u.pad_toggle_x, u.pad_toggle_y
        _pt_pady = u.pad_y
        if self._low_height and not portrait:
            _pt_ipadx = max(8, u.pad_toggle_x - 4)
            _pt_ipady = max(8, u.pad_toggle_y - 4)
            _pt_pady = max(4, u.pad_y - 4)
        self.btn_play.pack(ipadx=_pt_ipadx, ipady=_pt_ipady, padx=u.pad_x, pady=_pt_pady)
        self.btn_toggle_play = self.btn_play
        self._btn(t_right, "+5s", lambda: self.seek_ms(SEEK_STEP_MS), w=6)
        if portrait:
            self._btn(t_bottom, "Continue", self.play_from_last_end)
            self.lbl_time = tk.Label(t_bottom, text="", fg=ACCENT, bg=BG, font=u.font_mono)
            self.lbl_time.pack(side=tk.LEFT, padx=(u.pad_x * 2, u.pad_x), pady=u.pad_y_tight)
        else:
            self._btn(t_right, "Continue", self.play_from_last_end)
            self.lbl_time = tk.Label(t_right, text="", fg=ACCENT, bg=BG, font=u.font_mono)
            self.lbl_time.pack(side=tk.LEFT, padx=(u.pad_x * 2, u.pad_x), pady=u.pad_y)

        # --- File label row ---
        try:
            self.lbl_file.pack_forget()
        except tk.TclError:
            pass
        if portrait:
            self.lbl_file.pack(anchor=tk.W, fill=tk.X, padx=u.pad_x, pady=(4, 0))
        else:
            self.lbl_file.pack(
                side=tk.LEFT, padx=u.pad_x, pady=u.pad_y_tight, in_=self.fr_top_load
            )

        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _btn(self, parent, text, cmd, w=None):
        u = self.ui
        kw = dict(
            text=text,
            command=cmd,
            bg=BTN_BG,
            fg=TEXT,
            activeforeground=TEXT,
            activebackground=BTN_ACTIVE,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        if w:
            kw["width"] = w
        tk.Button(parent, **kw).pack(
            side=tk.LEFT,
            padx=u.pad_x,
            pady=u.pad_y_tight,
            ipadx=u.btn_ipadx,
            ipady=u.btn_ipady,
        )

    def _refresh_export_dir_label(self):
        """Legacy no-op: output path is not shown on the chrome (Browse… only)."""
        return

    def _apply_suggested_export_name(self) -> None:
        if self._closing:
            return
        if self.audio_path and self.pydub_audio is not None and self.span is not None:
            self._apply_export_filename_from_span()
            return
        sug = suggest_next_export_filename(self.export_dir)
        try:
            self.ent_name.delete(0, tk.END)
            self.ent_name.insert(0, sug)
        except tk.TclError:
            pass

    def _set_export_name_hints(self, speaker: str, transcript: str) -> None:
        """Speaker + transcript for auto export name (from AI trim preview, etc.)."""
        if self._closing:
            return
        self._export_speaker_hint = (speaker or "").strip()
        self._export_transcript_hint = (transcript or "").strip()
        self._debounce_export_filename_update()

    def _debounce_export_filename_update(self) -> None:
        if self._closing:
            return
        if self._export_name_after is not None:
            try:
                self.root.after_cancel(self._export_name_after)
            except (tk.TclError, ValueError):
                pass
            self._export_name_after = None
        try:
            self._export_name_after = self.root.after(40, self._do_export_filename_update)
        except tk.TclError:
            pass

    def _do_export_filename_update(self) -> None:
        self._export_name_after = None
        if self._closing:
            return
        self._apply_export_filename_from_span()

    def _apply_export_filename_from_span(self) -> None:
        """Set Filename entry from hints + current trim start (no disk bump — save applies bump)."""
        if self._closing:
            return
        try:
            self.ent_name
        except AttributeError:
            return
        if not self.audio_path or self.pydub_audio is None or self.span is None:
            return
        try:
            lo, hi = self.get_span_times_sec()
            if hi - lo < self.ui.min_trim_sec * 0.25:
                return
        except (tk.TclError, AttributeError, TypeError, ValueError):
            return
        fn = build_export_filename_from_hints(
            self._export_speaker_hint,
            self._export_transcript_hint,
            lo,
        )
        try:
            self.ent_name.delete(0, tk.END)
            self.ent_name.insert(0, fn)
        except tk.TclError:
            pass

    def set_export_dir(self, path: str) -> bool:
        """Set the directory used for Save trim. Creates the folder if needed."""
        if not path or self._closing:
            return False
        path = os.path.abspath(os.path.normpath(path))
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            try:
                messagebox.showerror("Output folder", str(e), parent=self.root)
            except tk.TclError:
                pass
            return False
        self.export_dir = path
        self._apply_suggested_export_name()
        return True

    def choose_export_directory(self):
        """Pick output folder via dialog (Browse…)."""
        if self._closing:
            return
        init = self.export_dir
        if not os.path.isdir(init):
            init = os.path.expanduser("~")
        d = filedialog.askdirectory(
            initialdir=init,
            title="Output folder for trimmed WAVs",
            parent=self.root,
        )
        if d:
            self.set_export_dir(d)

    def _on_root_map(self, _event=None):
        """Raise above taskbar / RDP client quirks; brief topmost on remote so shell does not stay on top."""
        if self._closing:
            return
        try:
            self.root.lift()
        except tk.TclError:
            return
        if not self._remote_session:
            return
        try:
            self.root.attributes("-topmost", True)
        except tk.TclError:
            return
        if self._topmost_clear_after is not None:
            try:
                self.root.after_cancel(self._topmost_clear_after)
            except (tk.TclError, ValueError):
                pass
        self._topmost_clear_after = self.root.after(450, self._clear_topmost_rdp)

    def _clear_topmost_rdp(self):
        self._topmost_clear_after = None
        if self._closing:
            return
        try:
            self.root.attributes("-topmost", False)
        except tk.TclError:
            pass

    def _bind_global_keys(self):
        # bind_all: root bindings miss keys when focus is on the matplotlib Tk widget.
        self.root.bind_all("<KeyPress-space>", self._tk_space)
        self.root.bind_all("<Escape>", self._tk_escape)
        self.root.bind_all("<Left>", self._tk_left)
        self.root.bind_all("<Right>", self._tk_right)

    def _on_plot_click_focus(self, event):
        if self._closing:
            return
        try:
            self._plot_tk.focus_set()
        except tk.TclError:
            pass

    def _on_plot_wrap_configure(self, event):
        if self._closing or event.widget is not self.plot_wrap:
            return
        if self._plot_resize_after is not None:
            try:
                self.root.after_cancel(self._plot_resize_after)
            except (tk.TclError, ValueError):
                pass
        self._plot_resize_after = self.root.after(140, self._resize_figure_to_plot_wrap)

    def _resize_figure_to_plot_wrap(self):
        self._plot_resize_after = None
        if self._closing:
            return
        try:
            w = max(self.plot_wrap.winfo_width(), 160)
            h = max(self.plot_wrap.winfo_height(), 160)
        except tk.TclError:
            return
        dpi = float(self.fig.get_dpi())
        try:
            self.fig.set_size_inches(w / dpi, h / dpi, forward=True)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _transport_keys_suppressed(self) -> bool:
        """Let Entry / buttons / checkboxes keep Space and arrows for their own UI."""
        try:
            w = self.root.focus_get()
        except tk.TclError:
            return False
        if w is None:
            return False
        if isinstance(w, (tk.Entry, tk.Text)):
            return True
        if isinstance(w, tk.Checkbutton):
            return True
        if isinstance(w, tk.Button) and w is not self.btn_toggle_play:
            return True
        return False

    def _tk_space(self, event):
        if self._closing or self._transport_keys_suppressed():
            return
        self.toggle_selection_transport()
        return "break"

    def _tk_escape(self, event):
        if self._closing or self._transport_keys_suppressed():
            return
        self.stop_all()
        return "break"

    def _tk_left(self, event):
        if self._closing or self._transport_keys_suppressed():
            return
        self.seek_ms(-SEEK_STEP_MS)
        return "break"

    def _tk_right(self, event):
        if self._closing or self._transport_keys_suppressed():
            return
        self.seek_ms(SEEK_STEP_MS)
        return "break"

    def _on_play_button(self):
        if self._closing:
            return
        self.toggle_selection_transport()

    def _sync_play_button(self):
        if self._closing:
            return
        try:
            playing = self.is_playing and not self._mixer_paused
            paused = self._mixer_paused and self._play_kind in ("selection", "full")
            if playing:
                text = "Pause"
            elif paused:
                text = "Resume"
            else:
                text = "Play"
            self.btn_play.config(text=text)
        except tk.TclError:
            pass

    def _on_mpl_key(self, event):
        if self._closing:
            return
        k = event.key
        if k == " ":
            if self._transport_keys_suppressed():
                return
            self.toggle_selection_transport()
        elif k == "escape":
            self.stop_all()
        elif k == "left":
            self.seek_ms(-SEEK_STEP_MS)
        elif k == "right":
            self.seek_ms(SEEK_STEP_MS)

    def _disconnect_xlim_callback(self):
        if self._xlim_cid is not None:
            try:
                self.ax.callbacks.disconnect(self._xlim_cid)
            except Exception:
                pass
            self._xlim_cid = None

    def _on_xlim_changed_axes(self, ax):
        if self._closing or ax is not self.ax:
            return
        self._apply_visible_amplitude_scale(ax)

    def _apply_visible_amplitude_scale(self, ax=None):
        """Fit amplitude to the waveform slice in the current X view (easier trims when zoomed)."""
        ax = ax or self.ax
        if self._y is None or self._sr is None or len(self._y) == 0:
            return
        if self._ylim_adjust_depth > 0:
            return
        x0, x1 = ax.get_xlim()
        lo, hi = sorted((float(x0), float(x1)))
        i0 = int(np.clip(np.floor(lo * self._sr), 0, len(self._y)))
        i1 = int(np.clip(np.ceil(hi * self._sr), 0, len(self._y)))
        sl = self._y if i1 <= i0 + 1 else self._y[i0:i1]
        peak = float(np.max(np.abs(sl))) if sl.size else 0.0
        peak = max(peak, 1e-9)
        pad = 0.12
        top = peak * (1.0 + pad)
        self._ylim_adjust_depth += 1
        try:
            ax.set_ylim(-top, top)
        finally:
            self._ylim_adjust_depth -= 1

    def _focus_plot_if_ok(self):
        if self._closing:
            return
        try:
            self._plot_tk.focus_set()
        except tk.TclError:
            pass

    def _destroy_span(self):
        if self.span is not None:
            try:
                self.span.disconnect_events()
            except Exception:
                pass
            self.span = None

    def select_all_waveform(self):
        """Set the trim span to the entire loaded file (full waveform)."""
        if self._closing or self.span is None or self.duration_sec <= 0:
            return
        try:
            self.span.extents = (0.0, float(self.duration_sec))
            self.canvas.draw_idle()
        except Exception:
            return
        self._on_span_changed()
        self._sync_trim_time_entries_from_span()

    def apply_typed_trim_times(self):
        """Set the SpanSelector from Start/End entry fields (seconds)."""
        if self._closing or self.span is None or self.duration_sec <= 0 or not self.audio_path:
            messagebox.showwarning("Trim times", "Load a file first.", parent=self.root)
            return
        try:
            raw0 = self.ent_trim_start.get().strip().replace(",", ".")
            raw1 = self.ent_trim_end.get().strip().replace(",", ".")
            t0 = float(raw0)
            t1 = float(raw1)
        except ValueError:
            messagebox.showwarning(
                "Trim times",
                "Enter valid numbers for start and end (seconds).",
                parent=self.root,
            )
            return
        lo, hi = sorted((t0, t1))
        lo = self._clamp_sec(lo)
        hi = self._clamp_sec(hi)
        if hi - lo < self.ui.min_trim_sec:
            messagebox.showwarning(
                "Trim times",
                f"Range must be at least {self.ui.min_trim_sec:.3f} s.",
                parent=self.root,
            )
            return
        try:
            self.span.extents = (float(lo), float(hi))
            self.canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Trim times", str(e), parent=self.root)
            return
        self._on_span_changed()
        self._sync_trim_time_entries_from_span()

    def _sync_trim_time_entries_from_span(self):
        """Mirror current selection into the typed-time entries (or clear if no span)."""
        if self._closing:
            return
        try:
            e0, e1 = self.ent_trim_start, self.ent_trim_end
        except AttributeError:
            return
        try:
            if self.span is None or self.duration_sec <= 0:
                e0.delete(0, tk.END)
                e1.delete(0, tk.END)
                return
            lo, hi = self.get_span_times_sec()
            e0.delete(0, tk.END)
            e0.insert(0, f"{lo:.4f}")
            e1.delete(0, tk.END)
            e1.insert(0, f"{hi:.4f}")
        except tk.TclError:
            pass
        self._debounce_export_filename_update()

    def _playback_span_sec(self) -> tuple[float, float] | None:
        """Region to play in seconds; expands an empty/degenerate span to the full file."""
        if self.pydub_audio is None:
            return None
        dur_s = len(self.pydub_audio) / 1000.0
        if self.span is not None and self.duration_sec > 0:
            lo, hi = self.get_span_times_sec()
            if hi - lo < 0.001:
                try:
                    self.span.extents = (0.0, float(self.duration_sec))
                except Exception:
                    pass
                lo, hi = self.get_span_times_sec()
            if hi - lo < 0.001:
                lo, hi = 0.0, float(min(self.duration_sec, dur_s))
        else:
            lo, hi = 0.0, float(dur_s)
        if hi - lo < 1e-6:
            return None
        return lo, hi

    def get_span_times_sec(self) -> tuple[float, float]:
        if self.span is None or self.duration_sec <= 0:
            return 0.0, 0.0
        try:
            lo, hi = self.span.extents
        except (AttributeError, TypeError, ValueError):
            return 0.0, 0.0
        t0 = self._clamp_sec(min(float(lo), float(hi)))
        t1 = self._clamp_sec(max(float(lo), float(hi)))
        return t0, t1

    def _on_span_select(self, vmin, vmax):
        self._on_span_changed()
        self._sync_trim_time_entries_from_span()

    def _on_span_move(self, vmin, vmax):
        self._on_span_changed()

    def _on_span_changed(self):
        if self.var_loop.get() and self._play_kind == "selection":
            self._debounce_loop_restart()
        if self.var_auto_zoom.get():
            self._debounce_zoom_to_selection()
        self._debounce_export_filename_update()

    def _debounce_zoom_to_selection(self):
        if self._zoom_debounce_after is not None:
            try:
                self.root.after_cancel(self._zoom_debounce_after)
            except (tk.TclError, ValueError):
                pass
        self._zoom_debounce_after = self.root.after(110, self._apply_auto_zoom)

    def _apply_auto_zoom(self):
        self._zoom_debounce_after = None
        if self._closing or self.span is None or self.pydub_audio is None:
            return
        if not self.var_auto_zoom.get():
            return
        try:
            self.zoom_to_selection()
        except Exception:
            pass

    def mark_trim_start(self):
        """Move selection left edge to current playhead (subtract/include from here)."""
        if self.span is None or not self.audio_path or self.pydub_audio is None:
            return
        t = self._clamp_sec(self.get_position_ms() / 1000.0)
        lo, hi = self.get_span_times_sec()
        new_lo = t
        new_hi = hi
        if new_hi - new_lo < self.ui.min_trim_sec:
            new_hi = self._clamp_sec(new_lo + self.ui.min_trim_sec)
        if new_hi <= new_lo:
            new_hi = self._clamp_sec(new_lo + self.ui.min_trim_sec)
        try:
            self.span.extents = (min(new_lo, new_hi), max(new_lo, new_hi))
            self.canvas.draw_idle()
        except Exception:
            return
        self._after_trim_mark()

    def mark_trim_end(self):
        """Move selection right edge to current playhead."""
        if self.span is None or not self.audio_path or self.pydub_audio is None:
            return
        t = self._clamp_sec(self.get_position_ms() / 1000.0)
        lo, hi = self.get_span_times_sec()
        new_lo = lo
        new_hi = t
        if new_hi - new_lo < self.ui.min_trim_sec:
            new_lo = self._clamp_sec(new_hi - self.ui.min_trim_sec)
        if new_hi <= new_lo:
            new_lo = self._clamp_sec(new_hi - self.ui.min_trim_sec)
        try:
            self.span.extents = (min(new_lo, new_hi), max(new_lo, new_hi))
            self.canvas.draw_idle()
        except Exception:
            return
        self._after_trim_mark()

    def _after_trim_mark(self):
        if self.var_loop.get() and self._play_kind == "selection":
            self._debounce_loop_restart()
        if self.var_auto_zoom.get():
            self.zoom_to_selection()
        self._sync_trim_time_entries_from_span()

    def _clamp_sec(self, x: float) -> float:
        if self.duration_sec <= 0:
            return 0.0
        return float(max(0.0, min(x, self.duration_sec)))

    def _empty_plot(self):
        self._disconnect_xlim_callback()
        self._destroy_span()
        self.ax.clear()
        self.ax.set_facecolor(BG)
        u = self.ui
        self.ax.tick_params(colors=MUTED, labelsize=u.mpl_tick)
        self.ax.set_xlabel("Time (s)", color=MUTED, fontsize=u.mpl_axis)
        self.ax.set_ylabel("Amplitude", color=MUTED, fontsize=u.mpl_axis)
        self.ax.text(
            0.5,
            0.5,
            "Open a WAV file",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=u.mpl_axis,
        )
        self.canvas.draw_idle()
        self._sync_trim_time_entries_from_span()

    def _prepare_load_cleanup(self):
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self._cancel_timers()
        self._cleanup_playback_temp()
        had = self.pydub_audio is not None or self._y is not None
        self._destroy_span()
        self.pydub_audio = None
        self._y = None
        self._sr = None
        self.duration_sec = 0.0
        self._play_kind = None
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._last_play_end_ms = None
        self._last_saved_trim_end_ms = None
        self._sync_play_button()
        self._sync_trim_time_entries_from_span()
        if had:
            gc.collect()

    def load_file(self):
        path = filedialog.askopenfilename(initialdir=DEFAULT_IN, filetypes=[("WAV", "*.wav")])
        if path:
            self.load_audio_path(path)

    def _close_pslicer_loading_ui(self) -> None:
        if self._pslicer_load_poll_id is not None:
            try:
                self.root.after_cancel(self._pslicer_load_poll_id)
            except (tk.TclError, ValueError):
                pass
            self._pslicer_load_poll_id = None
        if self._pslicer_load_pb is not None:
            try:
                self._pslicer_load_pb.stop()
            except tk.TclError:
                pass
            self._pslicer_load_pb = None
        self._pslicer_load_lbl = None
        if self._pslicer_load_win is not None:
            try:
                self._pslicer_load_win.grab_release()
            except tk.TclError:
                pass
            try:
                self._pslicer_load_win.destroy()
            except tk.TclError:
                pass
            self._pslicer_load_win = None
        self._pslicer_phase_queue = None

    def _poll_pslicer_phase_queue(self) -> None:
        self._pslicer_load_poll_id = None
        if self._closing:
            return
        q = self._pslicer_phase_queue
        if q is None:
            return
        last: str | None = None
        try:
            while True:
                last = q.get_nowait()
        except queue.Empty:
            pass
        if last is not None and self._pslicer_load_lbl is not None:
            try:
                self._pslicer_load_lbl.config(text=last)
            except tk.TclError:
                pass
        if self._pslicer_busy and not self._closing and self._pslicer_phase_queue is not None:
            try:
                self._pslicer_load_poll_id = self.root.after(100, self._poll_pslicer_phase_queue)
            except tk.TclError:
                pass

    def _open_pslicer_loading_ui(self, audio_path: str) -> None:
        """Build loading window; caller must run ``_close_pslicer_loading_ui`` first if re-opening."""
        u = self.ui
        top = tk.Toplevel(self.root)
        self._pslicer_load_win = top
        top.title("AI trim — working")
        top.configure(bg=PANEL)
        top.transient(self.root)
        top.minsize(420, 160)
        try:
            top.grab_set()
        except tk.TclError:
            pass
        bn = os.path.basename(audio_path)
        dur = self.duration_sec if self.duration_sec > 0 else None
        tk.Label(
            top,
            text=f"Processing  {bn}",
            fg=TEXT,
            bg=PANEL,
            font=u.font_btn_bold,
            anchor=tk.CENTER,
        ).pack(padx=u.pad_x, pady=(16, 4))
        if dur is not None:
            tk.Label(
                top,
                text=f"Source length: {dur:.1f} s",
                fg=MUTED,
                bg=PANEL,
                font=u.font_label,
            ).pack(pady=(0, 6))
        self._pslicer_load_lbl = tk.Label(
            top,
            text="Starting…\nLong files can take several minutes (GPU helps).",
            fg=ACCENT,
            bg=PANEL,
            font=u.font_label,
            wraplength=440,
            justify=tk.CENTER,
        )
        self._pslicer_load_lbl.pack(padx=u.pad_x, pady=(4, 12))
        pb = ttk.Progressbar(top, mode="indeterminate", length=380)
        self._pslicer_load_pb = pb
        pb.pack(padx=u.pad_x, pady=(0, 18), fill=tk.X)
        try:
            pb.start(12)
        except tk.TclError:
            pass
        try:
            top.geometry(
                "+%d+%d"
                % (
                    max(0, self.root.winfo_rootx() + 40),
                    max(0, self.root.winfo_rooty() + 60),
                )
            )
        except tk.TclError:
            pass

    def _open_settings_dialog(self) -> None:
        """Persist Hugging Face token; updates ``HF_TOKEN`` immediately for AI trim."""
        if self._closing:
            return
        win = self._settings_win
        if win is not None:
            try:
                if win.winfo_exists():
                    try:
                        win.lift()
                        win.focus_force()
                    except tk.TclError:
                        pass
                    return
            except tk.TclError:
                pass
            self._settings_win = None

        u = self.ui
        top = tk.Toplevel(self.root)
        self._settings_win = top
        top.title("Voice Engine — Settings")
        top.configure(bg=PANEL)
        top.transient(self.root)
        top.minsize(440, 280)
        try:
            top.geometry(
                "+%d+%d"
                % (
                    max(0, self.root.winfo_rootx() + 48),
                    max(0, self.root.winfo_rooty() + 48),
                )
            )
        except tk.TclError:
            pass

        def _status() -> str:
            return "Token: configured (AI trim can use diarization)" if _resolve_hf_token_probe() else "Token: not configured"

        tk.Label(
            top,
            text="Hugging Face (AI trim / diarization)",
            fg=ACCENT,
            bg=PANEL,
            font=u.font_btn_bold,
        ).pack(anchor=tk.W, padx=u.pad_x, pady=(14, 4))
        lbl_path = tk.Label(
            top,
            text=f"Saved to:\n{voice_engine_settings_path()}",
            fg=MUTED,
            bg=PANEL,
            font=u.font_label,
            justify=tk.LEFT,
        )
        lbl_path.pack(anchor=tk.W, padx=u.pad_x, pady=(0, 8))
        lbl_status = tk.Label(top, text=_status(), fg=TEXT, bg=PANEL, font=u.font_label, justify=tk.LEFT)
        lbl_status.pack(anchor=tk.W, padx=u.pad_x, pady=(0, 6))
        tk.Label(
            top,
            text="Paste a read token (hf_…). It is stored only on this machine.",
            fg=MUTED,
            bg=PANEL,
            font=u.font_label,
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=u.pad_x, pady=(0, 6))
        ent = tk.Entry(top, width=48, show="•", bg=BTN_BG, fg=TEXT, insertbackground=ACCENT, font=u.font_label)
        ent.pack(fill=tk.X, padx=u.pad_x, pady=(0, 8))

        def _save() -> None:
            raw = ent.get().strip()
            try:
                save_voice_engine_hf_token(raw if raw else None)
            except OSError as e:
                messagebox.showerror("Settings", f"Could not save settings:\n{e}", parent=top)
                return
            lbl_status.config(text=_status())
            try:
                messagebox.showinfo(
                    "Settings",
                    "Saved. AI trim will use this token immediately.",
                    parent=top,
                )
            except tk.TclError:
                pass

        def _clear() -> None:
            if not messagebox.askyesno(
                "Settings",
                "Remove the stored Hugging Face token from this computer?\n"
                "(This session’s HF_TOKEN will be cleared; huggingface-cli cache is unchanged.)",
                parent=top,
            ):
                return
            ent.delete(0, tk.END)
            try:
                save_voice_engine_hf_token(None)
            except OSError as e:
                messagebox.showerror("Settings", f"Could not clear settings:\n{e}", parent=top)
                return
            lbl_status.config(text=_status())

        def _open_hf() -> None:
            webbrowser.open("https://huggingface.co/settings/tokens")

        def _open_terms() -> None:
            webbrowser.open("https://huggingface.co/pyannote/speaker-diarization-community-1")

        row = tk.Frame(top, bg=PANEL)
        row.pack(fill=tk.X, padx=u.pad_x, pady=(4, 12))
        tk.Button(
            row,
            text="Save",
            command=_save,
            bg=ACCENT,
            fg=ACCENT_ON_PRIMARY,
            font=u.font_btn,
            activeforeground=ACCENT_ON_PRIMARY,
            activebackground=ACCENT_HOVER,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            row,
            text="Clear stored token",
            command=_clear,
            bg=BTN_BG,
            fg=TEXT,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            row,
            text="HF tokens…",
            command=_open_hf,
            bg=BTN_BG,
            fg=TEXT,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            row,
            text="Pyannote model",
            command=_open_terms,
            bg=BTN_BG,
            fg=TEXT,
            font=u.font_btn,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        ).pack(side=tk.LEFT)

        def _on_top_close() -> None:
            self._settings_win = None
            try:
                top.destroy()
            except tk.TclError:
                pass

        top.protocol("WM_DELETE_WINDOW", _on_top_close)

    def _on_pslicer_done_wrap(
        self,
        err: str | None,
        chunks: list[tuple[float, float, str, str]] | None,
        aligned_words: list[dict] | None,
    ) -> None:
        self._pslicer_done_after_id = None
        self._on_pslicer_compute_done(err, chunks, aligned_words)

    def _schedule_pslicer_compute_done(
        self,
        err: str | None,
        chunks: list[tuple[float, float, str, str]] | None,
        aligned_words: list[dict] | None,
    ) -> None:
        try:
            if self._pslicer_done_after_id is not None:
                try:
                    self.root.after_cancel(self._pslicer_done_after_id)
                except (tk.TclError, ValueError):
                    pass
                self._pslicer_done_after_id = None
        except tk.TclError:
            return
        try:
            self._pslicer_done_after_id = self.root.after(
                0,
                lambda e=err, c=chunks, w=aligned_words: self._on_pslicer_done_wrap(e, c, w),
            )
        except tk.TclError:
            self._pslicer_done_after_id = None

    def start_pslicer_auto_trim(self) -> None:
        """Run WhisperX/pslicer chunking off the UI thread; open preview when done."""
        if self._closing or self._pslicer_busy:
            return
        if not self.audio_path or not os.path.isfile(self.audio_path):
            messagebox.showwarning("AI trim", "Open a WAV file first.", parent=self.root)
            return
        if importlib.util.find_spec("torch") is None:
            messagebox.showerror(
                "AI trim",
                _ai_trim_missing_stack_message(
                    detail="PyTorch is not installed (import name: torch).",
                ),
                parent=self.root,
            )
            return
        hf_probe = _resolve_hf_token_probe()
        no_diarize = False
        if not hf_probe:
            if not messagebox.askyesno(
                "AI trim — Hugging Face",
                "No Hugging Face token was found.\n\n"
                "For speaker diarization, create a token at huggingface.co/settings/tokens, "
                "set environment variable HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), run "
                "huggingface-cli login, and accept the pyannote model terms on the Hub, e.g.\n"
                "  https://huggingface.co/pyannote/speaker-diarization-community-1\n\n"
                "Continue without diarization?\n"
                "• Sentence-based cuts still run (Silero VAD).\n"
                "• Clips use a single speaker label (no separation).\n\n"
                "Choose No to cancel and configure a token first.",
                parent=self.root,
            ):
                return
            no_diarize = True
        self._pslicer_busy = True
        self._close_pslicer_loading_ui()
        self._pslicer_phase_queue = queue.Queue()
        self._open_pslicer_loading_ui(self.audio_path)
        try:
            self._pslicer_load_poll_id = self.root.after(80, self._poll_pslicer_phase_queue)
        except tk.TclError:
            self._pslicer_load_poll_id = None
        try:
            self.btn_pslicer.config(state=tk.DISABLED)
        except tk.TclError:
            pass
        ap = self.audio_path
        root = self.root
        run_no_diarize = no_diarize

        def work() -> None:
            err: str | None = None
            chunks: list[tuple[float, float, str, str]] | None = None
            aligned_words: list[dict] | None = None
            try:
                import pslicer as _ps
            except ImportError as imp_err:
                msg = _ai_trim_import_error_message(imp_err)
                self._schedule_pslicer_compute_done(msg, None, None)
                return

            def phase_cb(msg: str) -> None:
                q = self._pslicer_phase_queue
                if q is not None:
                    try:
                        q.put_nowait(msg)
                    except Exception:
                        pass

            try:
                if sys.platform == "win32":
                    _ps.register_windows_torchcodec_dll_paths()
                _ps._preconfigure_cuda_for_pyannote()
                hf_tok, _ = _ps.resolve_hf_token()
                if run_no_diarize:
                    chunks, aligned_words = _ps.compute_auto_trim_chunks(
                        ap,
                        hf_token=None,
                        diarize=False,
                        verbose=False,
                        phase_callback=phase_cb,
                        return_aligned_words=True,
                    )
                elif not hf_tok:
                    err = (
                        "No Hugging Face token. Set HF_TOKEN or run: huggingface-cli login\n"
                        "(Accept pyannote speaker-diarization terms on the Hub.)"
                    )
                else:
                    chunks, aligned_words = _ps.compute_auto_trim_chunks(
                        ap,
                        hf_token=hf_tok,
                        diarize=True,
                        verbose=False,
                        phase_callback=phase_cb,
                        return_aligned_words=True,
                    )
            except _ps.PslicerUserError as e:
                err = str(e)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            self._schedule_pslicer_compute_done(err, chunks, aligned_words)

        threading.Thread(target=work, daemon=True).start()

    def _on_pslicer_compute_done(
        self,
        err: str | None,
        chunks: list[tuple[float, float, str, str]] | None,
        aligned_words: list[dict] | None = None,
    ) -> None:
        try:
            self._close_pslicer_loading_ui()
        except (tk.TclError, RuntimeError):
            pass
        self._pslicer_busy = False
        try:
            self.btn_pslicer.config(state=tk.NORMAL)
        except tk.TclError:
            pass
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        if self._closing:
            return
        if err:
            try:
                messagebox.showerror("AI trim", err, parent=self.root)
            except tk.TclError:
                pass
            return
        if not chunks:
            try:
                messagebox.showinfo("AI trim", "No sentence chunks produced.", parent=self.root)
            except tk.TclError:
                pass
            return
        try:
            PslicerTrimPreviewDialog(self, self.audio_path, chunks, padding_ms=120, aligned_words=aligned_words)
        except tk.TclError:
            pass

    def load_audio_path(self, path: str) -> None:
        """Load a WAV from disk (also used by stress tests)."""
        if not path or self._closing:
            return
        self._prepare_load_cleanup()
        self.audio_path = path
        self._export_speaker_hint = ""
        self._export_transcript_hint = ""

        try:
            raw = AudioSegment.from_wav(path)
        except Exception as e:
            self.audio_path = ""
            messagebox.showerror("Load failed", f"Could not read WAV:\n{e}", parent=self.root)
            self._empty_plot()
            self._safe_mixer_init()
            return

        load_path = path
        if raw.sample_width == 3:
            self.pydub_audio = raw.set_sample_width(2)
            fd, self._playback_temp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            self.pydub_audio.export(self._playback_temp, format="wav")
            load_path = self._playback_temp
        else:
            self.pydub_audio = raw

        try:
            self._reinit_mixer_for_segment(self.pydub_audio)
            pygame.mixer.music.load(load_path)
        except pygame.error as e:
            self.audio_path = ""
            self.pydub_audio = None
            messagebox.showerror("Audio engine", f"Pygame could not load audio:\n{e}", parent=self.root)
            self._cleanup_playback_temp()
            self._safe_mixer_init()
            self._empty_plot()
            return

        try:
            self._y, self._sr = librosa.load(path, sr=None, mono=True)
        except Exception as e:
            self.audio_path = ""
            self.pydub_audio = None
            messagebox.showerror("Analysis failed", f"librosa could not load file:\n{e}", parent=self.root)
            self._cleanup_playback_temp()
            self._safe_mixer_init()
            self._empty_plot()
            return

        self.duration_sec = min(float(len(self._y)) / float(self._sr), len(self.pydub_audio) / 1000.0)
        try:
            self.lbl_file.config(text=os.path.basename(path))
        except tk.TclError:
            return
        self._last_clock_text = None
        self._redraw_waveform()
        self.reset_view()
        self._sync_trim_time_entries_from_span()
        try:
            self.root.after_idle(self._focus_plot_if_ok)
        except tk.TclError:
            pass
        if self._play_kind == "selection" and self.var_loop.get():
            self._on_loop_toggle()

    def _safe_mixer_init(self):
        try:
            pygame.mixer.init()
        except pygame.error:
            pass

    def _mixer_size_from_pydub(self, sample_width: int) -> int:
        if sample_width <= 1:
            return 8
        if sample_width == 2:
            return -16
        if sample_width == 4:
            return -32
        return -16

    def _reinit_mixer_for_segment(self, segment: AudioSegment) -> None:
        try:
            pygame.mixer.quit()
        except pygame.error:
            pass
        try:
            pygame.mixer.init(
                frequency=int(segment.frame_rate),
                channels=int(segment.channels),
                size=self._mixer_size_from_pydub(segment.sample_width),
                allowedchanges=0,
            )
            self._mixer_initialized = True
        except pygame.error:
            self._mixer_initialized = False

    def _cleanup_playback_temp(self):
        t = self._playback_temp
        self._playback_temp = None
        if t and os.path.isfile(t):
            try:
                os.remove(t)
            except OSError:
                pass

    def _redraw_waveform(self):
        self._disconnect_xlim_callback()
        self._destroy_span()
        self.ax.clear()
        self.ax.set_facecolor(BG)
        u = self.ui
        self.ax.tick_params(colors=MUTED, labelsize=u.mpl_tick)
        self.ax.set_xlabel("Time (s)", color=MUTED, fontsize=u.mpl_axis)
        self.ax.set_ylabel("Amplitude", color=MUTED, fontsize=u.mpl_axis)
        self.ax.set_title(
            "Drag selection · markers follow playhead",
            color=MUTED,
            fontsize=u.mpl_title,
        )

        if self._y is None or self._sr is None:
            self.canvas.draw_idle()
            return

        librosa.display.waveshow(
            self._y,
            sr=self._sr,
            ax=self.ax,
            color=WAVEFORM_COLOR,
            alpha=0.88,
            lw=u.waveform_lw,
        )
        self._apply_visible_amplitude_scale(self.ax)
        self._xlim_cid = self.ax.callbacks.connect("xlim_changed", self._on_xlim_changed_axes)

        self.span = SpanSelector(
            self.ax,
            self._on_span_select,
            "horizontal",
            minspan=u.span_min_sec,
            useblit=False,
            props=dict(facecolor=SPAN_FACE, alpha=0.5, edgecolor=SPAN_EDGE, linewidth=u.span_lw),
            interactive=True,
            drag_from_anywhere=True,
            grab_range=u.span_grab,
            handle_props=dict(color=ACCENT, linewidth=u.span_handle_lw, alpha=1.0),
            onmove_callback=self._on_span_move,
        )
        if self.duration_sec > 0:
            self.span.extents = (0.0, float(self.duration_sec))

        self.canvas.draw_idle()

    def get_position_ms(self) -> int:
        if not self.audio_path or self.pydub_audio is None:
            return 0
        dur = len(self.pydub_audio)
        try:
            gp = pygame.mixer.music.get_pos()
        except pygame.error:
            return max(0, min(self._play_origin_ms, dur))
        if gp < 0:
            return max(0, min(self._play_origin_ms, dur))
        return max(0, min(self._play_origin_ms + gp, dur))

    def play_from_ms(self, ms: int, resume_if_playing: bool = True) -> bool:
        if not self.audio_path or self.pydub_audio is None:
            return False
        dur = len(self.pydub_audio)
        ms = int(max(0, min(ms, dur)))
        if dur > 0:
            ms = min(ms, dur - 1)
        start_sec = ms / 1000.0
        try:
            if ms <= 0:
                pygame.mixer.music.play()
            else:
                pygame.mixer.music.play(start=start_sec)
        except pygame.error:
            return False
        self._play_origin_ms = ms
        self._mixer_paused = False
        if not resume_if_playing:
            try:
                pygame.mixer.music.pause()
            except pygame.error:
                pass
            self._mixer_paused = True
        return True

    def stop_all(self):
        self._cancel_timers()
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._play_kind = None
        self._sync_play_button()

    def seek_ms(self, delta_ms: int):
        if not self.audio_path or self.pydub_audio is None or self._closing:
            return
        self._cancel_selection_stop()
        try:
            busy = pygame.mixer.music.get_busy()
        except pygame.error:
            busy = False
        want_audio = (not self._mixer_paused) and (self.is_playing or busy)
        pos = int(max(0, min(self.get_position_ms() + delta_ms, len(self.pydub_audio))))
        self.play_from_ms(pos, resume_if_playing=want_audio)
        self.is_playing = want_audio
        if self._play_kind == "selection" and self.var_loop.get():
            self._debounce_loop_restart()
        elif self._play_kind == "full" and want_audio and self.pydub_audio is not None:
            self._cancel_selection_stop()
            rem = max(1, len(self.pydub_audio) - pos)
            self._selection_stop_after = self.root.after(rem, self._on_full_play_end)

    def toggle_selection_transport(self):
        """Space: pause/resume, or start playback (full file if no usable span)."""
        if not self.audio_path or self.pydub_audio is None or self._closing:
            return
        if self._playback_span_sec() is None:
            return

        if self._play_kind in ("selection", "full") and (self.is_playing or self._mixer_paused):
            try:
                if self.is_playing:
                    pygame.mixer.music.pause()
                    self.is_playing = False
                    self._mixer_paused = True
                else:
                    pygame.mixer.music.unpause()
                    self._mixer_paused = False
                    self.is_playing = True
            except pygame.error:
                self.stop_all()
            self._sync_play_button()
            return

        if self.span is None:
            self._start_full_playback(0)
        else:
            self._start_selection_playback()
        self._sync_play_button()

    def play_from_last_end(self):
        """Select audio after the last saved trim (or last play end) and audition it."""
        if not self.audio_path or self.pydub_audio is None or self._closing:
            return
        if self.span is None or self.duration_sec <= 0:
            return
        dur = len(self.pydub_audio)
        if dur < 1:
            return

        if self._last_saved_trim_end_ms is not None:
            start_ms = int(max(0, min(self._last_saved_trim_end_ms, dur - 1)))
        elif self._last_play_end_ms is not None:
            start_ms = int(max(0, min(self._last_play_end_ms, dur - 1)))
        else:
            start_ms = 0

        lo = self._clamp_sec(start_ms / 1000.0)
        hi = self._clamp_sec(self.duration_sec)
        if hi - lo < self.ui.min_trim_sec:
            try:
                messagebox.showinfo(
                    "Continue",
                    "No remaining audio after that point (or file already fully exported).",
                    parent=self.root,
                )
            except tk.TclError:
                pass
            return

        try:
            self.span.extents = (float(lo), float(hi))
            self.canvas.draw_idle()
        except Exception:
            return
        self._on_span_changed()
        self._sync_trim_time_entries_from_span()
        self._start_selection_playback()

    def _apply_saved_trim_marker(self, trim_end_ms: int):
        """After a successful export: remainder for 'From last end' starts at this sample (ms)."""
        if self.pydub_audio is None:
            return
        d = len(self.pydub_audio)
        self._last_saved_trim_end_ms = int(max(0, min(trim_end_ms, d)))

    def _start_full_playback(self, start_ms: int):
        """Play from start_ms to end of file (no selection loop; ignores Loop checkbox)."""
        if not self.audio_path or self.pydub_audio is None or self._closing:
            return
        dur = len(self.pydub_audio)
        if dur < 1:
            return
        self._cancel_timers()
        start_ms = int(max(0, min(start_ms, dur - 1)))
        rem_ms = max(1, dur - start_ms)
        self._play_kind = "full"
        if not self.play_from_ms(start_ms, resume_if_playing=True):
            self._play_kind = None
            self._sync_play_button()
            return
        self.is_playing = True
        self._mixer_paused = False
        self._selection_stop_after = self.root.after(rem_ms, self._on_full_play_end)
        self._sync_play_button()

    def _on_full_play_end(self):
        self._selection_stop_after = None
        if self.pydub_audio is not None:
            self._last_play_end_ms = len(self.pydub_audio)
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._play_kind = None
        self._sync_play_button()

    def _start_selection_playback(self):
        self._cancel_timers()
        lo, hi = self.get_span_times_sec()
        dur_ms = int(max(0, (hi - lo) * 1000))
        if dur_ms < 1:
            self._sync_play_button()
            return
        self._play_kind = "selection"
        start_ms = int(lo * 1000)
        if not self.play_from_ms(start_ms, resume_if_playing=True):
            self._play_kind = None
            self.is_playing = False
            self._sync_play_button()
            return
        self.is_playing = True
        self._mixer_paused = False

        if self.var_loop.get():
            self._run_selection_loop()
        else:
            self._selection_stop_after = self.root.after(dur_ms, self._on_selection_once_end)
        self._sync_play_button()

    def _on_selection_once_end(self):
        self._selection_stop_after = None
        if self.pydub_audio is not None:
            lo, hi = self.get_span_times_sec()
            d = len(self.pydub_audio)
            self._last_play_end_ms = max(0, min(int(round(hi * 1000)), d))
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._play_kind = None
        self._sync_play_button()

    def _run_selection_loop(self):
        self._stop_loop_tick_only()

        def tick():
            self._preview_after = None
            if self._closing or not self.var_loop.get() or self._play_kind != "selection":
                return
            lo, hi = self.get_span_times_sec()
            end_ms = int(hi * 1000)
            start_ms = int(lo * 1000)
            if self.get_position_ms() >= end_ms - 12:
                self.play_from_ms(start_ms, resume_if_playing=True)
            if not self._closing:
                self._preview_after = self.root.after(40, tick)

        self._preview_after = self.root.after(40, tick)

    def _stop_loop_tick_only(self):
        if self._preview_after is not None:
            try:
                self.root.after_cancel(self._preview_after)
            except (tk.TclError, ValueError):
                pass
            self._preview_after = None

    def _debounce_loop_restart(self):
        if self._preview_bump_after is not None:
            try:
                self.root.after_cancel(self._preview_bump_after)
            except (tk.TclError, ValueError):
                pass
        self._preview_bump_after = self.root.after(100, self._bump_loop)

    def _bump_loop(self):
        self._preview_bump_after = None
        if self._closing or self._play_kind != "selection" or not self.var_loop.get():
            return
        self._run_selection_loop()

    def _on_loop_toggle(self):
        if not self.audio_path or self._play_kind != "selection":
            return
        if self.span is None:
            return
        if self.var_loop.get():
            self._cancel_selection_stop()
            self._run_selection_loop()
        else:
            self._stop_loop_tick_only()
            if self.is_playing and not self._mixer_paused:
                _, hi = self.get_span_times_sec()
                rem = max(1, int(hi * 1000) - self.get_position_ms())
                self._selection_stop_after = self.root.after(rem, self._on_selection_once_end)

    def zoom_to_selection(self):
        if self._closing or self.span is None or self.pydub_audio is None:
            return
        lo, hi = self.get_span_times_sec()
        span = max(hi - lo, 1e-6)
        pad = max(span * 0.06, 0.002)
        try:
            self.ax.set_xlim(lo - pad, hi + pad)
            self.canvas.draw_idle()
        except Exception:
            pass

    def reset_view(self):
        if self._closing or self.duration_sec <= 0:
            return
        try:
            self.ax.set_xlim(0.0, self.duration_sec)
            self.canvas.draw_idle()
        except Exception:
            pass

    def save_trim(self):
        if self._export_busy:
            messagebox.showinfo("Export", "An export is already in progress.", parent=self.root)
            return
        if self.pydub_audio is None or self.span is None:
            messagebox.showwarning("Export", "Load a file and select a region.", parent=self.root)
            return
        t0, t1 = self.get_span_times_sec()
        a = int(round(t0 * 1000))
        b = int(round(t1 * 1000))
        if b - a < 1:
            messagebox.showwarning("Export", "Selection is too small.", parent=self.root)
            return
        name = self.ent_name.get().strip()
        if not name:
            messagebox.showwarning("Export", "Enter a filename.", parent=self.root)
            return
        if not name.lower().endswith(".wav"):
            name += ".wav"
        try:
            os.makedirs(self.export_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Export", f"Cannot use output folder:\n{e}", parent=self.root)
            return
        name = bump_export_filename_if_exists(self.export_dir, name)
        try:
            self.ent_name.delete(0, tk.END)
            self.ent_name.insert(0, name)
        except tk.TclError:
            pass
        out_path = os.path.join(self.export_dir, name)

        try:
            to_export = self.pydub_audio[a:b]
        except Exception as e:
            messagebox.showerror("Export", f"Could not build trim:\n{e}", parent=self.root)
            return

        self._export_busy = True
        self.btn_save.config(state=tk.DISABLED, text="Saving…")
        rolling_path = os.path.join(self.export_dir, ROLLING_BACKUP_BASENAME)

        def worker():
            error = None
            try:
                if os.path.isfile(out_path):
                    shutil.copy2(out_path, out_path + ".bak")
                to_export.export(out_path, format="wav")
                try:
                    shutil.copy2(out_path, rolling_path)
                except OSError:
                    pass
            except Exception as e:
                error = e

            def ui():
                if self._closing:
                    return
                try:
                    if not self.root.winfo_exists():
                        return
                except tk.TclError:
                    return
                self._export_finished(out_path, error, trim_end_ms=b)

            try:
                self.root.after(0, ui)
            except tk.TclError:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _export_finished(self, out_path: str, err: Exception | None, trim_end_ms: int | None = None):
        if self._closing:
            return
        self._export_busy = False
        try:
            self.btn_save.config(state=tk.NORMAL, text="Export")
        except tk.TclError:
            return
        if err is not None:
            try:
                messagebox.showerror("Export failed", str(err), parent=self.root)
            except tk.TclError:
                pass
            return
        if trim_end_ms is not None:
            self._apply_saved_trim_marker(trim_end_ms)
        try:
            messagebox.showinfo("Export complete", f"Saved:\n{out_path}", parent=self.root)
        except tk.TclError:
            pass
        self._apply_export_filename_from_span()

    def _cancel_clock(self):
        if self._clock_after is not None:
            try:
                self.root.after_cancel(self._clock_after)
            except (tk.TclError, ValueError):
                pass
            self._clock_after = None

    def _cancel_selection_stop(self):
        aid = self._selection_stop_after
        self._selection_stop_after = None
        if aid is not None:
            try:
                self.root.after_cancel(aid)
            except (tk.TclError, ValueError):
                pass

    def _cancel_timers(self):
        if self._zoom_debounce_after is not None:
            try:
                self.root.after_cancel(self._zoom_debounce_after)
            except (tk.TclError, ValueError):
                pass
            self._zoom_debounce_after = None
        self._stop_loop_tick_only()
        if self._preview_bump_after is not None:
            try:
                self.root.after_cancel(self._preview_bump_after)
            except (tk.TclError, ValueError):
                pass
            self._preview_bump_after = None
        self._cancel_selection_stop()
        if self._export_name_after is not None:
            try:
                self.root.after_cancel(self._export_name_after)
            except (tk.TclError, ValueError):
                pass
            self._export_name_after = None

    def _schedule_clock(self):
        if self._closing:
            return
        try:
            if not self.root.winfo_exists():
                return
        except tk.TclError:
            return
        try:
            if self.pydub_audio is not None:
                c = self.get_position_ms() / 1000.0
                t = len(self.pydub_audio) / 1000.0
                lo, hi = self.get_span_times_sec()
                text = f"{c:7.2f}s / {t:7.2f}s   [{lo:6.2f} – {hi:6.2f}]"
                if text != self._last_clock_text:
                    self.lbl_time.config(text=text)
                    self._last_clock_text = text
            else:
                if self._last_clock_text != "":
                    self.lbl_time.config(text="")
                    self._last_clock_text = ""
        except (tk.TclError, pygame.error):
            return
        if not self._closing:
            self._clock_after = self.root.after(120, self._schedule_clock)

    def _on_close(self):
        self._closing = True
        if self._pslicer_done_after_id is not None:
            try:
                self.root.after_cancel(self._pslicer_done_after_id)
            except (tk.TclError, ValueError):
                pass
            self._pslicer_done_after_id = None
        if self._settings_win is not None:
            try:
                self._settings_win.destroy()
            except tk.TclError:
                pass
            self._settings_win = None
        self._close_pslicer_loading_ui()
        self._export_busy = False
        if self._topmost_clear_after is not None:
            try:
                self.root.after_cancel(self._topmost_clear_after)
            except (tk.TclError, ValueError):
                pass
            self._topmost_clear_after = None
        try:
            self.root.attributes("-topmost", False)
        except tk.TclError:
            pass
        if self._viewport_sync_after is not None:
            try:
                self.root.after_cancel(self._viewport_sync_after)
            except (tk.TclError, ValueError):
                pass
            self._viewport_sync_after = None
        self._disconnect_xlim_callback()
        if self._plot_resize_after is not None:
            try:
                self.root.after_cancel(self._plot_resize_after)
            except (tk.TclError, ValueError):
                pass
            self._plot_resize_after = None
        self._cancel_clock()
        self._cancel_timers()
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self._cleanup_playback_temp()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        self._destroy_span()
        self.pydub_audio = None
        self._y = None
        self._sr = None
        gc.collect()
        try:
            plt.close(self.fig)
        except Exception:
            pass
        self.root.destroy()


def _run_stress_harness() -> int:
    """Exercise UI/audio/export under both desktop and touch profiles (SANCTUM_UI)."""
    from pydub.generators import Sine

    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    seg = Sine(440).to_audio_segment(duration=600) + AudioSegment.silent(duration=600)
    seg = seg.apply_gain(-12)
    seg.export(wav, format="wav")

    prev_ui = os.environ.get("SANCTUM_UI")
    stress_out_dirs: list[str] = []
    n = 50
    try:
        for profile in ("desktop", "touch"):
            os.environ["SANCTUM_UI"] = profile
            root = tk.Tk()
            app = SanctumSurgicalV3(root)
            if app.ui_profile != profile:
                print("stress_fail profile", profile, "got", app.ui_profile, flush=True)
                return 1
            if app.btn_toggle_play is not app.btn_play:
                print("stress_fail toggle play alias", flush=True)
                return 1

            def pump():
                root.update_idletasks()
                root.update()

            dlg_target = tempfile.mkdtemp(prefix=f"slicer_stress_dlg_{profile}_")
            stress_out_dirs.append(dlg_target)
            real_ask = filedialog.askdirectory
            filedialog.askdirectory = lambda **kwargs: dlg_target
            try:
                app.choose_export_directory()
            finally:
                filedialog.askdirectory = real_ask
            if os.path.normcase(os.path.abspath(app.export_dir)) != os.path.normcase(
                os.path.abspath(dlg_target)
            ):
                print("stress_fail choose_export_directory", profile, flush=True)
                return 1

            for i in range(n):
                out_dir = tempfile.mkdtemp(prefix=f"slicer_out_{profile}_{i}_")
                stress_out_dirs.append(out_dir)
                if not app.set_export_dir(out_dir):
                    print("stress_fail set_export_dir", profile, i, flush=True)
                    return 1
                probe = os.path.join(app.export_dir, "probe.wav")
                if os.path.normcase(os.path.abspath(os.path.dirname(probe))) != os.path.normcase(
                    os.path.abspath(app.export_dir)
                ):
                    print("stress_fail export path join", profile, flush=True)
                    return 1

                app.load_audio_path(wav)
                pump()

                app.ent_trim_end.delete(0, tk.END)
                app.ent_trim_end.insert(0, "0.35")
                app.ent_trim_start.delete(0, tk.END)
                app.ent_trim_start.insert(0, "0.15")
                app.apply_typed_trim_times()
                pump()
                lo_m, hi_m = app.get_span_times_sec()
                if not (0.14 <= lo_m <= 0.16 and 0.34 <= hi_m <= 0.36):
                    print("stress_fail typed trim", profile, lo_m, hi_m, flush=True)
                    return 1

                if app.span is not None and app.duration_sec > 0:
                    try:
                        app.span.extents = (0.0, 0.0)
                    except Exception:
                        pass
                pump()
                app._focus_plot_if_ok()
                pump()
                app.toggle_selection_transport()
                pump()
                app._on_play_button()
                pump()
                app.toggle_selection_transport()
                pump()

                app.select_all_waveform()
                pump()
                app._on_play_button()
                pump()
                app.seek_ms(200)
                pump()

                app._apply_saved_trim_marker(400)
                app.stop_all()
                pump()
                app.play_from_last_end()
                pump()
                lo, hi = app.get_span_times_sec()
                dur_s = len(app.pydub_audio) / 1000.0
                if not (0.35 <= lo <= 0.45 and hi >= dur_s - 0.02):
                    print("stress_fail remainder span", profile, lo, hi, dur_s, flush=True)
                    return 1
                app.stop_all()
                pump()

                app._apply_saved_trim_marker(200)
                app._last_play_end_ms = 800
                app.play_from_last_end()
                pump()
                lo_p, _ = app.get_span_times_sec()
                if not (0.15 <= lo_p <= 0.25):
                    print("stress_fail saved vs play priority", profile, lo_p, flush=True)
                    return 1
                app.stop_all()
                pump()

                app._start_full_playback(100)
                pump()
                app.seek_ms(50)
                pump()
                app.stop_all()
                pump()

                app.zoom_to_selection()
                pump()
                app.ax.set_xlim(0.05, 0.25)
                app._apply_visible_amplitude_scale(app.ax)
                pump()
                app.reset_view()
                pump()

                app._resize_figure_to_plot_wrap()
                pump()
                if app.plot_wrap.winfo_width() < 50 or app.plot_wrap.winfo_height() < 50:
                    print(
                        "stress_fail plot_wrap size",
                        profile,
                        app.plot_wrap.winfo_width(),
                        app.plot_wrap.winfo_height(),
                        flush=True,
                    )
                    return 1

                if i % 10 == 0:
                    print(f"stress {profile} iter {i + 1}/{n}", flush=True)

            app._on_close()
    finally:
        if prev_ui is None:
            os.environ.pop("SANCTUM_UI", None)
        else:
            os.environ["SANCTUM_UI"] = prev_ui
        try:
            os.remove(wav)
        except OSError:
            pass
        for d in stress_out_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    print("slicer_stress_ok", n, "x2 profiles")
    return 0


def main():
    root = tk.Tk()
    try:
        root.update_idletasks()
    except tk.TclError:
        pass
    app = SanctumSurgicalV3(root)
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        wav_path = os.path.abspath(arg)
        if wav_path.lower().endswith(".wav") and os.path.isfile(wav_path):
            root.after(80, lambda p=wav_path: app.load_audio_path(p))
            break
    root.mainloop()


if __name__ == "__main__":
    if "--stress" in sys.argv:
        sys.exit(_run_stress_harness())
    main()
