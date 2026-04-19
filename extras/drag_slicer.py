"""
Audacity-style region slicer: SpanSelector + pygame.

Run (from repo root): ``python extras/drag_slicer.py``
"""

import gc
import os
import re
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from pydub import AudioSegment

DEFAULT_IN = r"C:\AI\SanctumCore\voice_assets\raw_source\clean"
DEFAULT_OUT = r"C:\AI\SanctumCore\voice_assets\Pure_Jarvis_Audio"

os.makedirs(DEFAULT_OUT, exist_ok=True)

BG = "#1a1a1a"
MAX_WAVE_POINTS = 280_000


class DragWaveSlicer:
    def __init__(self, root):
        self.root = root
        self.root.title("Drag Slicer — Region selection")
        self.root.geometry("1020x720")
        self.root.configure(bg=BG)

        self.pydub_audio = None
        self.audio_path = ""
        self.duration_sec = 0.0
        self._t_plot = self._y_plot = None

        self.span = None
        # 'full' | 'selection' | None — what Space / pause logic applies to
        self._play_kind = None

        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._playback_temp = None
        self._selection_stop_after = None
        self._preview_after = None
        self._preview_bump_after = None
        self._clock_after = None
        self._closing = False

        pygame.mixer.init()

        # --- Top ---
        top = tk.Frame(root, bg=BG)
        top.pack(side=tk.TOP, fill=tk.X, pady=6)
        tk.Button(top, text="Load", command=self.load_file, bg="#333", fg="white").pack(side=tk.LEFT, padx=8)
        self.lbl_file = tk.Label(top, text="No file loaded", fg="cyan", bg=BG)
        self.lbl_file.pack(side=tk.LEFT, padx=8)

        # --- Controls ---
        ctrl = tk.Frame(root, bg=BG)
        ctrl.pack(fill=tk.X, pady=4)
        tk.Button(ctrl, text="Zoom to Selection", command=self.zoom_to_selection, bg="#444", fg="white").pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(ctrl, text="Show Full Waveform", command=self.reset_view, bg="#444", fg="white").pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(ctrl, text="Play / Pause (full)", command=self.toggle_play, bg="gold").pack(side=tk.LEFT, padx=12)
        tk.Button(ctrl, text="<< -2s", command=lambda: self.seek_ms(-2000), width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="+2s >>", command=lambda: self.seek_ms(2000), width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Play Selection", command=self.play_selection, bg="#2a5a2a", fg="white").pack(
            side=tk.LEFT, padx=6
        )
        self.var_loop_selection = tk.IntVar(value=1)
        tk.Checkbutton(
            ctrl,
            text="Loop selection",
            variable=self.var_loop_selection,
            command=self._on_loop_selection_toggle,
            fg="white",
            bg=BG,
            selectcolor="#333",
            activebackground=BG,
            activeforeground="white",
        ).pack(side=tk.LEFT, padx=10)

        self.lbl_time = tk.Label(ctrl, text="", fg="white", bg=BG, font=("Consolas", 9))
        self.lbl_time.pack(side=tk.RIGHT, padx=8)

        # --- Plot ---
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
        self.ax.set_facecolor(BG)
        self.ax.tick_params(colors="white")
        self.ax.set_xlabel("Time (s)", color="white")
        self.ax.set_title(
            "Drag to select · drag edges to trim · Space = play/pause selection",
            color="#aaa",
            fontsize=10,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas.mpl_connect("key_press_event", self._on_mpl_key_press)

        # --- Export ---
        exp = tk.Frame(root, bg=BG)
        exp.pack(side=tk.BOTTOM, fill=tk.X, pady=8)
        tk.Label(exp, text="Filename:", fg="white", bg=BG).pack(side=tk.LEFT, padx=6)
        self.ent_name = tk.Entry(exp, width=36)
        self.ent_name.insert(0, "trim_001.wav")
        self.ent_name.pack(side=tk.LEFT, padx=4)
        tk.Button(exp, text="Save Trim", command=self.save_trim, bg="cyan", font=("Arial", 10, "bold")).pack(
            side=tk.RIGHT, padx=10
        )

        self.root.bind("<KeyPress-space>", self._on_space_key)

        self._empty_plot()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_clock()

    def _on_mpl_key_press(self, event):
        if self._closing or event.key != " ":
            return
        self.toggle_selection_playback()

    def _on_space_key(self, event):
        if self._closing:
            return
        try:
            if self.root.focus_get() == self.ent_name:
                return
        except tk.TclError:
            pass
        self.toggle_selection_playback()
        return "break"

    def _destroy_span(self):
        if self.span is not None:
            try:
                self.span.disconnect_events()
            except Exception:
                pass
            self.span = None

    def get_span_times_sec(self):
        """Current region (seconds) from SpanSelector; ordered (t0, t1)."""
        if self.span is None or self.duration_sec <= 0:
            return 0.0, 0.0
        lo, hi = self.span.extents
        lo = float(lo)
        hi = float(hi)
        t0 = self._clamp_sec(min(lo, hi))
        t1 = self._clamp_sec(max(lo, hi))
        return t0, t1

    def _on_span_select(self, vmin, vmax):
        self._on_span_changed()

    def _on_span_move(self, vmin, vmax):
        self._on_span_changed()

    def _on_span_changed(self):
        if self.var_loop_selection.get() and self._play_kind == "selection":
            self._debounce_preview_bump()

    def _on_close(self):
        self._closing = True
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
        self._release_audio_buffers()
        try:
            plt.close(self.fig)
        except Exception:
            pass
        self.root.destroy()

    def _cancel_clock(self):
        if self._clock_after is not None:
            try:
                self.root.after_cancel(self._clock_after)
            except (tk.TclError, ValueError):
                pass
            self._clock_after = None

    def _release_audio_buffers(self):
        had = self.pydub_audio is not None or self._t_plot is not None
        self._destroy_span()
        self.pydub_audio = None
        self._t_plot = None
        self._y_plot = None
        self.audio_path = ""
        self.duration_sec = 0.0
        self._play_kind = None
        if had:
            gc.collect()

    def _cancel_timers(self):
        self._stop_selection_loop()
        self._cancel_selection_stop()

    def _stop_selection_loop(self):
        for attr in ("_preview_after", "_preview_bump_after"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.root.after_cancel(aid)
                except (tk.TclError, ValueError):
                    pass
                setattr(self, attr, None)

    def _cancel_selection_stop(self):
        aid = self._selection_stop_after
        self._selection_stop_after = None
        if aid is not None:
            try:
                self.root.after_cancel(aid)
            except (tk.TclError, ValueError):
                pass

    def _mixer_size_from_pydub(self, sample_width):
        if sample_width <= 1:
            return 8
        if sample_width == 2:
            return -16
        if sample_width == 4:
            return -32
        return -16

    def _reinit_mixer_for_segment(self, segment):
        pygame.mixer.quit()
        pygame.mixer.init(
            frequency=int(segment.frame_rate),
            channels=int(segment.channels),
            size=self._mixer_size_from_pydub(segment.sample_width),
            allowedchanges=0,
        )

    def _cleanup_playback_temp(self):
        t = self._playback_temp
        self._playback_temp = None
        if t and os.path.isfile(t):
            try:
                os.remove(t)
            except OSError:
                pass

    def get_position_ms(self):
        if not self.audio_path or self.pydub_audio is None:
            return 0
        dur = len(self.pydub_audio)
        gp = pygame.mixer.music.get_pos()
        if gp < 0:
            return max(0, min(self._play_origin_ms, dur))
        pos = self._play_origin_ms + gp
        return max(0, min(pos, dur))

    def play_from_ms(self, ms, resume_if_playing=True):
        if not self.audio_path or self.pydub_audio is None:
            return
        dur = len(self.pydub_audio)
        ms = int(max(0, min(ms, dur)))
        if dur > 0:
            ms = min(ms, dur - 1)
        start_sec = ms / 1000.0
        if ms <= 0:
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.play(start=start_sec)
        self._play_origin_ms = ms
        self._mixer_paused = False
        if not resume_if_playing:
            pygame.mixer.music.pause()
            self._mixer_paused = True

    def _empty_plot(self):
        self._destroy_span()
        self.ax.clear()
        self.ax.set_facecolor(BG)
        self.ax.tick_params(colors="white")
        self.ax.set_xlabel("Time (s)", color="white")
        self.ax.text(0.5, 0.5, "Load a WAV file", transform=self.ax.transAxes, ha="center", color="white")
        self.canvas.draw_idle()

    def _waveform_from_segment(self, seg):
        raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            raw = raw.reshape((-1, seg.channels)).mean(axis=1)
        n = raw.shape[0]
        sr = float(seg.frame_rate)
        denom = float(seg.max_possible_amplitude or 1.0)
        y = raw / denom
        if n > MAX_WAVE_POINTS:
            step = max(1, n // MAX_WAVE_POINTS)
            y = y[::step]
            t = (np.arange(0, n, step, dtype=np.float64)[: y.shape[0]]) / sr
        else:
            t = np.arange(n, dtype=np.float64) / sr
        return t, y

    def _clamp_sec(self, x):
        if self.duration_sec <= 0:
            return 0.0
        return float(max(0.0, min(x, self.duration_sec)))

    def load_file(self):
        path = filedialog.askopenfilename(initialdir=DEFAULT_IN, filetypes=[("WAV", "*.wav")])
        if path:
            self.open_audio_path(path)

    def open_audio_path(self, path):
        """Load audio from disk; safe to call repeatedly when switching files."""
        if not path or self._closing:
            return
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self._cancel_timers()
        self._cleanup_playback_temp()
        had = self.pydub_audio is not None or self._t_plot is not None
        self._destroy_span()
        self.pydub_audio = None
        self._t_plot = None
        self._y_plot = None
        self.duration_sec = 0.0
        self._play_kind = None
        if had:
            gc.collect()
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self.audio_path = path

        try:
            raw = AudioSegment.from_wav(path)
        except Exception as e:
            self.audio_path = ""
            messagebox.showerror("Load error", str(e))
            self._empty_plot()
            try:
                pygame.mixer.init()
            except pygame.error:
                pass
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
            messagebox.showerror("Audio init error", str(e))
            self._cleanup_playback_temp()
            try:
                pygame.mixer.init()
            except pygame.error:
                pass
            self._empty_plot()
            return

        self.duration_sec = len(self.pydub_audio) / 1000.0
        self._t_plot, self._y_plot = self._waveform_from_segment(self.pydub_audio)

        self.lbl_file.config(text=os.path.basename(path))
        self._redraw_waveform_and_span()
        self.reset_view()
        self._on_loop_selection_toggle()

    def _create_span_selector(self):
        self._destroy_span()
        self.span = SpanSelector(
            self.ax,
            self._on_span_select,
            "horizontal",
            minspan=0.001,
            useblit=False,
            props=dict(facecolor="dodgerblue", alpha=0.38, edgecolor="deepskyblue", linewidth=1.2),
            interactive=True,
            drag_from_anywhere=True,
            grab_range=14,
            handle_props=dict(color="cyan", linewidth=2.5, alpha=0.95),
            onmove_callback=self._on_span_move,
        )
        if self.duration_sec > 0:
            self.span.extents = (0.0, float(self.duration_sec))

    def _redraw_waveform_and_span(self):
        self._destroy_span()
        self.ax.clear()
        self.ax.set_facecolor(BG)
        self.ax.tick_params(colors="white")
        self.ax.set_xlabel("Time (s)", color="white")
        if self._t_plot is None:
            self.canvas.draw_idle()
            return
        self.ax.plot(self._t_plot, self._y_plot, color="lightgray", linewidth=0.35, alpha=0.9, zorder=1)
        self._create_span_selector()
        self.canvas.draw_idle()

    def zoom_to_selection(self):
        if self.pydub_audio is None or self.span is None:
            return
        lo, hi = self.get_span_times_sec()
        span = max(hi - lo, 1e-6)
        pad = max(span * 0.06, 0.002)
        self.ax.set_xlim(lo - pad, hi + pad)
        self.canvas.draw_idle()

    def reset_view(self):
        if self.duration_sec <= 0:
            return
        self.ax.set_xlim(0.0, self.duration_sec)
        self.canvas.draw_idle()

    def seek_ms(self, delta_ms):
        if not self.audio_path or self.pydub_audio is None or self._closing:
            return
        self._cancel_selection_stop()
        want_audio = (not self._mixer_paused) and (
            self.is_playing or pygame.mixer.music.get_busy()
        )
        pos = int(max(0, min(self.get_position_ms() + delta_ms, len(self.pydub_audio))))
        self.play_from_ms(pos, resume_if_playing=want_audio)
        self.is_playing = want_audio
        if self._play_kind == "full":
            pass
        elif self.var_loop_selection.get() and self._play_kind == "selection":
            self._debounce_preview_bump()

    def toggle_play(self):
        if not self.audio_path:
            return
        self._stop_selection_loop()
        self._cancel_selection_stop()
        self._play_kind = "full"
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            self._mixer_paused = True
            return
        if self._mixer_paused:
            pygame.mixer.music.unpause()
            self._mixer_paused = False
            self.is_playing = True
            return
        self.play_from_ms(self.get_position_ms())
        self.is_playing = True

    def play_selection(self):
        """Start playback of the highlighted region (loop if checkbox on)."""
        if not self.audio_path or self.pydub_audio is None:
            return
        self._stop_selection_loop()
        self._cancel_selection_stop()
        lo, hi = self.get_span_times_sec()
        dur_ms = int(max(0, (hi - lo) * 1000))
        if dur_ms < 1:
            messagebox.showwarning("Selection", "Region too small.")
            return
        self._play_kind = "selection"
        if self.var_loop_selection.get():
            self._start_selection_loop()
        else:
            start_ms = int(lo * 1000)
            self.play_from_ms(start_ms, resume_if_playing=True)
            self.is_playing = True
            self._mixer_paused = False
            self._selection_stop_after = self.root.after(dur_ms, self._stop_playback)

    def toggle_selection_playback(self):
        """Space: pause/resume if selection is active; otherwise start selection play."""
        if not self.audio_path or self.pydub_audio is None:
            return
        if self._play_kind == "selection" and (self.is_playing or self._mixer_paused):
            if self.is_playing:
                pygame.mixer.music.pause()
                self.is_playing = False
                self._mixer_paused = True
            else:
                pygame.mixer.music.unpause()
                self._mixer_paused = False
                self.is_playing = True
            return
        self.play_selection()

    def _stop_playback(self):
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._selection_stop_after = None
        if self._play_kind == "selection" and not self.var_loop_selection.get():
            self._play_kind = None

    def _on_loop_selection_toggle(self):
        if not self.audio_path or self._play_kind != "selection":
            return
        if self.var_loop_selection.get():
            self._start_selection_loop()
            return
        self._stop_selection_loop()
        if self.is_playing and not self._mixer_paused:
            _, hi = self.get_span_times_sec()
            end_ms = int(hi * 1000)
            rem = max(1, end_ms - self.get_position_ms())
            self._selection_stop_after = self.root.after(rem, self._stop_playback)

    def _debounce_preview_bump(self):
        if not self.var_loop_selection.get():
            return
        if self._preview_bump_after is not None:
            try:
                self.root.after_cancel(self._preview_bump_after)
            except (tk.TclError, ValueError):
                pass
        self._preview_bump_after = self.root.after(120, self._bump_preview)

    def _bump_preview(self):
        self._preview_bump_after = None
        if self._closing or not self.var_loop_selection.get() or not self.audio_path:
            return
        if self._play_kind == "selection":
            self._start_selection_loop()

    def _start_selection_loop(self):
        if self._closing or not self.audio_path or self.pydub_audio is None:
            return
        lo, hi = self.get_span_times_sec()
        dur_ms = int(max(0, (hi - lo) * 1000))
        if dur_ms < 1:
            return
        start_ms = int(lo * 1000)
        self._stop_selection_loop()
        self._cancel_selection_stop()
        self._play_kind = "selection"
        self.play_from_ms(start_ms, resume_if_playing=True)
        self.is_playing = True
        self._mixer_paused = False

        def tick():
            self._preview_after = None
            if self._closing or not self.var_loop_selection.get() or not self.audio_path:
                return
            if self._play_kind != "selection":
                return
            _, hi2 = self.get_span_times_sec()
            end_ms = int(hi2 * 1000)
            lo2, _ = self.get_span_times_sec()
            start2 = int(lo2 * 1000)
            pos = self.get_position_ms()
            if pos >= end_ms - 15:
                self.play_from_ms(start2, resume_if_playing=True)
            if not self._closing:
                self._preview_after = self.root.after(40, tick)

        self._preview_after = self.root.after(40, tick)

    def _schedule_clock(self):
        if self._closing:
            return
        try:
            if self.pydub_audio is not None:
                c = self.get_position_ms() / 1000.0
                t = len(self.pydub_audio) / 1000.0
                lo, hi = self.get_span_times_sec()
                self.lbl_time.config(text=f"{c:6.2f}s / {t:6.2f}s   sel [{lo:5.2f} – {hi:5.2f}]")
            else:
                self.lbl_time.config(text="")
        except tk.TclError:
            return
        if not self._closing:
            self._clock_after = self.root.after(120, self._schedule_clock)

    def save_trim(self):
        if self.pydub_audio is None or self.span is None:
            messagebox.showwarning("No audio", "Load a WAV first.")
            return
        t0, t1 = self.get_span_times_sec()
        a = int(round(t0 * 1000))
        b = int(round(t1 * 1000))
        if b - a < 1:
            messagebox.showwarning("Bad selection", "Adjust the highlighted region.")
            return
        name = self.ent_name.get().strip()
        if not name:
            messagebox.showwarning("Filename", "Enter a filename.")
            return
        if not name.endswith(".wav"):
            name += ".wav"
        path = os.path.join(DEFAULT_OUT, name)
        self.pydub_audio[a:b].export(path, format="wav")
        messagebox.showinfo("Saved", name)
        self._increment_name()

    def _increment_name(self):
        text = self.ent_name.get()
        base = text[:-4] if text.lower().endswith(".wav") else text
        m = re.search(r"(\d+)$", base)
        if m:
            n = int(m.group(1)) + 1
            nb = base[: m.start()] + str(n).zfill(len(m.group(1)))
            self.ent_name.delete(0, tk.END)
            self.ent_name.insert(0, nb + (".wav" if text.lower().endswith(".wav") else ""))


def main():
    root = tk.Tk()
    DragWaveSlicer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
