"""
Sanctum Splicer PRO — waveform audit + trim.

Run (from repo root): ``python extras/pro_slicer.py``
Or:  python 02_pro_splicer.py
"""

import os
import re
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pygame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pydub import AudioSegment

# --- Lab Paths ---
DEFAULT_IN = r"C:\AI\SanctumCore\voice_assets\raw_source\clean"
DEFAULT_OUT = r"C:\AI\SanctumCore\voice_assets\Pure_Jarvis_Audio"

os.makedirs(DEFAULT_OUT, exist_ok=True)

BG = "#1a1a1a"


class ProSplicer:
    def __init__(self, root):
        self.root = root
        self.root.title("Sanctum Splicer PRO — Neural Extraction")
        self.root.geometry("1000x750")
        self.root.configure(bg=BG)

        self.y = None
        self.sr = None
        self.duration_sec = 0.0
        self.pydub_audio = None
        self.audio_path = ""
        self.start_anchor = 0.0  # seconds (waveform / librosa time)
        self.end_anchor = 0.0

        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._playback_temp = None
        self._selection_stop_after = None
        self._clock_after = None
        self._closing = False

        pygame.mixer.init()

        self.in_line = None
        self.out_line = None

        # --- Top ---
        top_frame = tk.Frame(root, bg=BG)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Button(top_frame, text="LOAD ISOTOPE (WAV)", command=self.load_file, bg="#333", fg="white").pack(
            side=tk.LEFT, padx=10
        )
        self.lbl_file = tk.Label(top_frame, text="No File Loaded", fg="cyan", bg=BG)
        self.lbl_file.pack(side=tk.LEFT)

        # --- Plot + matplotlib zoom toolbar (magnifying glass / home) ---
        plot_frame = tk.Frame(root, bg=BG)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplots(figsize=(10, 4), facecolor=BG)
        self.ax.set_facecolor(BG)
        self.ax.tick_params(colors="white")
        self.ax.set_xlabel("Time (s)", color="white")
        self.ax.set_title(
            "Left: START (green) · Right: END (red) · Esc: exit zoom/pan",
            color="#aaaaaa",
            fontsize=10,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar.update()

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.root.bind("<Escape>", self._exit_nav_mode)

        # --- Transport ---
        mid = tk.Frame(root, bg=BG)
        mid.pack(fill=tk.X, pady=4)
        tk.Button(mid, text="PLAY / PAUSE (full file)", command=self.toggle_play, bg="gold", width=22).pack(
            side=tk.LEFT, padx=6
        )
        tk.Button(mid, text="<< -2s", command=lambda: self.seek_ms(-2000), width=8).pack(side=tk.LEFT, padx=3)
        tk.Button(mid, text="+2s >>", command=lambda: self.seek_ms(2000), width=8).pack(side=tk.LEFT, padx=3)
        tk.Button(mid, text="PLAY SELECTION", command=self.play_selection, bg="green", fg="white", width=18).pack(
            side=tk.LEFT, padx=6
        )
        self.lbl_time = tk.Label(mid, text="—", fg="white", bg=BG, font=("Consolas", 10))
        self.lbl_time.pack(side=tk.LEFT, padx=12)

        # --- Bottom ---
        bottom_frame = tk.Frame(root, bg=BG)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        tk.Label(bottom_frame, text="Save Name:", fg="white", bg=BG).pack(side=tk.LEFT, padx=5)
        self.ent_name = tk.Entry(bottom_frame, width=30)
        self.ent_name.insert(0, "jarvis_line_001")
        self.ent_name.pack(side=tk.LEFT, padx=5)

        tk.Button(bottom_frame, text="TRIM & SAVE", command=self.save_trim, bg="cyan", font=("Arial", 10, "bold")).pack(
            side=tk.RIGHT, padx=10
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_clock()

    def _on_close(self):
        self._closing = True
        if self._clock_after is not None:
            try:
                self.root.after_cancel(self._clock_after)
            except (tk.TclError, ValueError):
                pass
            self._clock_after = None
        self._cancel_selection_stop()
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self._cleanup_playback_temp()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        self.pydub_audio = None
        self.y = self.sr = None
        self.audio_path = ""
        try:
            plt.close(self.fig)
        except Exception:
            pass
        self.root.destroy()

    def _nav_mode_token(self):
        m = getattr(self.toolbar, "mode", None)
        if m is None:
            return ""
        s = str(m).lower()
        return s.rsplit(".", 1)[-1]

    def _exit_nav_mode(self, event=None):
        """Leave matplotlib zoom/pan so clicks set anchors again (toolbar radio buttons)."""
        tok = self._nav_mode_token()
        if tok == "zoom":
            self.toolbar.zoom()
        elif tok == "pan":
            self.toolbar.pan()

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

    def _cancel_selection_stop(self):
        if self._selection_stop_after is not None:
            try:
                self.root.after_cancel(self._selection_stop_after)
            except (tk.TclError, ValueError):
                pass
            self._selection_stop_after = None

    def _stop_playback(self):
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self._selection_stop_after = None

    def load_file(self):
        path = filedialog.askopenfilename(initialdir=DEFAULT_IN, filetypes=[("WAV", "*.wav")])
        if not path:
            return
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self._cancel_selection_stop()
        self._cleanup_playback_temp()
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self.audio_path = path
        self.y = None
        self.sr = None

        try:
            raw = AudioSegment.from_wav(path)
        except Exception as e:
            self.audio_path = ""
            self.pydub_audio = None
            self.y = None
            messagebox.showerror("Load error", f"Could not read WAV:\n{e}")
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
            self.y = None
            messagebox.showerror("Audio init error", f"Pygame could not open this file:\n{e}")
            self._cleanup_playback_temp()
            pygame.mixer.init()
            return

        try:
            self.y, self.sr = librosa.load(path, sr=None, mono=True)
        except Exception as e:
            self.audio_path = ""
            self.pydub_audio = None
            self.y = None
            messagebox.showerror("Librosa error", f"Could not analyze audio:\n{e}")
            self._cleanup_playback_temp()
            pygame.mixer.init()
            return

        self.duration_sec = float(len(self.y)) / float(self.sr) if self.sr else 0.0
        pydub_sec = len(self.pydub_audio) / 1000.0
        self.duration_sec = min(self.duration_sec, pydub_sec)

        self.lbl_file.config(text=os.path.basename(path))
        self.start_anchor = 0.0
        self.end_anchor = self.duration_sec
        self.plot_waveform()

    def plot_waveform(self):
        self.ax.clear()
        self.ax.set_facecolor(BG)
        self.ax.tick_params(colors="white")
        self.ax.set_xlabel("Time (s)", color="white")
        self.ax.set_title(
            "Left: START (green) · Right: END (red) · Esc: exit zoom/pan",
            color="#aaaaaa",
            fontsize=10,
        )

        if self.y is None or self.sr is None:
            self.ax.text(0.5, 0.5, "Load a WAV", transform=self.ax.transAxes, ha="center", color="white")
            self.canvas.draw()
            self.in_line = self.out_line = None
            return

        librosa.display.waveshow(self.y, sr=self.sr, ax=self.ax, color="cyan", alpha=0.65)

        self.in_line = self.ax.axvline(self.start_anchor, color="lime", linewidth=2, label="START")
        self.out_line = self.ax.axvline(self.end_anchor, color="red", linewidth=2, label="END")
        self.ax.legend(loc="upper right", facecolor="#2a2a2a", edgecolor="#444")
        self.canvas.draw()

    def _clamp_time(self, t):
        if self.y is None or self.duration_sec <= 0:
            return 0.0
        return float(max(0.0, min(t, self.duration_sec)))

    def on_click(self, event):
        if self.y is None or event.inaxes != self.ax or event.xdata is None:
            return
        if self._nav_mode_token() in ("zoom", "pan"):
            return

        t = self._clamp_time(float(event.xdata))
        if event.button == 1:
            self.start_anchor = t
        elif event.button == 3:
            self.end_anchor = t
        else:
            return
        self.update_markers()

    def update_markers(self):
        if self.in_line is None or self.out_line is None:
            return
        self.in_line.set_xdata([self.start_anchor, self.start_anchor])
        self.out_line.set_xdata([self.end_anchor, self.end_anchor])
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

    def toggle_play(self):
        if not self.audio_path:
            return
        self._cancel_selection_stop()
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
        if not self.audio_path or self.pydub_audio is None:
            return
        self._cancel_selection_stop()
        a = self._clamp_time(min(self.start_anchor, self.end_anchor))
        b = self._clamp_time(max(self.start_anchor, self.end_anchor))
        if b - a < 0.001:
            messagebox.showwarning("Selection", "Set a non-empty region (green left of red).")
            return
        dur_ms = int((b - a) * 1000)
        start_ms = int(a * 1000)
        self.play_from_ms(start_ms, resume_if_playing=True)
        self.is_playing = True
        self._mixer_paused = False
        self._selection_stop_after = self.root.after(dur_ms, self._stop_playback)

    def _schedule_clock(self):
        if self._closing:
            return
        try:
            if self.audio_path and self.pydub_audio is not None:
                curr = self.get_position_ms() / 1000.0
                total = len(self.pydub_audio) / 1000.0
                self.lbl_time.config(
                    text=f"{curr:6.2f}s / {total:6.2f}s  |  IN {self.start_anchor:6.2f}s  OUT {self.end_anchor:6.2f}s"
                )
            else:
                self.lbl_time.config(text="—")
        except tk.TclError:
            return
        if not self._closing:
            self._clock_after = self.root.after(150, self._schedule_clock)

    def save_trim(self):
        if self.pydub_audio is None:
            messagebox.showwarning("No audio", "Load a WAV file first.")
            return

        t0 = self._clamp_time(min(self.start_anchor, self.end_anchor))
        t1 = self._clamp_time(max(self.start_anchor, self.end_anchor))
        a = int(round(t0 * 1000))
        b = int(round(t1 * 1000))
        if b - a < 1:
            messagebox.showwarning("Bad selection", "Start and end are too close. Adjust the anchors.")
            return

        name = self.ent_name.get().strip()
        if not name:
            messagebox.showwarning("Filename", "Enter a save name.")
            return
        if not name.endswith(".wav"):
            name += ".wav"

        save_path = os.path.join(DEFAULT_OUT, name)
        trim = self.pydub_audio[a:b]
        trim.export(save_path, format="wav")
        messagebox.showinfo("Saved", f"Saved: {name}")
        self.increment_name()

    def increment_name(self):
        text = self.ent_name.get()
        base = text[:-4] if text.lower().endswith(".wav") else text
        res = re.search(r"(\d+)$", base)
        if res:
            num = int(res.group(1)) + 1
            new_base = base[: res.start()] + str(num).zfill(len(res.group(1)))
            new_name = new_base + (".wav" if text.lower().endswith(".wav") else "")
            self.ent_name.delete(0, tk.END)
            self.ent_name.insert(0, new_name)


def main():
    root = tk.Tk()
    ProSplicer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
