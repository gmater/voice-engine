import os
import re
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

import pygame
from pydub import AudioSegment

# --- Laboratory Pathways ---
DEFAULT_IN = r"C:\AI\SanctumCore\voice_assets\raw_source\clean"
DEFAULT_OUT = r"C:\AI\SanctumCore\voice_assets\Pure_Jarvis_Audio"

os.makedirs(DEFAULT_OUT, exist_ok=True)


class SanctumSplicer:
    def __init__(self, root):
        self.root = root
        self.root.title("Sanctum Splicer 2026")
        self.root.geometry("600x500")
        self.root.configure(bg="#1a1a1a")

        # State Variables
        self.audio_path = ""
        self.pydub_audio = None
        self.start_anchor = 0  # In milliseconds
        self.end_anchor = 0  # In milliseconds
        self.is_playing = False
        # True only after pause(); never call unpause() on first play (can fault SDL on Windows).
        self._mixer_paused = False
        # File position (ms) at the start of the current pygame play() segment
        self._play_origin_ms = 0
        self._last_clock_text = ""
        self._playback_temp = None
        self._clock_after = None
        self._closing = False

        pygame.mixer.init()

        # --- UI ELEMENTS ---
        self.label_file = tk.Label(root, text="No File Loaded", fg="cyan", bg="#1a1a1a", font=("Arial", 10))
        self.label_file.pack(pady=10)

        self.btn_load = tk.Button(root, text="LOAD WAV", command=self.load_file, bg="#333", fg="white", width=20)
        self.btn_load.pack()

        # Time Tracking
        self.label_time = tk.Label(root, text="00:00.0 / 00:00.0", fg="white", bg="#1a1a1a", font=("Consolas", 14))
        self.label_time.pack(pady=20)

        # Controls
        ctrl_frame = tk.Frame(root, bg="#1a1a1a")
        ctrl_frame.pack()

        self.btn_rewind = tk.Button(ctrl_frame, text="<< -2s", command=lambda: self.seek(-2000), width=10)
        self.btn_rewind.grid(row=0, column=0, padx=5)

        self.btn_play = tk.Button(ctrl_frame, text="PLAY/PAUSE", command=self.toggle_play, width=15, bg="gold")
        self.btn_play.grid(row=0, column=1, padx=5)

        self.btn_forward = tk.Button(ctrl_frame, text="+2s >>", command=lambda: self.seek(2000), width=10)
        self.btn_forward.grid(row=0, column=2, padx=5)

        # Anchors
        anchor_frame = tk.Frame(root, bg="#1a1a1a")
        anchor_frame.pack(pady=20)

        self.btn_in = tk.Button(anchor_frame, text="SET START ( [ )", command=self.set_in, width=15, bg="green", fg="white")
        self.btn_in.grid(row=0, column=0, padx=10)

        self.btn_out = tk.Button(anchor_frame, text="SET END ( ] )", command=self.set_out, width=15, bg="red", fg="white")
        self.btn_out.grid(row=0, column=1, padx=10)

        # Export Area
        self.label_name = tk.Label(root, text="Custom Filename:", fg="white", bg="#1a1a1a")
        self.label_name.pack()

        self.entry_name = tk.Entry(root, width=40, font=("Arial", 12))
        self.entry_name.insert(0, "jarvis_line_01")
        self.entry_name.pack(pady=5)

        self.btn_save = tk.Button(root, text="TRIM & SAVE SELECTION", command=self.save_trim, height=2, width=30, bg="cyan", font=("Arial", 10, "bold"))
        self.btn_save.pack(pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.update_clock()

    def _on_close(self):
        self._closing = True
        if self._clock_after is not None:
            try:
                self.root.after_cancel(self._clock_after)
            except (tk.TclError, ValueError):
                pass
            self._clock_after = None
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
        self.audio_path = ""
        self.root.destroy()

    def _mixer_size_from_pydub(self, sample_width):
        if sample_width <= 1:
            return 8
        if sample_width == 2:
            return -16
        if sample_width == 4:
            return -32
        return -16

    def _reinit_mixer_for_segment(self, segment):
        """Match SDL mixer to this clip; default 44.1k stereo vs 48k/mono WAV often crashes on play."""
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
        """Playback head in the file (ms). pygame get_pos() is elapsed since play(), not file time."""
        if not self.audio_path or self.pydub_audio is None:
            return 0
        dur = len(self.pydub_audio)
        gp = pygame.mixer.music.get_pos()
        if gp < 0:
            return max(0, min(self._play_origin_ms, dur))
        pos = self._play_origin_ms + gp
        return max(0, min(pos, dur))

    def play_from_ms(self, ms, resume_if_playing=True):
        """Seek to ms and start streaming from there (pygame start= seconds)."""
        if not self.audio_path or self.pydub_audio is None:
            return
        dur = len(self.pydub_audio)
        ms = int(max(0, min(ms, dur)))
        if dur > 0:
            ms = min(ms, dur - 1)
        start_sec = ms / 1000.0
        # Plain play() avoids SDL music start-time path for position 0 (better WAV compatibility).
        if ms <= 0:
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.play(start=start_sec)
        self._play_origin_ms = ms
        self._mixer_paused = False
        if not resume_if_playing:
            pygame.mixer.music.pause()
            self._mixer_paused = True

    def load_file(self):
        path = filedialog.askopenfilename(initialdir=DEFAULT_IN, filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass
        self._cleanup_playback_temp()
        self.is_playing = False
        self._mixer_paused = False
        self._play_origin_ms = 0
        self.audio_path = path
        try:
            raw = AudioSegment.from_wav(path)
        except Exception as e:
            self.audio_path = ""
            self.pydub_audio = None
            messagebox.showerror("Load error", f"Could not read WAV:\n{e}")
            return
        # 24-bit WAV: decode to 16-bit file for SDL; mixer init must match what we load.
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
            messagebox.showerror("Audio init error", f"Pygame could not open this file:\n{e}")
            self._cleanup_playback_temp()
            pygame.mixer.init()
            return
        self.label_file.config(text=os.path.basename(path))
        self.start_anchor = 0
        self.end_anchor = len(self.pydub_audio)

    def toggle_play(self):
        if not self.audio_path:
            return
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

    def seek(self, delta_ms):
        if not self.audio_path or self.pydub_audio is None:
            return
        # Don't rely only on is_playing: it is False while paused even if SDL still reports busy briefly,
        # and can be stale vs get_busy() after load or edge cases.
        want_audio = (not self._mixer_paused) and (
            self.is_playing or pygame.mixer.music.get_busy()
        )
        pos = int(max(0, min(self.get_position_ms() + delta_ms, len(self.pydub_audio))))
        self.play_from_ms(pos, resume_if_playing=want_audio)
        self.is_playing = want_audio

    def set_in(self):
        if not self.audio_path:
            return
        self.start_anchor = int(self.get_position_ms())
        print(f"Start Anchor: {self.start_anchor}ms")

    def set_out(self):
        if not self.audio_path:
            return
        self.end_anchor = int(self.get_position_ms())
        print(f"End Anchor: {self.end_anchor}ms")

    def update_clock(self):
        if self._closing:
            return
        try:
            if self.audio_path and self.pydub_audio is not None:
                curr = self.get_position_ms() / 1000.0
                total = len(self.pydub_audio) / 1000.0
                text = f"{curr:.1f}s / {total:.1f}s | IN: {self.start_anchor / 1000:.1f}s OUT: {self.end_anchor / 1000:.1f}s"
                if text != self._last_clock_text:
                    self.label_time.config(text=text)
                    self._last_clock_text = text
        except tk.TclError:
            return
        if not self._closing:
            self._clock_after = self.root.after(100, self.update_clock)

    def save_trim(self):
        if self.pydub_audio is None:
            messagebox.showwarning("No audio", "Load a WAV file first.")
            return

        a, b = int(self.start_anchor), int(self.end_anchor)
        if a > b:
            a, b = b, a
        if b - a < 1:
            messagebox.showwarning("Bad selection", "Start and end are equal or invalid. Set IN/OUT again.")
            return

        name = self.entry_name.get().strip()
        if not name:
            messagebox.showwarning("Filename", "Enter a filename.")
            return
        if not name.endswith(".wav"):
            name += ".wav"

        save_path = os.path.join(DEFAULT_OUT, name)
        trim = self.pydub_audio[a:b]
        trim.export(save_path, format="wav")

        messagebox.showinfo("Success", f"Saved to: {name}")
        self.increment_filename()

    def increment_filename(self):
        current = self.entry_name.get()
        base = current[:-4] if current.lower().endswith(".wav") else current
        match = re.search(r"(\d+)$", base)
        if match:
            num = int(match.group(1))
            new_base = base[: match.start()] + str(num + 1).zfill(len(match.group(1)))
            new_name = new_base + (".wav" if current.lower().endswith(".wav") else "")
            self.entry_name.delete(0, tk.END)
            self.entry_name.insert(0, new_name)


if __name__ == "__main__":
    root = tk.Tk()
    SanctumSplicer(root)
    root.mainloop()
