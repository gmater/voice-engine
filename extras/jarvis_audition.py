"""
GUI: audition WAVs under raw_source/clean, move KEEP -> Pure_Jarvis_Audio, TRASH -> Trash.

Playback uses winsound (PCM). Non-PCM / float WAVs are decoded with pydub and sent
through a temp 16-bit PCM file so PlaySound works reliably on Windows.
"""

from __future__ import annotations

import os
import re
import tempfile
import wave
import winsound
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from pydub import AudioSegment

# --- Configuration ---
SOURCE_DIR = Path(r"C:\AI\SanctumCore\voice_assets\raw_source\clean")
PURE_DIR = Path(r"C:\AI\SanctumCore\voice_assets\Pure_Jarvis_Audio")
TRASH_DIR = Path(r"C:\AI\SanctumCore\voice_assets\Trash")

_WIN_INVALID = re.compile(r'[<>:"/\\|?*]')


def _safe_label(name: str, max_len: int = 120) -> str:
    name = _WIN_INVALID.sub("_", name.strip()) or "unknown"
    return name[:max_len]


class JarvisAuditionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("JARVIS Voice DNA Auditioner")
        self.root.geometry("520x320")

        PURE_DIR.mkdir(parents=True, exist_ok=True)
        TRASH_DIR.mkdir(parents=True, exist_ok=True)

        self.files = self._collect_wavs()
        self.files.sort()
        self.index = 0
        self._done_dialog_shown = False
        self._temp_play: Path | None = None

        self.label = tk.Label(root, text="", wraplength=480, font=("Segoe UI", 10), justify=tk.LEFT)
        self.label.pack(pady=16, padx=12, anchor="w")

        self.btn_play = tk.Button(root, text="Play (Space)", width=22, command=self.play_audio)
        self.btn_play.pack(pady=4)

        self.btn_keep = tk.Button(
            root, text="Keep — JARVIS (K)", width=22, bg="#1e7b34", fg="white", command=self.keep_file
        )
        self.btn_keep.pack(pady=4)

        self.btn_trash = tk.Button(
            root, text="Trash — other (T)", width=22, bg="#a61b1b", fg="white", command=self.trash_file
        )
        self.btn_trash.pack(pady=4)

        self.hint = tk.Label(
            root,
            text="Shortcuts: Space = play · K = keep · T = trash · Esc = stop sound",
            font=("Segoe UI", 8),
            fg="#555",
        )
        self.hint.pack(side="bottom", pady=(0, 4))

        self.status = tk.Label(root, text="", font=("Segoe UI", 9))
        self.status.pack(side="bottom", pady=8)

        self.root.bind("<space>", lambda e: self.play_audio())
        self.root.bind("k", lambda e: self.keep_file())
        self.root.bind("K", lambda e: self.keep_file())
        self.root.bind("t", lambda e: self.trash_file())
        self.root.bind("T", lambda e: self.trash_file())
        self.root.bind("<Escape>", lambda e: self.stop_audio())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.update_ui()

    def _on_close(self) -> None:
        self.stop_audio()
        self._cleanup_temp_play()
        self.root.destroy()

    def _collect_wavs(self) -> list[Path]:
        if not SOURCE_DIR.is_dir():
            return []
        out: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(SOURCE_DIR):
            dirnames.sort()
            for f in filenames:
                if f.lower().endswith(".wav"):
                    out.append(Path(dirpath) / f)
        return out

    @staticmethod
    def _unique_dest(dest_dir: Path, src: Path) -> Path:
        parent = _safe_label(src.parent.name)
        stem = _safe_label(src.stem)
        suffix = src.suffix.lower() or ".wav"
        candidate = dest_dir / f"{parent}__{stem}{suffix}"
        n = 2
        while candidate.exists():
            candidate = dest_dir / f"{parent}__{stem}_{n}{suffix}"
            n += 1
        return candidate

    def _refresh_controls(self) -> None:
        active = self.index < len(self.files)
        state = tk.NORMAL if active else tk.DISABLED
        self.btn_play.config(state=state)
        self.btn_keep.config(state=state)
        self.btn_trash.config(state=state)

    def update_ui(self) -> None:
        if self.index < len(self.files):
            cur = self.files[self.index]
            self.label.config(
                text=(
                    f"Folder: {cur.parent.name}\n"
                    f"File: {cur.name}\n"
                    f"Path: {cur}"
                )
            )
            self.status.config(text=f"Progress: {self.index + 1} / {len(self.files)}")
        else:
            if not self.files:
                self.label.config(text=f"No .wav files under:\n{SOURCE_DIR}")
                self.status.config(text="Progress: 0 / 0")
            else:
                self.label.config(text="Audit complete — no more files in the queue.")
                self.status.config(text=f"Progress: {len(self.files)} / {len(self.files)}")
            if self.files and not self._done_dialog_shown:
                self._done_dialog_shown = True
                messagebox.showinfo("Done", "Finished auditing all WAV files.")
            elif not self.files and not self._done_dialog_shown:
                self._done_dialog_shown = True
                messagebox.showwarning(
                    "No files",
                    f"No .wav files found under:\n{SOURCE_DIR}\n\n"
                    "Add clips there (e.g. from your harvest pipeline), then restart.",
                )
        self._refresh_controls()

    def stop_audio(self) -> None:
        winsound.PlaySound(None, winsound.SND_PURGE)

    def _cleanup_temp_play(self) -> None:
        if self._temp_play is not None:
            try:
                self._temp_play.unlink(missing_ok=True)
            except OSError:
                pass
            self._temp_play = None

    @staticmethod
    def _is_standard_pcm_wav(path: Path) -> bool:
        """winsound.PlaySound needs classic PCM WAV (float/24-bit often fails)."""
        try:
            with wave.open(str(path), "rb") as w:
                if w.getsampwidth() != 2:
                    return False
                if w.getnchannels() not in (1, 2):
                    return False
                return w.getframerate() in (
                    8000,
                    11025,
                    16000,
                    22050,
                    24000,
                    32000,
                    44100,
                    48000,
                )
        except (wave.Error, OSError):
            return False

    def play_audio(self) -> None:
        if self.index >= len(self.files):
            return
        path = self.files[self.index]
        self.stop_audio()
        self._cleanup_temp_play()

        if self._is_standard_pcm_wav(path):
            try:
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except RuntimeError:
                pass

        try:
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(22050)
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="jarvis_aud_")
            os.close(fd)
            tmp_path = Path(tmp)
            audio.export(tmp_path, format="wav")
            self._temp_play = tmp_path
            winsound.PlaySound(str(tmp_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            messagebox.showerror("Playback", f"Could not play:\n{path}\n\n{exc}")

    def keep_file(self) -> None:
        self._move_current(PURE_DIR)

    def trash_file(self) -> None:
        self._move_current(TRASH_DIR)

    def _move_current(self, dest_root: Path) -> None:
        if self.index >= len(self.files):
            return
        self.stop_audio()
        self._cleanup_temp_play()
        src = self.files[self.index]
        try:
            dst = self._unique_dest(dest_root, src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
        except OSError as exc:
            messagebox.showerror("Move failed", f"{src}\n→\n{exc}")
            return
        self.index += 1
        self.update_ui()


if __name__ == "__main__":
    _root = tk.Tk()
    JarvisAuditionApp(_root)
    _root.mainloop()
