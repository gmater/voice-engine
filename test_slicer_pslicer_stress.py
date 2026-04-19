"""
Stress / leak checks for Voice Engine AI trim preview (PslicerTrimPreviewDialog).

Validates repeated open/close, list + waveform churn, export path, and temp-dir cleanup.

Run:
  venv\\Scripts\\python.exe test_slicer_pslicer_stress.py
"""

from __future__ import annotations

import gc
import os
import queue
import shutil
import sys
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import slicer  # noqa: E402


def _stray_sanctum_preview_dirs() -> list[Path]:
    return [p for p in Path(tempfile.gettempdir()).glob("sanctum_pslicer_preview_*") if p.is_dir()]


class TestPslicerPreviewDialogStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wav = _ROOT / "test_fixtures" / "pslicer_tones" / "two_speaker_tones.wav"
        if not cls.wav.is_file():
            from test_pslicer_separation import build_two_speaker_tone_fixture

            build_two_speaker_tone_fixture(cls.wav)

    def setUp(self) -> None:
        # Baseline: ignore zombie temp dirs from other crashed runs on the same machine.
        self._sanctum_baseline = {p.resolve() for p in _stray_sanctum_preview_dirs()}

    def _assert_no_new_sanctum(self, msg: str = "") -> None:
        after = {p.resolve() for p in _stray_sanctum_preview_dirs()}
        extra = after - self._sanctum_baseline
        self.assertFalse(extra, msg or f"New sanctum preview dirs not removed: {extra}")

    def _pump(self, root: tk.Tk, n: int = 8) -> None:
        for _ in range(n):
            root.update_idletasks()
            try:
                root.update()
            except tk.TclError:
                break

    def test_many_open_close_cycles_no_leftover_dirs(self):
        root = tk.Tk()
        root.withdraw()
        app: slicer.SanctumSurgicalV3 | None = None
        try:
            app = slicer.SanctumSurgicalV3(root)
            app.load_audio_path(str(self.wav))
            self._pump(root)
            chunks = [
                (0.05, 1.0, "SPEAKER_00", "First."),
                (1.85, 3.2, "SPEAKER_01", "Second."),
            ]
            for _ in range(30):
                dlg = slicer.PslicerTrimPreviewDialog(app, str(self.wav), chunks, padding_ms=80)
                self.assertIsNotNone(dlg.top)
                self._pump(root, 12)
                dlg._include_all()
                dlg._exclude_all()
                dlg._include_all()
                dlg._refresh_meta_and_waveform(0)
                if len(dlg.paths) > 1 and dlg._list is not None:
                    dlg._list.selection_clear(0, tk.END)
                    dlg._list.selection_set(1)
                    dlg._on_list_select()
                    self._pump(root, 4)
                dlg._on_user_close()
                self._pump(root, 4)
                gc.collect()
                self._assert_no_new_sanctum()
        finally:
            if app is not None:
                try:
                    app._on_close()
                except Exception:
                    pass
            try:
                root.destroy()
            except tk.TclError:
                pass
            gc.collect()
            self._assert_no_new_sanctum()

    def test_speaker_view_filter_and_export_toggle(self):
        root = tk.Tk()
        root.withdraw()
        app: slicer.SanctumSurgicalV3 | None = None
        try:
            app = slicer.SanctumSurgicalV3(root)
            app.load_audio_path(str(self.wav))
            self._pump(root)
            chunks = [
                (0.05, 1.0, "SPEAKER_00", "A."),
                (1.85, 3.2, "SPEAKER_01", "B."),
            ]
            dlg = slicer.PslicerTrimPreviewDialog(app, str(self.wav), chunks, padding_ms=80)
            self._pump(root, 8)
            self.assertEqual(dlg._unique_speakers, ["SPEAKER_00", "SPEAKER_01"])
            dlg._view_filter_var.set("SPEAKER_01")
            dlg._populate_list_for_current_filter()
            self._pump(root, 4)
            self.assertEqual(len(dlg._visible_j), 1)
            self.assertEqual(dlg.chunks[dlg.src_idx[dlg._visible_j[0]]][2], "SPEAKER_01")
            dlg._apply_export_preset("Only · SPEAKER_01")
            self.assertFalse(dlg._included[0])
            self.assertTrue(dlg._included[1])
            self.assertEqual(dlg._export_preset_var.get(), "Only · SPEAKER_01")
            dlg._on_user_close()
            self._pump(root, 4)
            self._assert_no_new_sanctum()
        finally:
            if app is not None:
                try:
                    app._on_close()
                except Exception:
                    pass
            try:
                root.destroy()
            except tk.TclError:
                pass

    def test_pslicer_loading_ui_open_close_no_leak(self):
        """Loading dialog + queue polling can be torn down without leaving stray windows."""
        root = tk.Tk()
        root.withdraw()
        app: slicer.SanctumSurgicalV3 | None = None
        try:
            app = slicer.SanctumSurgicalV3(root)
            app.load_audio_path(str(self.wav))
            self._pump(root)
            app._close_pslicer_loading_ui()
            app._pslicer_phase_queue = queue.Queue()
            app._open_pslicer_loading_ui(str(self.wav))
            self._pump(root, 6)
            self.assertIsNotNone(app._pslicer_load_win)
            app._pslicer_phase_queue.put_nowait("Test phase message…")
            app._poll_pslicer_phase_queue()
            self._pump(root, 4)
            app._close_pslicer_loading_ui()
            self._pump(root, 4)
            self.assertIsNone(app._pslicer_load_win)
            self.assertIsNone(app._pslicer_load_pb)
        finally:
            if app is not None:
                try:
                    app._on_close()
                except Exception:
                    pass
            try:
                root.destroy()
            except tk.TclError:
                pass

    def test_double_close_is_safe(self):
        root = tk.Tk()
        root.withdraw()
        app: slicer.SanctumSurgicalV3 | None = None
        try:
            app = slicer.SanctumSurgicalV3(root)
            app.load_audio_path(str(self.wav))
            self._pump(root)
            dlg = slicer.PslicerTrimPreviewDialog(
                app, str(self.wav), [(0.05, 0.5, "SPEAKER_00", "x.")], padding_ms=40
            )
            self._pump(root, 6)
            dlg._on_user_close()
            dlg._on_user_close()
            self._pump(root, 4)
            self._assert_no_new_sanctum()
        finally:
            if app is not None:
                try:
                    app._on_close()
                except Exception:
                    pass
            try:
                root.destroy()
            except tk.TclError:
                pass

    def test_apply_trim_edits_updates_chunk_and_preview(self):
        root = tk.Tk()
        root.withdraw()
        app: slicer.SanctumSurgicalV3 | None = None
        try:
            app = slicer.SanctumSurgicalV3(root)
            app.load_audio_path(str(self.wav))
            self._pump(root, 4)
            aligned_words = [
                {"word": "First", "start": 0.05, "end": 0.45, "speaker": "SPEAKER_00"},
                {"word": "half.", "start": 0.55, "end": 0.95, "speaker": "SPEAKER_00"},
                {"word": "Second", "start": 1.85, "end": 2.4, "speaker": "SPEAKER_01"},
                {"word": "bit.", "start": 2.5, "end": 3.1, "speaker": "SPEAKER_01"},
            ]
            chunks = [
                (0.05, 1.0, "SPEAKER_00", "First half."),
                (1.85, 3.2, "SPEAKER_01", "Second bit."),
            ]
            dlg = slicer.PslicerTrimPreviewDialog(
                app, str(self.wav), chunks, padding_ms=80, aligned_words=aligned_words
            )
            self._pump(root, 8)
            self.assertIsNotNone(dlg.top)
            j0 = dlg._visible_j[0]
            si0 = dlg.src_idx[j0]
            t0_0, t1_0 = dlg.chunks[si0][0], dlg.chunks[si0][1]
            dlg._trim_t0_var.set("0.50")
            dlg._trim_t1_var.set("1.00")
            dlg._apply_pslicer_trim_edits()
            self._pump(root, 4)
            self.assertAlmostEqual(dlg.chunks[si0][0], 0.5, places=3)
            self.assertAlmostEqual(dlg.chunks[si0][1], 1.0, places=3)
            self.assertIn("half.", dlg.chunks[si0][3])
            self.assertNotIn("First", dlg.chunks[si0][3])
            self.assertTrue(os.path.isfile(dlg.paths[j0]))
            dlg._reset_pslicer_clip_to_ai()
            self._pump(root, 4)
            self.assertAlmostEqual(dlg.chunks[si0][0], t0_0, places=4)
            self.assertAlmostEqual(dlg.chunks[si0][1], t1_0, places=4)
            self.assertEqual(dlg.chunks[si0][3], "First half.")
            dlg._on_user_close()
            self._pump(root, 4)
        finally:
            if app is not None:
                try:
                    app._on_close()
                except Exception:
                    pass
            try:
                root.destroy()
            except tk.TclError:
                pass

    def test_save_current_clip_as_writes_file(self):
        root = tk.Tk()
        root.withdraw()
        out = tempfile.mkdtemp(prefix="slicer_ai_trim_single_")
        app: slicer.SanctumSurgicalV3 | None = None
        dest = os.path.join(out, "saved_one.wav")
        try:
            app = slicer.SanctumSurgicalV3(root)
            app.load_audio_path(str(self.wav))
            self._pump(root, 4)
            chunks = [(0.05, 0.9, "SPEAKER_00", "One.")]
            dlg = slicer.PslicerTrimPreviewDialog(app, str(self.wav), chunks, padding_ms=80)
            self._pump(root, 8)
            self.assertIsNotNone(dlg.top)
            with patch("slicer.filedialog.asksaveasfilename", return_value=dest), patch(
                "slicer.messagebox.showinfo"
            ), patch("slicer.messagebox.showwarning"):
                dlg._save_current_pslicer_clip_as()
            self._pump(root, 4)
            self.assertTrue(os.path.isfile(dest))
            dlg._on_user_close()
            self._pump(root, 4)
        finally:
            shutil.rmtree(out, ignore_errors=True)
            if app is not None:
                try:
                    app._on_close()
                except Exception:
                    pass
            try:
                root.destroy()
            except tk.TclError:
                pass

    def test_export_final_cleans_temp_and_writes_wavs(self):
        root = tk.Tk()
        root.withdraw()
        out = tempfile.mkdtemp(prefix="slicer_ai_trim_export_")
        app: slicer.SanctumSurgicalV3 | None = None
        try:
            app = slicer.SanctumSurgicalV3(root)
            app.load_audio_path(str(self.wav))
            self.assertTrue(app.set_export_dir(out))
            self._pump(root)
            chunks = [(0.05, 1.0, "SPEAKER_00", "Only.")]
            dlg = slicer.PslicerTrimPreviewDialog(app, str(self.wav), chunks, padding_ms=80)
            self._pump(root, 8)
            with patch("slicer.messagebox.showinfo"), patch("slicer.messagebox.showwarning"):
                dlg._export_final()
            self._pump(root, 8)
            self.assertIsNone(dlg.top)
            self._assert_no_new_sanctum()
            wavs = [f for f in os.listdir(out) if f.lower().endswith(".wav")]
            self.assertGreaterEqual(len(wavs), 1)
        finally:
            shutil.rmtree(out, ignore_errors=True)
            if app is not None:
                try:
                    app._on_close()
                except Exception:
                    pass
            try:
                root.destroy()
            except tk.TclError:
                pass
            gc.collect()
            self._assert_no_new_sanctum()

    def test_heavy_trim_transcript_wave_reset_cycles(self) -> None:
        """Many apply / reset / wave refresh cycles with aligned_words — stable temp dirs, no TclError."""
        root = tk.Tk()
        root.withdraw()
        app: slicer.SanctumSurgicalV3 | None = None
        aligned_words = [
            {"word": "First", "start": 0.05, "end": 0.45, "speaker": "SPEAKER_00"},
            {"word": "half.", "start": 0.55, "end": 0.95, "speaker": "SPEAKER_00"},
            {"word": "Second", "start": 1.85, "end": 2.4, "speaker": "SPEAKER_01"},
            {"word": "bit.", "start": 2.5, "end": 3.1, "speaker": "SPEAKER_01"},
        ]
        chunks = [
            (0.05, 1.0, "SPEAKER_00", "First half."),
            (1.85, 3.2, "SPEAKER_01", "Second bit."),
        ]
        try:
            app = slicer.SanctumSurgicalV3(root)
            app.load_audio_path(str(self.wav))
            self._pump(root, 4)
            for _ in range(22):
                dlg = slicer.PslicerTrimPreviewDialog(
                    app, str(self.wav), chunks, padding_ms=80, aligned_words=aligned_words
                )
                self.assertIsNotNone(dlg.top)
                self._pump(root, 6)
                for _i in range(4):
                    dlg._trim_t0_var.set("0.52")
                    dlg._trim_t1_var.set("0.98")
                    dlg._apply_pslicer_trim_edits()
                    self._pump(root, 2)
                    dlg._wave_context_var.set("0.28")
                    dlg._refresh_pslicer_wave_preview()
                    self._pump(root, 2)
                    dlg._wave_context_var.set("0.55")
                    dlg._refresh_pslicer_wave_preview()
                    self._pump(root, 2)
                    dlg._reset_pslicer_clip_to_ai()
                    self._pump(root, 2)
                if len(dlg.paths) > 1 and dlg._list is not None:
                    dlg._list.selection_clear(0, tk.END)
                    dlg._list.selection_set(1)
                    dlg._on_list_select()
                    self._pump(root, 3)
                    dlg._list.selection_clear(0, tk.END)
                    dlg._list.selection_set(0)
                    dlg._on_list_select()
                    self._pump(root, 3)
                dlg._on_user_close()
                self._pump(root, 4)
                gc.collect()
                self._assert_no_new_sanctum()
        finally:
            if app is not None:
                try:
                    app._on_close()
                except Exception:
                    pass
            try:
                root.destroy()
            except tk.TclError:
                pass
            gc.collect()
            self._assert_no_new_sanctum()


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPslicerPreviewDialogStress))
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if r.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
