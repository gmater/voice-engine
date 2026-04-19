"""
Tests for persisted HF token settings (no GUI).

Run:
  venv\\Scripts\\python.exe test_voice_engine_settings.py
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestVoiceEngineSettingsFile(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self._path = str(Path(self._td.name) / "settings.json")
        os.environ["VOICE_ENGINE_SETTINGS_PATH"] = self._path
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    def tearDown(self) -> None:
        os.environ.pop("VOICE_ENGINE_SETTINGS_PATH", None)

    def test_save_load_roundtrip(self) -> None:
        import slicer as s

        self.assertEqual(str(s.voice_engine_settings_path()), self._path)
        tok = "hf_unit_test_fake_token_xxxxxxxxxxxxxxxx"
        s.save_voice_engine_hf_token(tok)
        self.assertTrue(Path(self._path).is_file())
        data = json.loads(Path(self._path).read_text(encoding="utf-8"))
        self.assertEqual(data.get("hf_token"), tok)
        os.environ.pop("HF_TOKEN", None)
        s.load_voice_engine_settings_into_environ()
        self.assertEqual(os.environ.get("HF_TOKEN"), tok)

    def test_clear_removes_file_and_env(self) -> None:
        import slicer as s

        s.save_voice_engine_hf_token("hf_clear_me_xxxxxxxxxxxx")
        self.assertTrue(Path(self._path).is_file())
        s.save_voice_engine_hf_token(None)
        self.assertFalse(Path(self._path).is_file())
        self.assertIsNone(os.environ.get("HF_TOKEN"))

    def test_env_wins_over_file_on_load(self) -> None:
        import slicer as s

        s.save_voice_engine_hf_token("hf_from_file_xxxxxxxxxxxx")
        self.assertEqual(os.environ.get("HF_TOKEN"), "hf_from_file_xxxxxxxxxxxx")
        os.environ["HF_TOKEN"] = "hf_from_env_xxxxxxxxxxxx"
        s.load_voice_engine_settings_into_environ()
        self.assertEqual(os.environ.get("HF_TOKEN"), "hf_from_env_xxxxxxxxxxxx")

    def test_repeated_save_tracemalloc_bounded(self) -> None:
        import slicer as s

        tracemalloc.stop()
        tracemalloc.clear_traces()
        tracemalloc.start(25)
        for _ in range(400):
            s.save_voice_engine_hf_token("hf_repeat_xxxxxxxxxxxx")
        gc.collect()
        s1 = tracemalloc.take_snapshot()
        for _ in range(4000):
            s.save_voice_engine_hf_token("hf_repeat_xxxxxxxxxxxx")
        gc.collect()
        s2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        diffs = s2.compare_to(s1, "lineno")
        slicer_kb = 0.0
        for d in diffs[:40]:
            if d.size_diff <= 0:
                continue
            try:
                fn = d.traceback[0].filename
            except IndexError:
                continue
            if Path(fn).name != "slicer.py":
                continue
            slicer_kb += d.size_diff / 1024.0
        self.assertLess(slicer_kb, 4096.0, f"slicer.py tracemalloc growth ~{slicer_kb:.1f} KB")


class TestAiTrimCallbackHardening(unittest.TestCase):
    def test_compute_done_when_closing_does_not_raise(self) -> None:
        if importlib.util.find_spec("tkinter") is None:
            self.skipTest("no tkinter")
        import tkinter as tk

        import slicer

        root = tk.Tk()
        root.withdraw()
        app = None
        try:
            app = slicer.SanctumSurgicalV3(root)
            app._closing = True
            app._on_pslicer_compute_done("would show dialog if not closing", None, None)
        finally:
            if app is not None:
                try:
                    app._on_close()
                except (tk.TclError, RuntimeError):
                    try:
                        root.destroy()
                    except tk.TclError:
                        pass


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestVoiceEngineSettingsFile))
    suite.addTests(loader.loadTestsFromTestCase(TestAiTrimCallbackHardening))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
