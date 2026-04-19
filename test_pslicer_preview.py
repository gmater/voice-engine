"""
Preview workflow tests: temp dir cleanup, chunk filtering, stress cycles.

Run:
  venv\\Scripts\\python.exe test_pslicer_preview.py
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pslicer


class TestPreviewInteractive(unittest.TestCase):
    def setUp(self):
        self.wav = _ROOT / "test_fixtures" / "pslicer_tones" / "two_speaker_tones.wav"
        if not self.wav.is_file():
            from test_pslicer_separation import build_two_speaker_tone_fixture

            build_two_speaker_tone_fixture(self.wav)

    def test_preview_export_then_commit_cleans_temp(self):
        chunks = [
            (0.05, 1.0, "SPEAKER_00", "First phrase."),
            (1.85, 3.2, "SPEAKER_01", "Second phrase."),
        ]
        lines = iter(["", "e"])

        def fake_input() -> str:
            return next(lines)

        out, ok = pslicer.interactive_preview_trims(
            str(self.wav),
            chunks,
            padding_ms=80,
            preview_play_all=False,
            input_fn=fake_input,
        )
        self.assertTrue(ok)
        self.assertEqual(len(out), 2)
        stray = [p for p in Path(tempfile.gettempdir()).glob("pslicer_preview_*") if p.is_dir()]
        self.assertEqual(stray, [], f"preview temp dirs should be removed, found {stray}")

    def test_preview_abort_returns_empty_and_cleans(self):
        chunks = [(0.05, 1.0, "SPEAKER_00", "Hi.")]
        lines = iter(["", "q"])

        def fake_input() -> str:
            return next(lines)

        out, ok = pslicer.interactive_preview_trims(
            str(self.wav),
            chunks,
            padding_ms=80,
            input_fn=fake_input,
        )
        self.assertFalse(ok)
        self.assertEqual(out, [])
        stray = [p for p in Path(tempfile.gettempdir()).glob("pslicer_preview_*") if p.is_dir()]
        self.assertEqual(stray, [], stray)

    def test_preview_exclude_one_chunk(self):
        chunks = [
            (0.05, 1.0, "SPEAKER_00", "A."),
            (1.85, 3.2, "SPEAKER_01", "B."),
        ]
        lines = iter(["", "x 1", "e"])

        def fake_input() -> str:
            return next(lines)

        out, ok = pslicer.interactive_preview_trims(
            str(self.wav),
            chunks,
            padding_ms=80,
            input_fn=fake_input,
        )
        self.assertTrue(ok)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][2], "SPEAKER_01")


class TestPreviewStress(unittest.TestCase):
    def setUp(self):
        self.wav = _ROOT / "test_fixtures" / "pslicer_tones" / "two_speaker_tones.wav"
        if not self.wav.is_file():
            from test_pslicer_separation import build_two_speaker_tone_fixture

            build_two_speaker_tone_fixture(self.wav)

    def test_many_preview_cycles_no_leftover_dirs(self):
        """Repeated temp export + rmtree should not accumulate preview folders."""
        chunks = [(0.05, 0.5, "SPEAKER_00", "Test.")]
        for _ in range(25):
            lines = iter(["", "q"])

            def fake_input() -> str:
                return next(lines)

            pslicer.interactive_preview_trims(
                str(self.wav),
                chunks,
                padding_ms=40,
                input_fn=fake_input,
            )
            gc.collect()
            stray = [p for p in Path(tempfile.gettempdir()).glob("pslicer_preview_*") if p.is_dir()]
            self.assertEqual(stray, [], stray)


class TestExportSourceIndices(unittest.TestCase):
    def test_return_source_indices_aligns_paths_to_chunks(self):
        wav = _ROOT / "test_fixtures" / "pslicer_tones" / "two_speaker_tones.wav"
        if not wav.is_file():
            from test_pslicer_separation import build_two_speaker_tone_fixture

            build_two_speaker_tone_fixture(wav)
        chunks = [(0.05, 1.0, "SPEAKER_00", "a.")]
        d = tempfile.mkdtemp(prefix="pslicer_export_idx_")
        try:
            r = pslicer.export_wav_clips(
                str(wav),
                d,
                chunks,
                stem="t",
                padding_ms=80,
                manifest_path=None,
                return_source_indices=True,
            )
            self.assertIsInstance(r, tuple)
            paths, idx = r
            self.assertEqual(len(paths), len(idx))
            self.assertEqual(idx, [0])
            for p in paths:
                self.assertTrue(os.path.isfile(p))
        finally:
            shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestPreviewInteractive))
    suite.addTests(loader.loadTestsFromTestCase(TestPreviewStress))
    suite.addTests(loader.loadTestsFromTestCase(TestExportSourceIndices))
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if r.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
