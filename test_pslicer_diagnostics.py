"""
Diagnostics for pslicer transcript + AI-trim helpers: fuzz, tracemalloc growth caps, no crashes.

Run:
  venv\\Scripts\\python.exe test_pslicer_diagnostics.py
"""

from __future__ import annotations

import gc
import os
import random
import sys
import tempfile
import tracemalloc
import unittest
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pslicer  # noqa: E402
import wav_metadata  # noqa: E402
from pydub.generators import Sine  # noqa: E402


def _random_words(rng: random.Random, n: int) -> list[dict]:
    words: list[dict] = []
    t = 0.0
    for i in range(n):
        dur = rng.uniform(0.02, 0.15)
        words.append(
            {
                "word": f"w{i}",
                "start": t,
                "end": t + dur,
                "speaker": f"SPEAKER_{rng.randint(0, 3):02d}",
            }
        )
        t += dur + rng.uniform(0.0, 0.08)
    return words


class TestTranscriptFuzz(unittest.TestCase):
    def test_transcript_for_time_range_random_inputs_no_crash(self) -> None:
        rng = random.Random(2026)
        for _ in range(2500):
            words = _random_words(rng, rng.randint(0, 80))
            t0 = rng.uniform(-0.5, 5.0)
            t1 = rng.uniform(-0.5, 5.0)
            r = pslicer.transcript_for_time_range(words, t0, t1)
            self.assertIsInstance(r, str)
        gc.collect()


class TestTranscriptTracemalloc(unittest.TestCase):
    def test_transcript_repeated_calls_bounded_tracemalloc_growth(self) -> None:
        """Many calls should not show runaway allocations in pslicer transcript path."""
        words = _random_words(random.Random(1), 400)
        tracemalloc.stop()
        tracemalloc.clear_traces()
        tracemalloc.start(25)
        for _ in range(400):
            pslicer.transcript_for_time_range(words, 0.1, 2.0)
        gc.collect()
        s1 = tracemalloc.take_snapshot()
        for _ in range(12000):
            pslicer.transcript_for_time_range(words, 0.05, 3.5)
        gc.collect()
        s2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diffs = s2.compare_to(s1, "lineno")
        pslicer_positive_kb = 0.0
        for d in diffs[:50]:
            if d.size_diff <= 0:
                continue
            try:
                fn = d.traceback[0].filename
            except IndexError:
                continue
            if Path(fn).name != "pslicer.py":
                continue
            pslicer_positive_kb += d.size_diff / 1024.0
        # Only ``transcript_for_time_range`` runs between snapshots; cap catches runaway leaks.
        self.assertLess(
            pslicer_positive_kb,
            8192.0,
            f"net tracemalloc growth attributed to pslicer.py ~{pslicer_positive_kb:.1f} KB",
        )


class TestWavMetadataEmbed(unittest.TestCase):
    def _write_minimal_wav(self, path: str) -> None:
        seg = Sine(440).to_audio_segment(duration=80).apply_gain(-20)
        seg.export(path, format="wav")

    def test_embed_voice_engine_many_roundtrips_no_crash(self) -> None:
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self._write_minimal_wav(wav)
            for i in range(800):
                wav_metadata.embed_voice_engine_wav_metadata(
                    wav,
                    source_audio_basename="source.wav",
                    speaker="SPEAKER_00",
                    transcript=f"line {i} " * 12,
                    trim_export_start_ms=10 + i % 50,
                    trim_export_end_ms=70 + i % 40,
                )
                with wave.open(wav, "rb") as wf:
                    self.assertGreater(wf.getnframes(), 0)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
        gc.collect()

    def test_embed_tracemalloc_growth_bounded(self) -> None:
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._write_minimal_wav(wav)
        try:
            tracemalloc.stop()
            tracemalloc.clear_traces()
            tracemalloc.start(25)
            for _ in range(200):
                wav_metadata.embed_voice_engine_wav_metadata(
                    wav,
                    source_audio_basename="x.wav",
                    speaker="A",
                    transcript="t",
                    trim_export_start_ms=0,
                    trim_export_end_ms=50,
                )
            gc.collect()
            s1 = tracemalloc.take_snapshot()
            for _ in range(5000):
                wav_metadata.embed_voice_engine_wav_metadata(
                    wav,
                    source_audio_basename="y.wav",
                    speaker="B",
                    transcript="long " * 400,
                    trim_export_start_ms=1,
                    trim_export_end_ms=99,
                )
            gc.collect()
            s2 = tracemalloc.take_snapshot()
            tracemalloc.stop()

            diffs = s2.compare_to(s1, "lineno")
            wm_kb = 0.0
            for d in diffs[:80]:
                if d.size_diff <= 0:
                    continue
                try:
                    fn = d.traceback[0].filename
                except IndexError:
                    continue
                if Path(fn).name != "wav_metadata.py":
                    continue
                wm_kb += d.size_diff / 1024.0
            self.assertLess(
                wm_kb,
                12288.0,
                f"net tracemalloc growth attributed to wav_metadata.py ~{wm_kb:.1f} KB",
            )
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass
            gc.collect()


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriptFuzz))
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriptTracemalloc))
    suite.addTests(loader.loadTestsFromTestCase(TestWavMetadataEmbed))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
