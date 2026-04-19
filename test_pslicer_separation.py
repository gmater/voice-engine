"""
Tests for pslicer contextual speaker smoothing and WAV export separation.

Run (no GPU / no HF token required):
  venv\\Scripts\\python.exe test_pslicer_separation.py

Uses a pre-built two-tone WAV (distinct fundamentals per "speaker" region) plus
synthetic word timings with intentional diarization noise; checks that smoothing
fixes labels and that exported clips match the expected dominant frequency bands.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

# Repo root: voice_engine/
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pydub import AudioSegment
from pydub.generators import Sine

import pslicer


def _dominant_peak_hz(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(n)
    if ch != 1:
        raise ValueError("expected mono")
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if len(y) < 256:
        return 0.0
    y = y / (np.max(np.abs(y)) + 1e-9)
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y * win))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    k = int(np.argmax(spec[1:]) + 1)
    return float(freqs[k])


def build_two_speaker_tone_fixture(
    out_path: Path,
    *,
    f0: float = 440.0,
    f1: float = 554.37,
    dur_a_ms: int = 1600,
    gap_ms: int = 250,
    dur_b_ms: int = 1600,
) -> None:
    """Two back-to-back sine bursts (mono 16 kHz) — simulates two distinct voice bands."""
    a = Sine(f0).to_audio_segment(duration=dur_a_ms, volume=-12)
    gap = AudioSegment.silent(duration=gap_ms)
    b = Sine(f1).to_audio_segment(duration=dur_b_ms, volume=-12)
    mix = a + gap + b
    mix = mix.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mix.export(str(out_path), format="wav")


class TestContextualSmoothing(unittest.TestCase):
    def test_duration_vote_fixes_single_word_flip(self):
        """Middle word wrongly tagged; neighbors agree — smoothing should correct it."""
        words = [
            {"word": "Alpha", "start": 0.0, "end": 0.2, "speaker": "SPEAKER_00"},
            {"word": "beta", "start": 0.25, "end": 0.45, "speaker": "SPEAKER_01"},  # glitch
            {"word": "gamma.", "start": 0.5, "end": 0.85, "speaker": "SPEAKER_00"},
        ]
        pslicer.apply_contextual_speaker_smoothing(
            words,
            half_window_sec=0.3,
            max_island_words=2,
            max_island_sec=0.2,
            preserve_raw=False,
        )
        self.assertEqual(pslicer._speaker_label(words[1]), "SPEAKER_00")

    def test_island_collapse_sandwiched_run(self):
        """A...B...A with tiny B run — B merged into A."""
        words = [
            {"word": "a1", "start": 0.0, "end": 0.1, "speaker": "SPEAKER_A"},
            {"word": "b1", "start": 0.12, "end": 0.18, "speaker": "SPEAKER_B"},
            {"word": "a2.", "start": 0.2, "end": 0.35, "speaker": "SPEAKER_A"},
        ]
        pslicer.apply_contextual_speaker_smoothing(
            words,
            half_window_sec=0.05,  # weak vote; rely on island pass
            max_island_words=2,
            max_island_sec=0.2,
            preserve_raw=False,
        )
        self.assertEqual(pslicer._speaker_label(words[1]), "SPEAKER_A")


class TestToneFixtureSeparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix_dir = _ROOT / "test_fixtures" / "pslicer_tones"
        cls.wav_path = cls.fix_dir / "two_speaker_tones.wav"
        build_two_speaker_tone_fixture(cls.wav_path)

    def test_exported_clips_match_expected_speakers_and_tones(self):
        """
        Synthetic words follow the fixture layout:
        - ~0–1.6s: SPEAKER_00 (440 Hz burst) — one sentence
        - ~1.85–3.45s: SPEAKER_01 (554 Hz) — second sentence
        Inject one mis-tagged word in the second region; smoothing should keep S01 dominant.
        """
        words = [
            {"word": "First", "start": 0.05, "end": 0.35, "speaker": "SPEAKER_00"},
            {"word": "phrase.", "start": 0.4, "end": 1.2, "speaker": "SPEAKER_00"},
            {"word": "Second", "start": 1.9, "end": 2.25, "speaker": "SPEAKER_01"},
            {"word": "noise", "start": 2.28, "end": 2.42, "speaker": "SPEAKER_00"},  # intentional glitch
            {"word": "phrase.", "start": 2.45, "end": 3.2, "speaker": "SPEAKER_01"},
        ]
        pslicer.apply_contextual_speaker_smoothing(
            words,
            half_window_sec=0.5,
            max_island_words=2,
            max_island_sec=0.25,
            preserve_raw=True,
        )
        self.assertEqual(pslicer._speaker_label(words[3]), "SPEAKER_01")

        chunks = pslicer.iter_sentence_speaker_chunks(words, min_duration=0.05)
        self.assertGreaterEqual(
            len(chunks), 2, "expected at least two sentence-level chunks for two speakers"
        )

        out = tempfile.mkdtemp(prefix="pslicer_test_")
        try:
            paths = pslicer.export_wav_clips(
                str(self.wav_path),
                out,
                chunks,
                stem="fixture",
                padding_ms=80,
                verbose=False,
            )
            self.assertGreaterEqual(len(paths), 2, paths)

            # Map exported files to dominant frequency; first chunk should be low, second high
            peaks = [_dominant_peak_hz(p) for p in paths[:2]]
            self.assertLess(peaks[0], 480, f"expected ~440 Hz in first clip, got {peaks[0]}")
            self.assertGreater(peaks[1], 500, f"expected ~554 Hz in second clip, got {peaks[1]}")

            # Filenames should encode distinct speakers for the two sentences
            base0, base1 = os.path.basename(paths[0]), os.path.basename(paths[1])
            self.assertIn("SPEAKER_00", base0)
            self.assertIn("SPEAKER_01", base1)
        finally:
            shutil.rmtree(out, ignore_errors=True)


class _MockVoiceCmp:
    """Minimal stand-in for ChunkVoiceComparator in pause-merge tests."""

    def __init__(self, cos: float) -> None:
        self._cos = float(cos)

    def tail_head_cosine(self, *args: object, **kwargs: object) -> float:
        return self._cos

    def pause_merge_voice_score(self, *args: object, **kwargs: object) -> float:
        return self._cos


class TestExportOneWavClip(unittest.TestCase):
    def test_writes_padded_slice(self) -> None:
        wav = _ROOT / "test_fixtures" / "pslicer_tones" / "two_speaker_tones.wav"
        if not wav.is_file():
            build_two_speaker_tone_fixture(wav)
        out = tempfile.mkstemp(suffix=".wav", prefix="pslicer_one_")
        os.close(out[0])
        try:
            pslicer.export_one_wav_clip(str(wav), out[1], 0.05, 0.55, padding_ms=40)
            self.assertTrue(os.path.isfile(out[1]))
            self.assertGreater(os.path.getsize(out[1]), 800)
        finally:
            try:
                os.remove(out[1])
            except OSError:
                pass


class TestTranscriptForTimeRange(unittest.TestCase):
    def test_overlapping_words_joined_in_order(self):
        words = [
            {"word": "Hello", "start": 0.05, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "world.", "start": 0.5, "end": 0.95, "speaker": "SPEAKER_00"},
        ]
        self.assertEqual(pslicer.transcript_for_time_range(words, 0.0, 1.0), "Hello world.")
        self.assertEqual(pslicer.transcript_for_time_range(words, 0.45, 1.0), "world.")
        self.assertEqual(pslicer.transcript_for_time_range(words, 0.05, 0.42), "Hello")

    def test_empty_when_no_overlap(self):
        words = [{"word": "Hi", "start": 1.0, "end": 1.2, "speaker": "SPEAKER_00"}]
        self.assertEqual(pslicer.transcript_for_time_range(words, 0.0, 0.5), "")


class TestSmartPauseMerge(unittest.TestCase):
    def test_same_speaker_period_then_capital_merges_within_text_window_without_voice(self):
        chunks = [
            (0.0, 1.0, "SPEAKER_00", "Hello."),
            (1.5, 2.8, "SPEAKER_00", "World today."),
        ]
        out = pslicer.merge_chunks_smart_same_speaker_pauses(
            chunks,
            voice_compare=None,
            breath_gap_sec=0.42,
            period_glue_if_lowercase_sec=1.25,
            max_glue_gap_sec=2.5,
        )
        self.assertEqual(len(out), 1)
        self.assertIn("Hello.", out[0][3])
        self.assertIn("World", out[0][3])

    def test_same_speaker_period_capital_not_merged_past_text_only_cap_without_voice(self):
        chunks = [
            (0.0, 1.0, "SPEAKER_00", "Hello."),
            (2.4, 3.5, "SPEAKER_00", "World today."),
        ]
        out = pslicer.merge_chunks_smart_same_speaker_pauses(
            chunks,
            voice_compare=None,
            breath_gap_sec=0.42,
            period_glue_if_lowercase_sec=1.25,
            max_glue_gap_sec=2.5,
        )
        self.assertEqual(len(out), 2)

    def test_same_speaker_longer_pause_merges_when_voice_scores_high(self):
        chunks = [
            (0.0, 1.0, "SPEAKER_00", "Hello."),
            (2.4, 3.5, "SPEAKER_00", "World today."),
        ]
        out = pslicer.merge_chunks_smart_same_speaker_pauses(
            chunks,
            voice_compare=_MockVoiceCmp(0.92),
            voice_min_cosine=0.55,
            voice_veto_cosine=0.38,
            breath_gap_sec=0.42,
            period_glue_if_lowercase_sec=1.25,
            max_glue_gap_sec=2.5,
        )
        self.assertEqual(len(out), 1)

    def test_same_speaker_long_pause_blocked_when_voice_vetoes(self):
        chunks = [
            (0.0, 1.0, "SPEAKER_00", "Hello."),
            (2.4, 3.5, "SPEAKER_00", "World today."),
        ]
        out = pslicer.merge_chunks_smart_same_speaker_pauses(
            chunks,
            voice_compare=_MockVoiceCmp(0.15),
            voice_min_cosine=0.55,
            voice_veto_cosine=0.38,
            breath_gap_sec=0.42,
            period_glue_if_lowercase_sec=1.25,
            max_glue_gap_sec=2.5,
        )
        self.assertEqual(len(out), 2)


class TestChronologicalExportNames(unittest.TestCase):
    """Output WAV names lead with ``tXXXXXXXXX`` (start ms) so lexicographic sort = timeline order."""

    def test_sort_token_monotonic_with_time(self):
        times = [i * 0.017 + 0.001 for i in range(500)]
        toks = [pslicer._chunk_start_sort_token_and_ms(t)[0] for t in times]
        self.assertEqual(toks, sorted(toks))

    def test_sort_token_stress_random_increments(self):
        rng = np.random.default_rng(42)
        t = 0.0
        prev_tok = ""
        for _ in range(3000):
            t += float(rng.uniform(0.002, 0.08))
            tok, ms = pslicer._chunk_start_sort_token_and_ms(t)
            self.assertGreater(tok, prev_tok, (t, tok, prev_tok, ms))
            prev_tok = tok

    def test_export_paths_list_sorted_equals_timeline(self):
        words = [
            {"word": "A.", "start": 0.1, "end": 0.3, "speaker": "SPEAKER_00"},
            {"word": "B.", "start": 2.5, "end": 2.9, "speaker": "SPEAKER_00"},
        ]
        chunks = pslicer.iter_sentence_speaker_chunks(words, min_duration=0.05)
        out = tempfile.mkdtemp(prefix="pslicer_chrono_")
        try:
            wav = _ROOT / "test_fixtures" / "pslicer_tones" / "two_speaker_tones.wav"
            if not wav.is_file():
                build_two_speaker_tone_fixture(wav)
            paths = pslicer.export_wav_clips(
                str(wav),
                out,
                chunks,
                stem="chrono",
                padding_ms=0,
                verbose=False,
            )
            names = [os.path.basename(p) for p in paths]
            self.assertEqual(names, sorted(names))
            self.assertTrue(all("_t" in n and "_auto_" in n for n in names))
            self.assertLess(names[0], names[1])
        finally:
            shutil.rmtree(out, ignore_errors=True)


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestContextualSmoothing))
    suite.addTests(loader.loadTestsFromTestCase(TestToneFixtureSeparation))
    suite.addTests(loader.loadTestsFromTestCase(TestExportOneWavClip))
    suite.addTests(loader.loadTestsFromTestCase(TestTranscriptForTimeRange))
    suite.addTests(loader.loadTestsFromTestCase(TestSmartPauseMerge))
    suite.addTests(loader.loadTestsFromTestCase(TestChronologicalExportNames))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
