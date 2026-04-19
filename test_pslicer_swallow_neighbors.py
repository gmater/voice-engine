"""
Unit tests for AI trim swallow: any chunk fully inside the edited range is removed.

Run:
  venv\\Scripts\\python.exe test_pslicer_swallow_neighbors.py
"""

from __future__ import annotations

import gc
import random
import sys
import tracemalloc
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import slicer  # noqa: E402


class TestChunkSwallow(unittest.TestCase):
    def test_empty_and_bad_si(self) -> None:
        self.assertEqual(slicer._pslicer_chunk_indices_fully_swallowed([], 0, 0.0, 10.0), [])
        chunks = [(0.0, 1.0, "A", "a")]
        self.assertEqual(slicer._pslicer_chunk_indices_fully_swallowed(chunks, -1, 0.0, 10.0), [])
        self.assertEqual(slicer._pslicer_chunk_indices_fully_swallowed(chunks, 5, 0.0, 10.0), [])

    def test_swallows_all_other_chunks_fully_inside(self) -> None:
        chunks = [
            (0.0, 1.0, "A", "a"),
            (1.0, 2.0, "B", "b"),
            (2.0, 3.0, "C", "c"),
        ]
        got = sorted(slicer._pslicer_chunk_indices_fully_swallowed(chunks, 1, 0.0, 3.0))
        self.assertEqual(got, [0, 2])

    def test_swallows_non_adjacent_in_timeline(self) -> None:
        """Expanding the middle clip can remove clips that are not immediate neighbors in order."""
        chunks = [
            (0.0, 1.0, "A", "a"),
            (1.0, 2.0, "B", "b"),
            (2.0, 3.0, "C", "c"),
            (3.0, 4.0, "D", "d"),
        ]
        got = sorted(slicer._pslicer_chunk_indices_fully_swallowed(chunks, 1, 0.0, 4.0))
        self.assertEqual(got, [0, 2, 3])

    def test_no_swallow_when_only_partial_overlap(self) -> None:
        chunks = [
            (0.0, 1.0, "A", "a"),
            (1.0, 2.0, "B", "b"),
            (2.0, 3.0, "C", "c"),
        ]
        self.assertEqual(slicer._pslicer_chunk_indices_fully_swallowed(chunks, 1, 1.1, 1.9), [])

    def test_single_other_chunk(self) -> None:
        chunks = [
            (0.0, 1.0, "A", "a"),
            (1.0, 2.0, "B", "b"),
        ]
        self.assertEqual(slicer._pslicer_chunk_indices_fully_swallowed(chunks, 0, 0.0, 2.0), [1])

    def test_subset_detection_eps(self) -> None:
        inner = (0.10001, 0.19999, "X", "x")
        self.assertTrue(slicer._pslicer_clip_interval_subset(inner, 0.1, 0.2, eps=1e-3))


class TestChunkSwallowTracemalloc(unittest.TestCase):
    def test_repeated_swallow_query_bounded_growth(self) -> None:
        rng = random.Random(42)
        chunks = [(float(i), float(i) + 0.5, "S", "x") for i in range(40)]
        tracemalloc.stop()
        tracemalloc.clear_traces()
        tracemalloc.start(25)
        for _ in range(500):
            si = rng.randrange(0, len(chunks))
            t0 = rng.uniform(-0.2, 2.0)
            t1 = t0 + rng.uniform(0.5, 4.0)
            slicer._pslicer_chunk_indices_fully_swallowed(chunks, si, t0, t1)
        gc.collect()
        s1 = tracemalloc.take_snapshot()
        for _ in range(8000):
            si = rng.randrange(0, len(chunks))
            t0 = rng.uniform(-0.2, 2.0)
            t1 = t0 + rng.uniform(0.5, 4.0)
            slicer._pslicer_chunk_indices_fully_swallowed(chunks, si, t0, t1)
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


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestChunkSwallow))
    suite.addTests(loader.loadTestsFromTestCase(TestChunkSwallowTracemalloc))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
