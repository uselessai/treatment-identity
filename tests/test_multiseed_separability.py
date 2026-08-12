"""Matched-seed and distributional treatment-separability checks."""

from __future__ import annotations

import unittest

import numpy as np

from treatment_identity import check_separability


SEEDS = tuple(range(16))


def _draw(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(100.0, 10.0, (8, 8, 3))


class MultiseedSeparabilityTests(unittest.TestCase):
    def test_exact_equality_fails_before_statistical_inference(self) -> None:
        result = check_separability(
            _draw, _draw, seeds=SEEDS, distributional=True)
        self.assertEqual(result.status, "FAIL", result.message)
        self.assertIn("UNEXPECTED EQUALITY", result.message)
        self.assertNotIn("energy_test", result.evidence)

    def test_reproducible_distribution_shift_passes_declared_threshold(self) -> None:
        result = check_separability(
            _draw, lambda seed: _draw(seed) + 20.0,
            seeds=SEEDS, distributional=True, permutations=4095,
            permutation_seed=19)
        self.assertEqual(result.status, "PASS", result.message)
        self.assertLessEqual(result.evidence["energy_test"]["p_value"], 0.05)
        self.assertGreater(result.evidence["energy_test"]["statistic_rms_0_255"], 0)

    def test_visible_draw_differences_do_not_replace_distributional_evidence(self) -> None:
        result = check_separability(
            _draw,
            lambda seed: np.random.default_rng(seed + 10_000).normal(
                100.0, 10.0, (8, 8, 3)),
            seeds=SEEDS, distributional=True, permutations=255,
            permutation_seed=31)
        self.assertEqual(result.status, "FAIL", result.message)
        self.assertGreater(result.evidence["draws_distinguishable"], 0)
        self.assertGreater(result.evidence["energy_test"]["p_value"], 0.05)
        self.assertIn("not proof", result.message)

    def test_single_draw_mode_remains_backward_compatible(self) -> None:
        result = check_separability(
            lambda: np.zeros((2, 2, 3)),
            lambda: np.ones((2, 2, 3)))
        self.assertEqual(result.status, "PASS", result.message)
        self.assertEqual(result.evidence["seeds"], [0])

    def test_nonfinite_treatment_fails_explicitly(self) -> None:
        result = check_separability(
            _draw, lambda seed: np.full((8, 8, 3), np.nan),
            seeds=SEEDS, distributional=True)
        self.assertEqual(result.status, "FAIL", result.message)
        self.assertIn("NaN or infinity", result.message)

    def test_variable_seed_shapes_fail_without_crashing(self) -> None:
        def variable(seed: int) -> np.ndarray:
            return np.zeros((2 + seed % 2, 2, 3))

        result = check_separability(
            variable, lambda seed: variable(seed) + 1,
            seeds=SEEDS, distributional=True)
        self.assertEqual(result.status, "FAIL", result.message)
        self.assertIn("varies across seeds", result.message)


if __name__ == "__main__":
    unittest.main()
