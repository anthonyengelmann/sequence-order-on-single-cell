import numpy as np
import pytest

from lyra_lite.analysis.ece import expected_calibration_error


def test_perfectly_calibrated_is_zero():
    # 100 samples all predicted 0.5; exactly half are positive.
    # In that one bin: confidence = 0.5, accuracy = 0.5 -> |diff| = 0.
    y_score = np.full(100, 0.5)
    y_true = np.array([1, 0] * 50)
    assert expected_calibration_error(y_true, y_score, n_bins=10) == pytest.approx(0.0)


def test_maximally_miscalibrated_is_one():
    # Model screams 1.0 with full confidence, but every label is 0.
    # confidence = 1.0, accuracy = 0.0, bin weight = 1 -> ECE = 1.0.
    y_score = np.ones(50)
    y_true = np.zeros(50, dtype=int)
    assert expected_calibration_error(y_true, y_score, n_bins=10) == pytest.approx(1.0)


def test_hand_computed_example():
    # n_bins=2 -> edges [0, 0.5, 1.0].
    # bin (0, 0.5]: scores 0.2,0.3 -> conf 0.25, acc 1/2 -> contrib (2/4)*0.25 = 0.125
    # bin (0.5, 1.0]: scores 0.8,0.9 -> conf 0.85, acc 2/2 -> contrib (2/4)*0.15 = 0.075
    # total = 0.2
    y_score = np.array([0.2, 0.3, 0.8, 0.9])
    y_true = np.array([0, 1, 1, 1])
    assert expected_calibration_error(y_true, y_score, n_bins=2) == pytest.approx(0.2)


def test_output_is_in_unit_interval():
    rng = np.random.default_rng(0)
    y_score = rng.random(500)
    y_true = rng.integers(0, 2, size=500)
    ece = expected_calibration_error(y_true, y_score, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_score_of_exactly_one_is_not_dropped():
    # If 1.0 fell into no bin, the count wouldn't sum to n and this would skew.
    y_score = np.array([0.0, 1.0])
    y_true = np.array([0, 1])
    # both perfectly calibrated extremes -> ECE 0
    assert expected_calibration_error(y_true, y_score, n_bins=10) == pytest.approx(0.0)