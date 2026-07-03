"""Fast CPU-only tests for the stats + config layers (no GPU / model needed)."""
import math

from sparse_actions.stats import (
    beta_posterior,
    calibration_report,
    clopper_pearson,
    log10_abs_error,
    required_n,
)


def test_required_n_scales_inverse_with_p():
    assert required_n(1e-3) > required_n(1e-2) > required_n(1e-1)
    # ballpark: ~10x more samples per decade rarer
    assert 5 < required_n(1e-3) / required_n(1e-2) < 20


def test_clopper_pearson_bounds():
    lo, hi = clopper_pearson(10, 1000)
    assert lo < 0.01 < hi
    assert clopper_pearson(0, 100)[0] == 0.0
    assert clopper_pearson(100, 100)[1] == 1.0


def test_beta_posterior_mean():
    mean, lo, hi = beta_posterior(5, 1000)
    assert lo < mean < hi
    assert abs(mean - 5 / 1000) < 5e-3


def test_log10_abs_error_zero_when_equal():
    assert log10_abs_error(1e-3, 1e-3) < 1e-9
    assert abs(log10_abs_error(2e-3, 1e-3) - math.log10(2)) < 1e-6


def test_calibration_report_perfect():
    targets = [1e-1, 1e-2, 1e-3]
    rep = calibration_report(targets, targets)
    assert rep["mean_log10_abs_error"] < 1e-6
    assert abs(rep["log_log_slope"] - 1.0) < 1e-6
