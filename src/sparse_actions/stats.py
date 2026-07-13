"""Statistics for rare-rate estimation and calibration scoring.

Estimating p ~ 1e-3 by sampling needs N*p successes to be non-tiny. Use
`required_n` to size runs, and Clopper-Pearson / Beta intervals to report honest
uncertainty. Calibration is scored in log10 space (the natural space for rates).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def required_n(p: float, rel_err: float = 0.2, z: float = 1.96) -> int:
    """Samples needed so the relative std error of a Binomial rate estimate ~ rel_err.

    SE/p = sqrt((1-p)/(N p)) => N = (1-p) / (p * rel_err^2) * (z^2 handled loosely).
    We include z so `rel_err` is read as a (z-)confidence half-width fraction.
    """
    p = max(p, 1e-12)
    return int(math.ceil((z ** 2) * (1.0 - p) / (p * rel_err ** 2)))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (conservative) binomial CI for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else sps.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else sps.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for k successes in n trials (the interval used by Serrano et al. 2026)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def beta_posterior(k: int, n: int, alpha: float = 0.05, a0: float = 0.5, b0: float = 0.5):
    """Jeffreys-prior Beta posterior for the rate. Returns (mean, lo, hi)."""
    a, b = a0 + k, b0 + (n - k)
    mean = a / (a + b)
    lo = sps.beta.ppf(alpha / 2, a, b)
    hi = sps.beta.ppf(1 - alpha / 2, a, b)
    return float(mean), float(lo), float(hi)


def log10_abs_error(realized: float, target: float, floor: float = 1e-9) -> float:
    """|log10(realized) - log10(target)|. A value of ~0.1 == within ~26% of target."""
    return abs(math.log10(max(realized, floor)) - math.log10(max(target, floor)))


def calibration_report(targets, realized) -> dict:
    """Aggregate calibration metrics over a swept grid (log10 space)."""
    t = np.asarray(targets, dtype=float)
    r = np.clip(np.asarray(realized, dtype=float), 1e-12, 1.0)
    log_err = np.abs(np.log10(r) - np.log10(t))
    # Slope of realized-vs-target line in log space (1.0 == perfectly proportional).
    lt, lr = np.log10(t), np.log10(r)
    slope = float(np.polyfit(lt, lr, 1)[0]) if len(t) > 1 else float("nan")
    return {
        "mean_log10_abs_error": float(log_err.mean()),
        "max_log10_abs_error": float(log_err.max()),
        "log_log_slope": slope,   # want ~1.0
        "n_points": int(len(t)),
    }
