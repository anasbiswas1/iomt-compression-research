"""
iomtc_metrics.py  -  the stable, reused metric functions for the trust layer.

These are imported by every analysis notebook so the methodology is defined
once, not re-implemented per stage (the plan's single-shared-function rule).
Only foundational, stable functions live here. Stage-specific logic (the
four-way partition loader, the shortcut probes) arrives in its own modules
during Phase 1, keyed to the metadata that the inventory step confirms exists.

Conventions:
  - ECE primary estimator is ADAPTIVE (equal-mass) binning; fixed-bin reported
    only for literature comparability (plan 7.1).
  - Bootstrap is percentile, B from iomtc_config.C.BOOTSTRAP_B by default.
  - Everything returns plain floats / numpy arrays; no global state.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Calibration error
# ---------------------------------------------------------------------------
def ece_fixed(y_true, p_pred, n_bins: int = 15):
    """Fixed-width-bin ECE (the classic, biased estimator). For comparability only."""
    y_true = np.asarray(y_true).astype(float)
    p_pred = np.asarray(p_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_pred, bins[1:-1], right=False)
    ece = 0.0
    n = len(p_pred)
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        conf = p_pred[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def ece_adaptive(y_true, p_pred, n_bins: int = 15):
    """
    Equal-mass (adaptive) binning ECE  -  the primary calibration metric.
    Each bin holds ~equal sample count, which removes the empty/over-full bin
    bias that reviewers flag in fixed-bin ECE.
    """
    y_true = np.asarray(y_true).astype(float)
    p_pred = np.asarray(p_pred).astype(float)
    n = len(p_pred)
    if n == 0:
        return float("nan")
    order = np.argsort(p_pred)
    p_sorted = p_pred[order]
    y_sorted = y_true[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        conf = p_sorted[lo:hi].mean()
        acc = y_sorted[lo:hi].mean()
        ece += ((hi - lo) / n) * abs(acc - conf)
    return float(ece)


def brier(y_true, p_pred):
    """Brier score (a proper scoring rule). Lower is better."""
    y_true = np.asarray(y_true).astype(float)
    p_pred = np.asarray(p_pred).astype(float)
    return float(np.mean((p_pred - y_true) ** 2))


def reliability_curve(y_true, p_pred, n_bins: int = 15, adaptive: bool = True):
    """
    Return (bin_conf, bin_acc, bin_count) for a reliability diagram.
    adaptive=True uses equal-mass bins (matches ece_adaptive).
    """
    y_true = np.asarray(y_true).astype(float)
    p_pred = np.asarray(p_pred).astype(float)
    n = len(p_pred)
    confs, accs, counts = [], [], []
    if adaptive:
        order = np.argsort(p_pred)
        p_sorted, y_sorted = p_pred[order], y_true[order]
        edges = np.linspace(0, n, n_bins + 1).astype(int)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if hi <= lo:
                continue
            confs.append(p_sorted[lo:hi].mean())
            accs.append(y_sorted[lo:hi].mean())
            counts.append(hi - lo)
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(p_pred, bins[1:-1], right=False)
        for b in range(n_bins):
            m = idx == b
            if not np.any(m):
                continue
            confs.append(p_pred[m].mean())
            accs.append(y_true[m].mean())
            counts.append(int(m.sum()))
    return np.array(confs), np.array(accs), np.array(counts)


# ---------------------------------------------------------------------------
# Uncertainty intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(values, statistic=np.mean, B: int = 1000, alpha: float = 0.05,
                 seed: int = 42):
    """
    Percentile bootstrap CI for a 1-D array statistic.
    Returns (point, lo, hi). Use for quartile gaps, correlations passed as arrays, etc.
    """
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(values))
    boot = np.empty(B)
    for b in range(B):
        sample = values[rng.integers(0, n, n)]
        boot[b] = statistic(sample)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


def paired_bootstrap_diff(a, b, statistic=np.mean, B: int = 1000, alpha: float = 0.05,
                          seed: int = 42):
    """
    Paired bootstrap for the difference statistic(a) - statistic(b) on paired arrays
    (same indices resampled for both). Returns (diff, lo, hi). Use for compression
    deltas and protocol deltas where samples are paired by instance/class.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert len(a) == len(b), "paired arrays must match length"
    rng = np.random.default_rng(seed)
    n = len(a)
    diff = float(statistic(a) - statistic(b))
    boot = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boot[i] = statistic(a[idx]) - statistic(b[idx])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return diff, lo, hi


def wilson_interval(successes: int, total: int, z: float = 1.96):
    """
    Wilson score interval for a binomial proportion. Use for per-class detection
    rates on small classes (better than normal approx at low n).
    Returns (phat, lo, hi).
    """
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    phat = successes / total
    denom = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))) / denom
    return float(phat), float(centre - half), float(centre + half)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 2000), 0, 1)
    print("ECE adaptive:", round(ece_adaptive(y, p), 4))
    print("ECE fixed   :", round(ece_fixed(y, p), 4))
    print("Brier       :", round(brier(y, p), 4))
    print("Wilson(8,10):", tuple(round(x, 3) for x in wilson_interval(8, 10)))
