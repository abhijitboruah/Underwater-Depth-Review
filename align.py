"""
Per-image least-squares scale-shift alignment.

All methods except UniDepth V2 produce relative depth (inverse depth,
scale-and-shift-invariant disparity, normalised depth). To compute
reference-based metrics like AbsRel and RMSE, we fit (s, t) per image
that minimise || s * pred + t - gt ||_2 over valid GT pixels.

This is the convention adopted across the contemporary depth literature
(e.g. MiDaS Ranftl 2020, DPT 2021, Depth Anything 2024).
"""

import numpy as np


def fit_scale_shift(pred, gt, mask=None):
    """Closed-form LS estimate of (s, t) for pred -> gt.

    Solves:
        min_{s, t}  sum_x [ s * pred(x) + t - gt(x) ]^2

    over pixels where mask is True (default: GT > 0 and finite, pred finite).

    Returns (s, t). If the system is degenerate (constant pred), returns
    (0, mean(gt)) so the aligned prediction is just the GT mean.
    """
    if mask is None:
        mask = np.isfinite(gt) & (gt > 0)
    mask = mask & np.isfinite(pred)
    if mask.sum() < 50:
        return 0.0, float(np.nanmean(gt)) if np.any(np.isfinite(gt)) else 0.0

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    p_mean = p.mean()
    g_mean = g.mean()
    p_centered = p - p_mean
    denom = (p_centered ** 2).sum()
    if denom < 1e-12:
        return 0.0, float(g_mean)

    s = float((p_centered * (g - g_mean)).sum() / denom)
    t = float(g_mean - s * p_mean)
    return s, t


def apply_alignment(pred, s, t):
    return s * pred + t
