"""
Standard depth metrics + the proposed UDAC metric.

Reference metrics follow Eigen et al. NeurIPS 2014 conventions:
  - AbsRel, SqRel, RMSE, RMSE-log, SILog, delta_1/2/3

UDAC (Underwater Depth-Attenuation Consistency) is the Pearson correlation
between the per-pixel near-ness signal v(x) = R/(R+G+B+eps) and a min-max
normalised inverted prediction n_hat. Scale-invariant by construction;
requires no ground-truth depth.
"""

import numpy as np


# Small epsilons to keep things sane
EPS = 1e-6


# -------------------------------------------------------------------
# Standard reference-based metrics
# -------------------------------------------------------------------

def reference_metrics(pred, gt, mask=None):
    """Compute the standard depth-estimation metric suite on one image.

    Args:
        pred: H x W float, predicted depth in metres (already aligned).
        gt:   H x W float, ground-truth depth in metres.
        mask: H x W bool, optional. True where the pixel is valid.

    Returns:
        dict of metric_name -> float
    """
    if mask is None:
        mask = np.isfinite(gt) & (gt > 0)
    mask = mask & np.isfinite(pred) & (pred > 0)
    if mask.sum() < 50:
        # Not enough valid pixels -- return NaNs so the aggregator can drop it
        return {k: float('nan') for k in
                ['absrel', 'sqrel', 'rmse', 'rmse_log',
                 'd1', 'd2', 'd3', 'silog']}

    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    # absrel & sqrel
    diff = np.abs(g - p)
    absrel = np.mean(diff / g)
    sqrel = np.mean((g - p) ** 2 / g)

    # rmse + rmse_log
    rmse = np.sqrt(np.mean((g - p) ** 2))
    log_diff = np.log(g + EPS) - np.log(p + EPS)
    rmse_log = np.sqrt(np.mean(log_diff ** 2))

    # threshold accuracies
    ratio = np.maximum(g / p, p / g)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))

    # SILog (scale-invariant log error -- Eigen et al.)
    silog = np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2) * 100

    return {
        'absrel':   float(absrel),
        'sqrel':    float(sqrel),
        'rmse':     float(rmse),
        'rmse_log': float(rmse_log),
        'd1':       d1,
        'd2':       d2,
        'd3':       d3,
        'silog':    float(silog),
    }


# -------------------------------------------------------------------
# UDAC (Underwater Depth-Attenuation Consistency)
# -------------------------------------------------------------------

def near_ness_signal(rgb_image):
    """Per-pixel near-ness from the RGB image.

    v(x) = R(x) / (R(x) + G(x) + B(x) + eps)

    Higher v corresponds to redder pixels, which (by the underwater
    image-formation model) are physically nearer in expectation.

    Note: an earlier version of this used a thresholded "red dominance"
    test (1 if R > G and R > B, else 0). That signal saturates to ~0
    on most FLSea images where the red channel is already smaller than
    G+B everywhere. The fractional form below preserves dynamic range
    on those images.
    """
    R = rgb_image[..., 0]
    G = rgb_image[..., 1]
    B = rgb_image[..., 2]
    return R / (R + G + B + EPS)


def normalised_near_ness(pred):
    """n_hat(x) = (z_max - z(x)) / (z_max - z_min); large for near pixels."""
    p = pred.astype(np.float64)
    mask = np.isfinite(p)
    if mask.sum() < 50:
        return np.zeros_like(p)
    z_min = np.nanmin(p)
    z_max = np.nanmax(p)
    if (z_max - z_min) < EPS:
        return np.zeros_like(p)
    return (z_max - p) / (z_max - z_min)


def udac(rgb_image, pred):
    """Compute UDAC for one image.

    Returns the Pearson correlation between v(x) and n_hat(x) over
    pixels where both are well-defined. The aggregator over the dataset
    is the simple mean (see aggregate_udac).

    Returns:
        float in [-1, 1], or NaN if the correlation is undefined
        (one of the signals has zero variance).
    """
    v = near_ness_signal(rgb_image)
    n = normalised_near_ness(pred)

    mask = np.isfinite(v) & np.isfinite(n)
    if mask.sum() < 50:
        return float('nan')

    v = v[mask].flatten().astype(np.float64)
    n = n[mask].flatten().astype(np.float64)

    if v.std() < EPS or n.std() < EPS:
        return float('nan')

    return float(np.corrcoef(v, n)[0, 1])


# -------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------

def aggregate(per_image_results):
    """Mean over per-image metric dicts, ignoring NaNs.

    Also reports the standard deviation per metric (helpful for noting
    method variability across the dataset).
    """
    if not per_image_results:
        return {}

    keys = sorted(per_image_results[0].keys())
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in per_image_results
                         if k in r and np.isfinite(r[k])],
                        dtype=np.float64)
        if vals.size == 0:
            out[f'{k}_mean'] = float('nan')
            out[f'{k}_std']  = float('nan')
            out[f'{k}_n']    = 0
        else:
            out[f'{k}_mean'] = float(vals.mean())
            out[f'{k}_std']  = float(vals.std())
            out[f'{k}_n']    = int(vals.size)
    return out
