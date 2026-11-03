"""
Calibrate beta_eff for the classical methods (UDCP, PC).

UDCP and Peng-Cosman both produce transmission-based depths of the form
    z = -log(t) / beta_eff
with `beta_eff` unknown. We fit it once per dataset on a held-out 10%
calibration subset by minimising mean-squared error against GT (after the
same LS scale-shift alignment used at evaluation time).

In practice the LS alignment soaks up scale errors anyway, so `beta_eff`
mostly affects the dynamic range of the depth map. We still calibrate it
for completeness and to make the per-image alignment less prone to numerical
issues on extreme images.

Usage:
    from methods._classical_calibrate import fit_beta_eff
    beta = fit_beta_eff('udcp', 'flsea')
"""

import os
import json
import numpy as np

from align import fit_scale_shift, apply_alignment
from datasets import get_dataset


CALIB_FRAC = 0.10           # use 10% of the dataset
CALIB_CACHE = 'outputs/calibration'


def _iter_calibration_set(dataset_name):
    """Take every 10th image (deterministic, no RNG needed)."""
    for i, item in enumerate(get_dataset(dataset_name)):
        if i % int(1.0 / CALIB_FRAC) == 0:
            yield item


def fit_beta_eff(method_key, dataset_name,
                 search_grid=None, cache=True):
    """Search over beta_eff and pick the value giving lowest mean AbsRel.

    Args:
        method_key:   'udcp' or 'pc'.
        dataset_name: 'flsea' or 'squid'.
        search_grid:  list of beta values to try (default: log-spaced).
        cache:        write/read result from CALIB_CACHE.
    """
    if search_grid is None:
        # geometric grid covering plausible Jerlov 1--3 effective rates
        search_grid = np.geomspace(0.05, 5.0, 25)

    cache_path = os.path.join(CALIB_CACHE,
                              f'{method_key}_{dataset_name}_beta.json')
    if cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)['beta_eff']

    from methods import build
    print(f'calibrating beta_eff for {method_key} on {dataset_name} '
          f'({len(search_grid)} grid points)...')

    best_beta, best_err = None, float('inf')
    for beta in search_grid:
        pred = build(method_key, beta_eff=float(beta))
        errs = []
        for image_id, rgb, gt in _iter_calibration_set(dataset_name):
            try:
                p = pred.predict(rgb)
                s, t = fit_scale_shift(p, gt)
                p_aligned = apply_alignment(p, s, t)
                mask = np.isfinite(gt) & (gt > 0) & np.isfinite(p_aligned)
                if mask.sum() < 50:
                    continue
                e = np.mean(np.abs(p_aligned[mask] - gt[mask]) / gt[mask])
                errs.append(e)
            except Exception as ex:
                # bad image; skip
                print(f'  warn: {image_id} -> {ex}')
                continue

        if errs:
            mean_err = float(np.mean(errs))
            if mean_err < best_err:
                best_err = mean_err
                best_beta = float(beta)
            print(f'  beta={beta:.3f}  absrel={mean_err:.4f}')

    print(f'best beta_eff = {best_beta:.4f}  (AbsRel = {best_err:.4f})')

    if cache:
        os.makedirs(CALIB_CACHE, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'beta_eff': best_beta, 'absrel': best_err}, f, indent=2)

    return best_beta


if __name__ == '__main__':
    # Run with: python -m methods._classical_calibrate udcp flsea
    import sys
    fit_beta_eff(sys.argv[1], sys.argv[2])
