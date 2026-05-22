"""
Main evaluation driver.

Example invocations:
    python run.py --method udcp --dataset flsea
    python run.py --method unidepth --dataset squid --save_preds
    python run.py --method dav2 --dataset flsea \
        --weights weights/depth_anything_v2_vitl.pth

Writes:
    outputs/{dataset}/{method}/metrics.json   -- aggregated metrics
    outputs/{dataset}/{method}/per_image.csv  -- per-image rows
  [outputs/{dataset}/{method}/preds/*.npy     -- if --save_preds]
"""

import os
import json
import argparse
import time

import numpy as np
from tqdm import tqdm

from datasets import get_dataset
from metrics import reference_metrics, udac, aggregate
from align import fit_scale_shift, apply_alignment
import methods


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_method(method_key, args):
    """Construct the predictor with method-specific kwargs from CLI."""
    kw = dict(device=args.device)

    if method_key in ('udcp', 'pc'):
        if args.beta_eff is not None:
            kw['beta_eff'] = args.beta_eff
        else:
            # auto-calibrate per dataset (cached)
            from methods._classical_calibrate import fit_beta_eff
            kw['beta_eff'] = fit_beta_eff(method_key, args.dataset)

    elif method_key == 'md2':
        if args.weights is None:
            raise ValueError('--weights is required for Monodepth2 '
                             '(directory containing encoder.pth, depth.pth)')
        kw['weights_dir'] = args.weights
        if args.resnet:
            kw['resnet'] = args.resnet

    elif method_key == 'udepth':
        if args.weights is None:
            raise ValueError('--weights is required for UDepth '
                             '(path to the pretrained .pth file)')
        kw['weights_path'] = args.weights

    elif method_key == 'dav2':
        if args.weights is None:
            raise ValueError('--weights is required for Depth Anything V2 '
                             '(path to depth_anything_v2_vitl.pth)')
        kw['weights_path'] = args.weights

    elif method_key == 'unidepth':
        if args.hf_id:
            kw['hf_id'] = args.hf_id

    return methods.build(method_key, **kw)


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=methods.available())
    ap.add_argument('--dataset', required=True, choices=['flsea', 'squid'])
    ap.add_argument('--weights', default=None,
                    help='weights path/dir (method-dependent)')
    ap.add_argument('--hf_id', default=None,
                    help='HuggingFace repo id for UniDepth V2')
    ap.add_argument('--beta_eff', type=float, default=None,
                    help='effective attenuation for UDCP/PC '
                         '(auto-calibrated if omitted)')
    ap.add_argument('--resnet', type=int, default=None, choices=[18, 50],
                    help='Monodepth2 backbone (default 50)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out_root', default='outputs')
    ap.add_argument('--save_preds', action='store_true',
                    help='save per-image .npy predictions (lots of disk)')
    ap.add_argument('--limit', type=int, default=None,
                    help='early-stop after N images (debugging)')
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.dataset, args.method)
    _ensure_dir(out_dir)
    preds_dir = os.path.join(out_dir, 'preds')
    if args.save_preds:
        _ensure_dir(preds_dir)

    # Build the predictor
    print(f'building {args.method}...')
    predictor = _build_method(args.method, args)

    # Open per-image CSV
    csv_path = os.path.join(out_dir, 'per_image.csv')
    cols = ['image_id', 'absrel', 'sqrel', 'rmse', 'rmse_log',
            'd1', 'd2', 'd3', 'silog', 'udac', 'runtime_ms']
    fcsv = open(csv_path, 'w')
    fcsv.write(','.join(cols) + '\n')

    # Iterate
    per_image_results = []
    ds = get_dataset(args.dataset)
    n_done, n_failed = 0, 0
    pbar = tqdm(ds, desc=f'{args.method}/{args.dataset}')
    for image_id, rgb, gt in pbar:
        if args.limit is not None and n_done >= args.limit:
            break
        try:
            t0 = time.time()
            pred = predictor.predict(rgb)
            t_ms = (time.time() - t0) * 1000

            # Standard metrics: on aligned prediction
            s, t = fit_scale_shift(pred, gt)
            pred_aligned = apply_alignment(pred, s, t)
            ref = reference_metrics(pred_aligned, gt)

            # UDAC: on the raw prediction (scale-invariant)
            u = udac(rgb, pred)

            row = {**ref, 'udac': u, 'runtime_ms': t_ms,
                   'image_id': image_id}
            per_image_results.append(row)

            fcsv.write(','.join(str(row.get(c, '')) for c in cols) + '\n')
            fcsv.flush()

            if args.save_preds:
                np.save(os.path.join(preds_dir, f'{image_id}.npy'),
                        pred.astype(np.float32))

            n_done += 1
            if n_done % 50 == 0:
                pbar.set_postfix(absrel=f'{ref["absrel"]:.3f}',
                                 udac=f'{u:.3f}')
        except Exception as e:
            n_failed += 1
            print(f'  [fail] {image_id}: {e}')

    fcsv.close()

    # Aggregate
    agg = aggregate(per_image_results)
    agg['_meta'] = {
        'method':    args.method,
        'dataset':   args.dataset,
        'n_images':  n_done,
        'n_failed':  n_failed,
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    print(f'\ndone. {n_done} images ({n_failed} failed).')
    print(f'  AbsRel = {agg.get("absrel_mean", float("nan")):.4f}')
    print(f'  UDAC   = {agg.get("udac_mean",   float("nan")):.4f}')


if __name__ == '__main__':
    main()
