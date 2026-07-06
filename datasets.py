"""
Dataset loaders for FLSea and SQUID.
"""

import os
import glob
import numpy as np
from PIL import Image
from scipy.io import loadmat


# Edit this to your data root
DATA_ROOT = os.environ.get(
    'UWDEPTH_DATA_ROOT',
    '/content/drive/MyDrive/uwdepth/data'
)



# FLSea


# Sub-collections we use (per Cai & Metzler 2025)
FLSEA_CANYONS = ['u_canyon', 'flatiron']
FLSEA_RED_SEA = ['big_dice_loop', 'cross_pyramid_loop',
                 'coral_table_loop', 'sub_pier']


def _flsea_scene_paths(group, scene):
    """Return (img_dir, depth_dir) for a FLSea scene."""
    root = os.path.join(DATA_ROOT, 'flsea', group, scene)
    # the canyons release uses 'imgs', red_sea uses 'imgs' too
    img_dir = os.path.join(root, 'imgs')
    depth_dir = os.path.join(root, 'depth')
    return img_dir, depth_dir


def _flsea_depth_path(img_path, depth_dir):
    
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(depth_dir, f'{stem}_SeaErra_abs_depth.tif')


def iter_flsea(group='all'):
    """Iterate FLSea images in our split.

    Args:
        group: 'all', 'canyons', or 'red_sea'.
    Yields:
        (image_id, rgb_float32, depth_metres)
    """
    if group == 'all':
        scenes = [('canyons', s) for s in FLSEA_CANYONS] \
               + [('red_sea', s) for s in FLSEA_RED_SEA]
    elif group == 'canyons':
        scenes = [('canyons', s) for s in FLSEA_CANYONS]
    elif group == 'red_sea':
        scenes = [('red_sea', s) for s in FLSEA_RED_SEA]
    else:
        raise ValueError(f'unknown FLSea group: {group}')

    for grp, scene in scenes:
        img_dir, depth_dir = _flsea_scene_paths(grp, scene)
        if not os.path.isdir(img_dir):
            print(f'  [skip] missing {img_dir}')
            continue

        # Image files: any .tif/.tiff/.jpg in imgs/
        
        patterns = ['*.tiff', '*.tif', '*.jpg', '*.png']
        img_paths = []
        for p in patterns:
            img_paths.extend(sorted(glob.glob(os.path.join(img_dir, p))))

        for img_path in img_paths:
            d_path = _flsea_depth_path(img_path, depth_dir)
            if not os.path.exists(d_path):
                continue  # no GT for this frame -- silently skip

            try:
                rgb = np.array(Image.open(img_path).convert('RGB'),
                               dtype=np.float32) / 255.0
                depth = np.array(Image.open(d_path), dtype=np.float32)
            except Exception as e:
                print(f'  [error] {img_path}: {e}')
                continue

            # FLSea depth: zero = invalid (sometimes seen at horizon)
            depth = np.where(depth > 0, depth, np.nan)

            stem = os.path.splitext(os.path.basename(img_path))[0]
            image_id = f'{grp}_{scene}_{stem}'
            yield image_id, rgb, depth


# SQUID


SQUID_SITES = ['Satil', 'Nachsolim', 'Katzaa', 'Michmoret']


def _load_mat_depth(path):
    
    mat = loadmat(path)
    for key, val in mat.items():
        if key.startswith('__'):
            continue
        if isinstance(val, np.ndarray) and val.ndim == 2 \
                and val.dtype.kind in 'fi':
            return val.astype(np.float32)
    raise KeyError(f'no 2D float array found in {path}')


def iter_squid():
    
    root = os.path.join(DATA_ROOT, 'squid')

    for site in SQUID_SITES:
        site_dir = os.path.join(root, site)
        if not os.path.isdir(site_dir):
            print(f'  [skip] missing {site_dir}')
            continue

        # Each scene is a image_set_NN/ subdir
        scene_dirs = sorted(d for d in os.listdir(site_dir)
                            if d.startswith('image_set_'))

        for scene in scene_dirs:
            sd = os.path.join(site_dir, scene)
            
            lefts = [f for f in os.listdir(sd)
                     if 'LFT' in f and 'resizedUndistort' in f
                     and f.endswith(('.tif', '.tiff'))]
            if not lefts:
                continue
            img_path = os.path.join(sd, sorted(lefts)[0])

            mat_path = os.path.join(sd, 'distanceFromCamera.mat')
            if not os.path.exists(mat_path):
                continue

            try:
                rgb = np.array(Image.open(img_path).convert('RGB'),
                               dtype=np.float32) / 255.0
                depth = _load_mat_depth(mat_path)
            except Exception as e:
                print(f'  [error] {site}/{scene}: {e}')
                continue

            depth = np.where(depth > 0, depth, np.nan)

            image_id = f'{site}_{scene}_LFT'
            yield image_id, rgb, depth



def get_dataset(name):
    """Return an iterator for the named dataset."""
    name = name.lower()
    if name == 'flsea':
        return iter_flsea(group='all')
    elif name == 'squid':
        return iter_squid()
    else:
        raise ValueError(f'unknown dataset: {name}')


if __name__ == '__main__':
    # Quick sanity check: count how many usable images each dataset has.
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else 'flsea'
    n = 0
    for image_id, rgb, depth in get_dataset(name):
        if n < 3:
            print(f'  {image_id}: rgb={rgb.shape}, depth={depth.shape}, '
                  f'range=[{np.nanmin(depth):.2f}, {np.nanmax(depth):.2f}]')
        n += 1
    print(f'{name}: {n} usable images')
