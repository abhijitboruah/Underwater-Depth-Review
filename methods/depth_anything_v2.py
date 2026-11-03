"""
Depth Anything V2 wrapper.

Official repo:
    https://github.com/DepthAnything/Depth-Anything-V2

Clone the repo and download `depth_anything_v2_vitl.pth` from the
releases page (the ViT-L checkpoint, ~1.3 GB).

We use the ViT-Large variant (335M parameters). The smaller variants
(Base, Small) are also supported -- set `encoder='vitb'` or `'vits'`.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from ._paths import EXTERNAL_REPOS


_DAV2_PATH = EXTERNAL_REPOS.get('depth_anything_v2', 'external/depth_anything_v2')
if _DAV2_PATH not in sys.path:
    sys.path.insert(0, _DAV2_PATH)


class DepthAnythingV2:
    # Config dicts copied from the official repo's run.py
    _MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64,
                 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128,
                 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256,
                 'out_channels': [256, 512, 1024, 1024]},
    }

    def __init__(self, weights_path, encoder='vitl',
                 input_size=518, device='cuda', **_unused):
        self.device = device
        self.input_size = input_size

        try:
            from depth_anything_v2.dpt import DepthAnythingV2 as _Net
        except ImportError as e:
            raise ImportError(
                "Depth-Anything-V2 official code not on sys.path. Clone "
                f"https://github.com/DepthAnything/Depth-Anything-V2 into "
                f"{_DAV2_PATH}."
            ) from e

        cfg = self._MODEL_CONFIGS[encoder]
        self.net = _Net(**cfg)
        sd = torch.load(weights_path, map_location='cpu')
        self.net.load_state_dict(sd)
        self.net = self.net.to(device).eval()

    @torch.no_grad()
    def predict(self, rgb):
        if rgb.dtype != np.uint8:
            arr = (rgb * 255).astype(np.uint8)
        else:
            arr = rgb
        # The official repo exposes a convenient .infer_image() that
        # handles its own resize-multiple-of-14 and normalisation. Use it.
        depth = self.net.infer_image(arr, self.input_size)
        # output is relative depth (already at input resolution)
        return depth.astype(np.float32)
