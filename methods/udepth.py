"""
UDepth wrapper.

Official repo: https://github.com/uf-robopi/UDepth
Clone it and either install or add to sys.path (see methods/_paths.py).

The released checkpoint produces per-image normalised depth in [0, 1], not
metric metres -- the LS scale-shift alignment in run.py is applied
regardless, so this isn't a practical issue for the evaluation, but it's
worth noting since "underwater-specific = metric" would be too easy an
interpretation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ._paths import EXTERNAL_REPOS


_UDEPTH_PATH = EXTERNAL_REPOS.get('udepth', 'external/udepth')
if _UDEPTH_PATH not in sys.path:
    sys.path.insert(0, _UDEPTH_PATH)

# The class name depends on the release; the current repo exposes
# `UDepth` as the top-level model. We import lazily inside __init__ so
# that ImportError surfaces with a clear message about the repo.


class UDepth:
    def __init__(self, weights_path, device='cuda',
                 input_size=(256, 256), **_unused):
        self.device = device
        self.input_h, self.input_w = input_size

        try:
            from model.udepth import UDepth as _UDepthNet
        except ImportError as e:
            raise ImportError(
                "UDepth's official code wasn't found on sys.path. Clone "
                f"https://github.com/uf-robopi/UDepth into {_UDEPTH_PATH} "
                "and edit methods/_paths.py if needed."
            ) from e

        self.net = _UDepthNet(n_bins=80)
        sd = torch.load(weights_path, map_location=device)
        # the released checkpoint sometimes wraps weights in 'model' key
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        self.net.load_state_dict(sd, strict=False)
        self.net = self.net.to(device).eval()

        # ImageNet stats are NOT used here -- the network is trained in
        # the RMI feature space which is computed internally from raw
        # [0, 1] RGB input. (Spent an evening figuring this out; the
        # README is a bit terse about it.)
        self._tfm = transforms.Compose([
            transforms.Resize((self.input_h, self.input_w)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def predict(self, rgb):
        if rgb.dtype != np.uint8:
            arr = (rgb * 255).astype(np.uint8)
        else:
            arr = rgb
        h0, w0 = arr.shape[:2]
        pil = Image.fromarray(arr)

        x = self._tfm(pil).unsqueeze(0).to(self.device)
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]  # the net returns (depth, bin_edges) in newer code
        out = F.interpolate(out, size=(h0, w0), mode='bilinear',
                            align_corners=False)
        depth = out.squeeze().cpu().numpy()
        return depth.astype(np.float32)
