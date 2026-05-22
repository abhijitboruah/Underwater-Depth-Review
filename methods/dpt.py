"""
DPT (Dense Prediction Transformer) wrapper.

Loaded via torch.hub from the Intel ISL MiDaS repository. We use the
DPT-Large variant (ViT-L/16 backbone). Output is scale-and-shift-invariant
inverse depth -- aligned in run.py via least squares.
"""

import numpy as np
import torch
import torch.nn.functional as F


class DPT:
    def __init__(self, device='cuda', **_unused):
        self.device = device
        # The first call to torch.hub downloads the checkpoint to
        # ~/.cache/torch/hub on first run.
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large",
                                    trust_repo=True)
        self.model = self.model.to(device).eval()

        midas_tfms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                    trust_repo=True)
        self.transform = midas_tfms.dpt_transform

    @torch.no_grad()
    def predict(self, rgb):
        if rgb.dtype != np.uint8:
            arr = (rgb * 255).astype(np.uint8)
        else:
            arr = rgb
        h0, w0 = arr.shape[:2]

        x = self.transform(arr).to(self.device)
        out = self.model(x)
        out = F.interpolate(out.unsqueeze(1), size=(h0, w0),
                            mode='bilinear', align_corners=False)
        # DPT outputs inverse depth. Invert to a depth-like quantity for
        # consistency with the other methods. LS alignment handles scale.
        inv = out.squeeze().cpu().numpy()
        depth = 1.0 / np.clip(inv, 1e-6, None)
        return depth.astype(np.float32)
