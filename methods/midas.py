"""
MiDaS v3.1 wrapper.

Loaded via torch.hub. We use the BeiT-Large 512 variant -- it's the
strongest of the v3.1 family (others are BeiT-L 384, Swin2-L 384, etc.).

Output is scale-and-shift-invariant inverse depth.
"""

import numpy as np
import torch
import torch.nn.functional as F


class MiDaS:
    """Args:
        variant: which torch.hub MiDaS model to load. Defaults to the
                 best v3.1 checkpoint (BeiT-Large 512).
        device:  'cuda' or 'cpu'.
    """

    def __init__(self, variant='DPT_BEiT_L_512', device='cuda', **_unused):
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", variant,
                                    trust_repo=True)
        self.model = self.model.to(device).eval()

        midas_tfms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                    trust_repo=True)
        # Pick the right transform for the BEiT-L 512 model.
        # The MiDaS repo exposes one transform per backbone family.
        if 'BEiT' in variant or 'beit' in variant:
            self.transform = midas_tfms.beit512_transform
        elif 'Swin' in variant or 'swin' in variant:
            self.transform = midas_tfms.swin384_transform
        else:
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
        inv = out.squeeze().cpu().numpy()
        depth = 1.0 / np.clip(inv, 1e-6, None)
        return depth.astype(np.float32)
