"""
UniDepth V2 wrapper.

UniDepth V2 produces *metric* depth directly by conditioning on camera
intrinsics. At inference, the model exposes an internal intrinsics-estimation
head that runs when no K is supplied -- we use that here so the protocol
stays purely monocular and matched to the other methods.

Install:
    pip install unidepth        # easiest
or clone https://github.com/lpiccinelli-eth/UniDepth and install from source.

Reference checkpoint: 'lpiccinelli/unidepth-v2-vitl14' (ViT-Large).
"""

import numpy as np
import torch


class UniDepthV2:
    def __init__(self, hf_id='lpiccinelli/unidepth-v2-vitl14',
                 device='cuda', **_unused):
        self.device = device

        try:
            from unidepth.models import UniDepthV2 as _Net
        except ImportError as e:
            raise ImportError(
                "UniDepth not installed. Try `pip install unidepth` or "
                "clone https://github.com/lpiccinelli-eth/UniDepth."
            ) from e

        self.net = _Net.from_pretrained(hf_id)
        self.net = self.net.to(device).eval()

    @torch.no_grad()
    def predict(self, rgb):
        if rgb.dtype != np.uint8:
            arr = (rgb * 255).astype(np.uint8)
        else:
            arr = rgb

        # UniDepth expects (3, H, W) tensor on the model's device, uint8 OK.
        # We pass intrinsics=None so the network estimates them internally;
        # this keeps the comparison purely monocular.
        x = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
        out = self.net.infer(x, intrinsics=None)

        # `out` is a dict; the depth is under 'depth' (and intrinsics
        # under 'intrinsics' if you ever want them).
        depth = out['depth']        # shape (1, 1, H, W) tensor
        depth = depth.squeeze().cpu().numpy()
        return depth.astype(np.float32)
