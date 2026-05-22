"""
Monodepth2 wrapper.

We use the official PyTorch implementation:
    https://github.com/nianticlabs/monodepth2

Clone the repo and either install it (`pip install -e .`) or add the path
to `sys.path` via the EXTERNAL_REPOS dict in methods/_paths.py. Pretrained
weights -- pick the `mono+stereo_no_pt_640x192` checkpoint from the README.

The encoder checkpoint stores not just network weights but also the input
height/width used at training time. We respect those (loading at the train
resolution) and resize back to the original after inference.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ._paths import EXTERNAL_REPOS

# add monodepth2 to path
_MD2_PATH = EXTERNAL_REPOS.get('monodepth2', 'external/monodepth2')
if _MD2_PATH not in sys.path:
    sys.path.insert(0, _MD2_PATH)

from networks import ResnetEncoder, DepthDecoder  # noqa: E402


class Monodepth2:
    """Stereo-trained Monodepth2 (KITTI).

    Args:
        weights_dir: directory containing encoder.pth and depth.pth from
                     the official release. We use the ResNet-50 variant.
        device:      'cuda' or 'cpu'.
    """

    def __init__(self, weights_dir, device='cuda', resnet=50, **_unused):
        self.device = device

        # Encoder
        self.encoder = ResnetEncoder(num_layers=resnet, pretrained=False)
        enc_path = os.path.join(weights_dir, 'encoder.pth')
        enc_sd = torch.load(enc_path, map_location=device)

        # The official checkpoint includes scalar fields ('height', 'width')
        # alongside the parameter tensors. Pull those out before loading.
        self.train_h = int(enc_sd.get('height', 192))
        self.train_w = int(enc_sd.get('width',  640))
        enc_sd = {k: v for k, v in enc_sd.items()
                  if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(enc_sd)

        # Decoder
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc,
                                    scales=range(4))
        self.decoder.load_state_dict(
            torch.load(os.path.join(weights_dir, 'depth.pth'),
                       map_location=device)
        )

        self.encoder = self.encoder.to(device).eval()
        self.decoder = self.decoder.to(device).eval()

        self._tfm = transforms.Compose([
            transforms.Resize((self.train_h, self.train_w)),
            transforms.ToTensor(),  # converts to [0, 1] tensor
        ])

    @torch.no_grad()
    def predict(self, rgb):
        """rgb: H x W x 3 float32 in [0, 1] OR uint8 in [0, 255]."""
        if rgb.dtype != np.uint8:
            arr = (rgb * 255).astype(np.uint8)
        else:
            arr = rgb
        h0, w0 = arr.shape[:2]
        pil = Image.fromarray(arr)

        x = self._tfm(pil).unsqueeze(0).to(self.device)
        feats = self.encoder(x)
        outs = self.decoder(feats)
        disp = outs[('disp', 0)]   # (1, 1, H_train, W_train)

        # back to original resolution
        disp = F.interpolate(disp, size=(h0, w0),
                             mode='bilinear', align_corners=False)
        disp = disp.squeeze().cpu().numpy()

        # disparity -> depth (inverse). Absolute scale is arbitrary; LS
        # alignment is done in run.py.
        depth = 1.0 / np.clip(disp, 1e-6, None)
        return depth.astype(np.float32)
