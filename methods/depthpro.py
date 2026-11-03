"""
Depth Pro wrapper.

Depth Pro is a zero-shot metric foundation model built on a multi-scale ViT. It
predicts a sharp high-resolution depth map in under a second and estimates focal
length internally, so no camera intrinsics are required. We disable the
field-of-view head, as it affects only absolute metric scale, which the per-image
alignment removes and to which UDAC is invariant. The authors document degraded
behaviour on translucent surfaces and volumetric scattering, a caveat directly
relevant to the underwater regime.

Loaded from the Hugging Face checkpoint through the transformers library:
    pip install "transformers>=4.45" accelerate
Reference checkpoint: 'apple/DepthPro-hf'.
Output: metric depth (high = far); scale-aligned per image downstream.
"""

import numpy as np
import torch
from PIL import Image


class DepthPro:
    def __init__(self, hf_id='apple/DepthPro-hf', device='cuda', **_unused):
        from transformers import AutoImageProcessor, DepthProForDepthEstimation
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.net = DepthProForDepthEstimation.from_pretrained(hf_id).to(device).eval()

    @torch.no_grad()
    def predict(self, rgb):
        arr = (rgb * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        h, w = arr.shape[:2]
        pil = Image.fromarray(arr)
        inputs = self.processor(images=pil, return_tensors='pt').to(self.device)
        outputs = self.net(**inputs)
        # The field-of-view head is ignored; we use only the predicted depth.
        post = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)])
        depth = post[0]['predicted_depth']
        return depth.squeeze().float().cpu().numpy().astype(np.float32)
