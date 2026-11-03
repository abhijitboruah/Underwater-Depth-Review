"""
ZoeDepth wrapper.

ZoeDepth adds a metric "bin" head on top of a frozen MiDaS relative-depth
backbone, so it inherits the MiDaS multi-dataset representation while producing
metric output. It is trained jointly on the indoor NYU and outdoor KITTI
corpora; we apply no underwater fine-tuning. The processor pads the input to the
backbone resolution, and the original image size is supplied at post-processing
so the returned prediction is unpadded and resized to full resolution.

Loaded from the Hugging Face checkpoint through the transformers library:
    pip install "transformers>=4.41" accelerate
Reference checkpoint: 'Intel/zoedepth-nyu-kitti'.
Output: metric depth (high = far); scale-aligned per image downstream.
"""

import numpy as np
import torch
from PIL import Image


class ZoeDepth:
    def __init__(self, hf_id='Intel/zoedepth-nyu-kitti', device='cuda', **_unused):
        from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(hf_id)
        self.net = ZoeDepthForDepthEstimation.from_pretrained(hf_id).to(device).eval()

    @torch.no_grad()
    def predict(self, rgb):
        arr = (rgb * 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        h, w = arr.shape[:2]
        pil = Image.fromarray(arr)
        inputs = self.processor(images=pil, return_tensors='pt').to(self.device)
        outputs = self.net(**inputs)
        post = self.processor.post_process_depth_estimation(
            outputs, source_sizes=[(h, w)])
        depth = post[0]['predicted_depth']
        return depth.squeeze().float().cpu().numpy().astype(np.float32)
