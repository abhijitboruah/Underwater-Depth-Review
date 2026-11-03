"""
Lotus wrapper (deterministic depth variant, "Lotus-D").

Lotus adapts a pretrained Stable Diffusion v2 image backbone into a depth
predictor, but predicts depth in a single denoising step rather than the tens or
hundreds of steps that Marigold and related samplers require. This single
forward pass makes it the fast representative of the diffusion family, with an
inference cost comparable to the feed-forward foundation models.

Lotus ships a custom diffusers pipeline (not a stock diffusers class), so we
clone the repo once and import LotusDPipeline from it.

    pip install "diffusers>=0.28" "transformers>=4.45" accelerate peft
Reference checkpoint: 'jingheya/lotus-depth-d-v1-1'.
Repo (custom pipeline): https://github.com/EnVision-Research/Lotus
Output: affine-invariant relative depth in [0,1] (high = far); scale-aligned
per image downstream. No underwater fine-tuning.
"""

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from methods._paths import EXTERNAL_REPOS

LOTUS_REPO = Path(EXTERNAL_REPOS.get('lotus', 'external/Lotus'))
LOTUS_HF_ID = 'jingheya/lotus-depth-d-v1-1'


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


class Lotus:
    name = 'lotus'
    processing_res = 768
    timestep = 999

    def __init__(self, hf_id=LOTUS_HF_ID, device='cuda', **_unused):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Clone the custom-pipeline repo once.
        if not (LOTUS_REPO / 'pipeline.py').exists():
            LOTUS_REPO.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ['git', 'clone', '--depth', '1',
                 'https://github.com/EnVision-Research/Lotus.git', str(LOTUS_REPO)],
                check=True)
        if str(LOTUS_REPO) not in sys.path:
            sys.path.insert(0, str(LOTUS_REPO))
        from pipeline import LotusDPipeline  # provided by the cloned repo
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self.pipe = LotusDPipeline.from_pretrained(hf_id, torch_dtype=dtype).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def predict(self, rgb):
        arr = rgb.astype(np.float32)
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 255.0
        H, W = arr.shape[:2]
        x = torch.from_numpy(arr * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).to(self.device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
        autocast = (torch.autocast(self.device.type)
                    if self.device.type == 'cuda' else _nullctx())
        with autocast:
            pred = self.pipe(
                rgb_in=x, prompt='', num_inference_steps=1, output_type='np',
                timesteps=[self.timestep], task_emb=task_emb,
                processing_res=self.processing_res, match_input_res=True,
                resample_method='bilinear',
            ).images[0]
        depth = pred.mean(axis=-1).astype(np.float32)   # high = far, no flip
        if depth.shape != (H, W):
            depth = np.asarray(Image.fromarray(depth).resize((W, H), Image.BILINEAR),
                               dtype=np.float32)
        return depth
