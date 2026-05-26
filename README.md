# Monocular Underwater Depth Estimation: Methods + UDAC

Reference implementations for the experimental review of monocular depth
estimation underwater. Eight methods are evaluated on FLSea and SQUID, and
we introduce the Underwater Depth-Attenuation Consistency (UDAC) metric.

If you use this code, please cite the paper:

```
@article{...,
  title  = {An Experimental Review of Monocular Underwater Depth Estimation},
  author = {Abhijit Boruah and Nayan M. Kakoty and Debajit Sarma},
  journal= {IEEE Trans. Computational Imaging},
  year   = {2026}
}
```

## Methods

| Key | Method            | Source                                                            |
|-----|-------------------|-------------------------------------------------------------------|
| udcp     | UDCP                  | implemented here (Drews et al. 2013)                          |
| pc       | Peng--Cosman          | implemented here (Peng & Cosman 2017)                         |
| md2      | Monodepth2            | https://github.com/nianticlabs/monodepth2 (stereo, KITTI)     |
| udepth   | UDepth                | https://github.com/uf-robopi/UDepth                           |
| dpt      | DPT-Large             | torch.hub (Intel-ISL MiDaS)                                   |
| midas    | MiDaS v3.1 (BeiT-L)   | torch.hub                                                     |
| dav2     | Depth Anything V2     | https://github.com/DepthAnything/Depth-Anything-V2            |
| unidepth | UniDepth V2 (ViT-L)   | https://github.com/lpiccinelli-eth/UniDepth                   |

## Setup

```bash
pip install -r requirements.txt
```

For the deep-learning methods that aren't on torch.hub, clone the official
repos and either install them or set `EXTERNAL_REPOS` in `methods/_paths.py`:

```bash
git clone https://github.com/nianticlabs/monodepth2 external/monodepth2
git clone https://github.com/uf-robopi/UDepth external/udepth
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/depth_anything_v2
pip install unidepth   # UniDepth V2 ships as a pip package
```

Pretrained weights:
- Monodepth2: download `mono+stereo_no_pt_640x192` from the repo README.
- UDepth: weights are in the official repo under `pretrained/`.
- DA V2: download `depth_anything_v2_vitl.pth` from the repo's releases.

## Data

We use two underwater datasets with metric ground-truth depth:

- **FLSea** (Randall & Treibitz 2023). Forward-looking RGB + SfM depth.
  We use the Canyons + Red Sea sub-collections (~12k images) following
  the split of Cai & Metzler (2025), arXiv:2507.02148.
- **SQUID** (Berman et al. 2020). 57 stereo pairs across four Israeli
  dive sites; we read the rectified left image + the `distanceFromCamera.mat`
  per scene.

Expected directory layout (edit paths in `datasets.py`):
```
DATA_ROOT/
  flsea/
    canyons/{u_canyon,flatiron,...}/imgs/*.tiff
                                  /depth/*_SeaErra_abs_depth.tif
    red_sea/{big_dice_loop,sub_pier,...}/...
  squid/
    {Satil,Nachsolim,Katzaa,Michmoret}/
      image_set_NN/
        LFT_*resizedUndistort.tif
        distanceFromCamera.mat
```

## Running

```bash
# Single method on a single dataset
python run.py --method udcp --dataset flsea
python run.py --method unidepth --dataset squid

# All methods on both datasets
for m in udcp pc md2 udepth dpt midas dav2 unidepth; do
  for d in flsea squid; do
    python run.py --method $m --dataset $d
  done
done
```

Outputs are written to `outputs/{dataset}/{method}/metrics.json` plus
per-image predictions if `--save_preds`.

## A note on alignment

Every method except UniDepth V2 outputs scale-and-shift-ambiguous depth
(disparity, relative depth, etc.). We do per-image least-squares
`(s, t)` alignment of `s*pred + t` against GT before computing reference
metrics. UDAC is computed on the raw prediction (it's scale-invariant
by construction).

## A note on UDCP / PC `beta_eff`

The classical methods need an effective attenuation `beta_eff` to convert
transmission to depth. We fit it once per dataset on a held-out calibration
subset (10% of images, fixed seed). See `methods/_classical_calibrate.py`.

## License

MIT. See `LICENSE`.
