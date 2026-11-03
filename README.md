# Monocular Underwater Depth Estimation: Methods + UDAC

Reference implementations and the evaluation code for the benchmark of monocular
depth estimation underwater. Eleven methods are evaluated on FLSea and SQUID
(which have metric ground-truth depth) and on COU (which does not). We introduce
the Underwater Depth-Attenuation Consistency (UDAC) metric, a reference-free
diagnostic that measures whether the ordering of a predicted depth map agrees
with the attenuation cue in the input image. UDAC is not a measure of metric
accuracy or physical correctness; a high UDAC is necessary but not sufficient
for a faithful prediction.

If you use this code, please cite the paper:

```
@article{boruah2026benchmarking,
  title  = {Benchmarking Monocular Underwater Depth Estimation:
            Datasets, Methods, and Evaluation Metrics},
  author = {Abhijit Boruah and Nayan M. Kakoty and Debajit Sarma},
  journal= {International Journal of Computer Vision (under review)},
  year   = {2026}
}
```

## Methods

| Key      | Method                     | Family      | Source                                                        |
|----------|----------------------------|-------------|--------------------------------------------------------------|
| udcp     | UDCP                       | Classical   | implemented here (Drews et al. 2013)                         |
| pc       | Peng--Cosman               | Classical   | implemented here (Peng & Cosman 2017)                        |
| md2      | Monodepth2                 | CNN         | https://github.com/nianticlabs/monodepth2 (stereo, KITTI)   |
| udepth   | UDepth                     | CNN (UW)    | https://github.com/uf-robopi/UDepth                         |
| dpt      | DPT-Large                  | ViT         | torch.hub (Intel-ISL MiDaS)                                 |
| midas    | MiDaS v3.1 (BeiT-L)        | ViT         | torch.hub                                                   |
| dav2     | Depth Anything V2 (ViT-L)  | Foundation  | https://github.com/DepthAnything/Depth-Anything-V2          |
| zoedepth | ZoeDepth (BeiT-L)          | Foundation  | transformers: `Intel/zoedepth-nyu-kitti`                    |
| depthpro | Depth Pro                  | Foundation  | transformers: `apple/DepthPro-hf`                           |
| unidepth | UniDepth V2 (ViT-L)        | Foundation  | https://github.com/lpiccinelli-eth/UniDepth                 |
| lotus    | Lotus-D (SD v2, 1-step)    | Diffusion   | https://github.com/EnVision-Research/Lotus + `jingheya/lotus-depth-d-v1-1` |

## Setup

```bash
pip install -r requirements.txt
```

Methods that load from Hugging Face or pip need no external clone:

```bash
pip install "transformers>=4.45" accelerate     # ZoeDepth, Depth Pro
pip install "diffusers>=0.28" peft               # Lotus (custom pipeline)
pip install unidepth                             # UniDepth V2
```

For the remaining deep methods, clone the official repos and either install them
or set `EXTERNAL_REPOS` in `methods/_paths.py`:

```bash
git clone https://github.com/nianticlabs/monodepth2 external/monodepth2
git clone https://github.com/uf-robopi/UDepth external/udepth
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/depth_anything_v2
# Lotus is cloned automatically on first use into external/Lotus
```

Pretrained weights:
- Monodepth2: download `mono+stereo_no_pt_640x192` from the repo README.
- UDepth: weights are in the official repo under `pretrained/`.
- DA V2: download `depth_anything_v2_vitl.pth` from the repo's releases.
- ZoeDepth, Depth Pro, Lotus, UniDepth V2: weights download automatically from
  Hugging Face on first use.

## Data

Two datasets carry metric ground-truth depth:

- **FLSea** (Randall & Treibitz 2023). Forward-looking RGB + SfM depth. We use
  the Canyons + Red Sea sub-collections (~12k images) following the split of
  Cai & Metzler (2025), arXiv:2507.02148.
- **SQUID** (Berman et al. 2020). 57 stereo pairs across four Israeli dive
  sites; we read the rectified left image + the `distanceFromCamera.mat` per
  scene.

One dataset is reference-free (UDAC only, no accuracy metrics):

- **COU** (Common Objects Underwater). RGB test images with no ground-truth
  depth. Used for the reference-free UDAC evaluation.

Expected directory layout (edit paths in `datasets.py`, or set
`UWDEPTH_DATA_ROOT`):
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
  cou/
    images/*.jpg
```

The evaluation splits are defined in `datasets.py` (the FLSea sub-collections
and the SQUID sites); no separate split files are needed to reproduce them.

## Running

```bash
# Single method on a single dataset
python run.py --method udcp     --dataset flsea
python run.py --method lotus    --dataset squid
python run.py --method unidepth --dataset cou      # UDAC only (no GT)

# All eleven methods on the three datasets
for m in udcp pc md2 udepth dpt midas dav2 zoedepth depthpro unidepth lotus; do
  for d in flsea squid cou; do
    python run.py --method $m --dataset $d
  done
done
```

Outputs are written to `outputs/{dataset}/{method}/`:
- `metrics.json` -- aggregated metrics.
- `per_image.csv` -- the per-image score arrays (AbsRel, ..., UDAC, runtime).
- `preds/*.npy` -- per-image predictions, if `--save_preds` is given.

On COU, only the UDAC column is populated, since no ground-truth depth exists.

## A note on alignment

Every method except the native metric models outputs scale-and-shift-ambiguous
depth (disparity, relative depth, etc.). We do per-image least-squares `(s, t)`
alignment of `s*pred + t` against GT before computing the reference metrics. The
native metric models (UniDepth V2, Depth Pro, ZoeDepth) are reported both before
and after this alignment in the paper. UDAC is computed on the raw prediction; it
is scale-invariant by construction.

## A note on UDCP / PC `beta_eff`

The classical methods need an effective attenuation `beta_eff` to convert
transmission to depth. We fit it once per dataset on a calibration subset of one
image in every ten (see `methods/_classical_calibrate.py`). Because `beta_eff`
enters the depth only as a single global scale factor, and every reported
classical-method result is computed after the per-image least-squares alignment
(which removes any global scale) or as scale-invariant UDAC, its value and the
composition of this calibration subset do not change any reported number. On COU,
which has no ground truth, `beta_eff` is fixed rather than calibrated.

## License

MIT. See `LICENSE`.
