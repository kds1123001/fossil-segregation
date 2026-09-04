# fossilseg

3D U-Net segmentation for pulling fossils out of micro-CT rock matrix, trained on procedurally generated data since I don't have access to a synchrotron or a pile of manually-traced specimens.

## why synthetic

Real micro-CT fossil segmentation papers (see FossilNeRF, the Bruker/SkyScan paleo pipelines, etc.) all hit the same wall: bone and shell get mineralized with the same infill minerals as the surrounding rock during diagenesis, so a scalar density value alone doesn't separate them. The interesting part of the problem is a model that can pick up on texture and shape continuity instead of raw intensity.

I don't have real specimen scans or the months of hand-tracing needed to label them, so `fossilseg/synth.py` builds volumes that have the same failure mode on purpose: tube/spiral/blob "fossils" embedded in Perlin-ish noise matrix with density distributions that overlap, but different local texture statistics (matrix is grainier, fossil is smoother — this is the actual signal the network has to learn). Boundaries get blurred to fake partial-volume effect, plus ring artifacts, speckle noise, and a few crack fissures thrown in.

It's not a substitute for the real problem. It's a stand-in that exercises the same architecture, loss, and inference pipeline you'd point at an actual synchrotron dataset, without needing one.

## what's here

```
fossilseg/
  synth.py       synthetic volume + mask generator
  model.py       3D U-Net, 4 downsamples, InstanceNorm, LeakyReLU
  losses.py      Dice + CrossEntropy combo loss
  dataset.py     patch sampler, oversamples fossil-containing crops, flip/rot90/gamma aug
  infer.py       sliding window inference, Gaussian-weighted patch blending
  mesh.py        marching cubes -> .obj export
  visualize.py   orthogonal slice viewer with mask/pred overlay
scripts/
  generate_data.py
  predict.py
```

`train.py` lives inside the package and doubles as a script (`python -m fossilseg.train ...`).

## running it

```
pip install -r requirements.txt

python scripts/generate_data.py --out_dir data/train --n 40 --shape 96 96 96 --seed0 0
python scripts/generate_data.py --out_dir data/val   --n 8  --shape 96 96 96 --seed0 10000

python -m fossilseg.train --train_dir data/train --val_dir data/val \
    --out_dir checkpoints --patch_size 64 --batch_size 2 --epochs 150 --steps_per_epoch 250

python scripts/predict.py --checkpoint checkpoints/best.pt --volume data/val/vol_0000.npy \
    --mask data/val/mask_0000.npy --out_dir predictions --mesh
```

`predict.py` writes the predicted mask as `.npy`, a 3-panel orthogonal slice PNG with the prediction overlaid, and (with `--mesh`) an `.obj` you can drop into Blender/MeshLab.

## architecture notes

Plain 3D U-Net, base channel width 16 by default (~6M params), InstanceNorm instead of BatchNorm since batch sizes on 3D volumes are tiny. Loss is Dice + CE — Dice alone is unstable early in training when the fossil class is a couple percent of the volume, CE alone doesn't push hard enough on the minority class, so I'm running both and summing.

Inference is patch-based with overlapping windows and Gaussian blending at the seams (straight tiling gives visible block artifacts at patch boundaries, this fixes that).

The fossil-biased patch sampling in `dataset.py` matters more than it looks — at ~1-3% fossil voxel fraction per volume, uniform random crops mean most training patches are pure matrix and the model just learns to predict background. Biasing crop centers toward fossil voxels (with a fallback to uniform crops so the model still sees background-only patches) fixed convergence for me.

## what I'd add with more time

- SwinUNETR / UNETR variant to actually test whether global attention helps on the ambiguous-boundary cases, benchmarked against the plain U-Net on identical synthetic splits
- Deeper CT artifact simulation (proper sinogram-domain streak artifacts instead of the crude radial sinusoid hack currently in `synth.py`)
- Uncertainty-based active learning loop (flag high-entropy boundary voxels for review) — this is the actual production shape of these pipelines, full manual tracing is what you're trying to eliminate but someone still has to sign off on the boundary
- Test-time augmentation for inference (flip/rotate ensembling)

## honest limitations

Synthetic fossils are geometrically simple (parametric tubes and spirals) compared to real anatomy. Density/texture separation is a designed-in signal, not empirically measured from real specimen scans, so results here don't tell you what Dice score you'd get on an actual synchrotron dataset. Treat this as a working pipeline you'd retarget at real data, not a benchmark result.

# credits

all credits go to me yo boi kds1123001 aka dinokid64
