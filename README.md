# flow-hsi
Hyperspectral imaging

# git lfs
https://git-lfs.com/
1. git lfs install
2. git lfs track "*.pdf"
    
# usage
usage: python main.py [--help] [--hydra-help] [--version] [--cfg {job,hydra,all}] [--resolve] [--package PACKAGE]
               [--run] [--multirun] [--shell-completion] [--config-path CONFIG_PATH]
               [--config-name CONFIG_NAME] [--config-dir CONFIG_DIR]
               [--experimental-rerun EXPERIMENTAL_RERUN]
               [--info [{all,config,defaults,defaults-tree,plugins,searchpath}]]
               [overrides ...]

run: python main.py [--config-name CONFIG_NAME] [--config-dir CONFIG_DIR] [[config_group.name=VALUE] ...] 

print: python main.py --cfg {job,hydra,all}
help: python main.py --help

# examples
python main.py
python main.py --config.name config.test
python main.py --config.name config dataset.type=img

# features
python main.py sample-image-data -i /data/korepanov/color-transfer/huawei/src -o /home/korepanov/work/color-sr/.data/x3 -n 3
    -i - input folder containing source images
    -o - output dataset folder
    -n - scale

# Examples

$ python main.py
$ python main.py task=test

## NTIRE: shared spectral encoder, two independent experiments

The new encoder-decoder baseline is available through two **separate runs**:

```powershell
# 1. Five epochs of masked-channel encoder pretraining.
python main.py --config-name pretrain task=train
# 2. Twenty epochs of paired NTIRE restoration using the exported encoder.
python main.py --config-name restore task=train
```

Equivalent flow-group commands are
`python main.py flows@_global_=pretrain task=train` and
`python main.py flows@_global_=restore task=train`.
Both select NTIRE automatically. Set the paired source/target paths in
`.configs\flows\data\ntire.yaml` for your machine before training. Inputs
are **31-band HSI**, not three-channel RGB; use consistent source/target
normalization and the same ordered wavelengths. Training uses 128x128 patches;
validation uses full images. The default device is GPU 0; append
`accelerator=cpu devices=1` for CPU execution.

### Encoder and task heads

The shared encoder has widths **32 -> 64 -> 128 -> 192**, at full, half,
quarter and eighth resolution. A 3x3 stem and a local residual convolutional
block process full-resolution detail; stride-2 convolutions lead to **1, 2, 2
MSAB-style blocks**, using **2, 4, 6 attention heads**, at the reduced scales.
Each attention path is pre-normalized. Feed-forward paths use 1x1 expansion,
depthwise 3x3 mixing and 1x1 reduction with expansion 2. Per-pixel channel
LayerNorm replaces spatial GroupNorm. Spectral attention still uses spatial
statistics: full-image inference and tiled inference need not be identical.
Explicit target-size interpolation supports odd spatial sizes.

The **pretraining head** reconstructs from the bottleneck only, through
192 -> 96 -> 48 -> 32 -> 31 convolutional features and bilinear upsampling.
It has no attention, shallow feature skips or input residual bypass. Training
alternates source and target cubes between batches, flipping the starting
domain each epoch so even a single-batch dataset uses both domains over time.
Each selected cube is its own reconstruction target, not a source-to-target
translation pair. Training zeroes 16 random whole bands and embeds their explicit
31-band missingness flags into the encoder stem, distinguishing a hidden band
from a genuinely dark one. The objective is **hidden-band MAE + 0.1 visible-band
MAE + 0.15 (1 - SSIM)**. All five epochs do reconstruction; there is no automatic transition to
paired training. This is dense masked-channel autoencoding, not sparse
token-dropping ViT MAE.

The **restoration head** projects the encoder and decoder features separately,
concatenates them, then fuses them with a learned 1x1 projection at each skip.
It uses 2 MSAB-style blocks at quarter resolution, 1 at half resolution, and
**three local residual convolutional blocks at full resolution**
(128 -> 64 -> 32 -> 31). It predicts a residual added
to the source HSI. Small nonzero output weights initialize it near identity
without blocking decoder gradients. There are **no Gaussian/hypernetwork
branches or repeated U-Net cascades** in this baseline.

The refined decoder is the default for `restore`. Its encoder and MAE head are
unchanged: existing pretrained `encoder.pt` artifacts remain usable, without
repeating pretraining. Full-resolution refinement increases activation memory;
reduce batch size if needed. No extra full-encoder pass is added.

For a separate coarse-calibration ablation:

```powershell
python main.py flows@_global_=restore_calibrated task=train
# Equivalent root config:
python main.py --config-name restore_calibrated task=train
```

This adds a small 1x1/GELU/3x3 head on eighth-resolution encoder features.
It predicts per-band gain and offset fields, bounded by tanh and bilinearly
upsampled, to produce `source * (1 + gain) + offset + detail_residual`.
Both corrections start at zero (calibration is identity initially); their
default magnitude limits are 0.1 each. Configure
`model.params.layers.spectral.params.calibration_gain_scale` and
`calibration_offset_scale` for a different bound. No output clipping is used.
The calibration head belongs to the decoder optimizer group and trains during
the frozen-encoder warmup. It is absent from the refined baseline, allowing
the two experiments to isolate its contribution.

### Transfer, schedule and resume

Pretraining checkpoints are ranked by deterministic `val_masked_mae`, measured
over target cubes in the entire validation loader with repeatable,
batch-index-seeded masks. Validation remains target-only for comparable
checkpoint selection; training metrics follow the currently selected domain.
The best encoder is exported atomically, on rank zero, to:

```text
.experiments\spectral.mae.ntire\logs\checkpoints\encoder.pt
```

The artifact contains only encoder weights, architecture metadata and selection
metrics, not decoder weights or optimizer state. Restoration requires this
artifact and checks its encoder metadata and state dictionary strictly. Change
`pipeline.params.encoder_checkpoint=...` when pretraining used a different
experiment/output directory. A missing or incompatible artifact fails
explicitly. To deliberately train a random-encoder ablation:

```powershell
python main.py --config-name restore pipeline.params.encoder_checkpoint=null pipeline.params.allow_random_encoder=true experiment=spectral.scratch.ntire
```

Restoration freezes the encoder for the first **500 optimizer updates, capped
at the end of epoch 0**, then fine-tunes the complete model. With fewer than
500 updates per epoch, it unfreezes at epoch 1. Gradient accumulation does not
count microbatches as separate optimizer updates. Decoder LR starts at **2e-4**;
encoder LR starts at **2e-5**; both groups are present from the beginning in one
AdamW optimizer, with epoch-wise cosine decay and no optimizer reset.
`pipeline.params.freeze_encoder_steps=0` disables the frozen warmup.
Pretraining uses LR 2e-4 for both encoder and its lightweight head.

Restoration retains all-band MAE, floored MRAE, spectral cosine, adjacent-band
difference and CIE Lab perceptual losses described below and adds
**0.15 (1 - SSIM)**; checkpoints are
ranked by `val_mrae`. Both experiments show training/validation **PSNR,
CIEDE2000 (`de`) and spectral angle mapper (`sam`)**. SAM averages the
per-pixel angle across all 31 bands in **degrees** (lower is better).
Identical zero spectra receive angle 0; exactly one zero spectrum receives
90 degrees to keep dark-pixel metrics finite. During pretraining these describe masked
self-reconstruction; during restoration they describe source-to-target correction. Previews reuse
the current validation prediction, showing masked input for pretraining.

SSIM is differentiable spatial SSIM computed independently per spectral band
and averaged across all 31 bands, not an RGB projection or a spectral-angle
loss. It uses a Gaussian 11x11 window (sigma 1.5), a smaller window for tiny
patches, fixed data range 1, float32 arithmetic, and no prediction clipping.
Both experiments expose `pipeline.params.ssim_weight=0.15`; set it to 0 for
the previous objective. L1 was already used; no L2 term was replaced.
`ssim_loss` and `val_ssim_loss` log the unweighted structural term.
Checkpoint selection remains masked MAE for pretraining and MRAE for restoration.
SAM logs now use degrees, including CSV logs; do not compare new values
directly with older radian logs (multiply old values by 180/pi).

The runs have separate output directories: `spectral.mae.ntire`,
`spectral.restore.refined.ntire` and `spectral.restore.calibrated.ntire`.
Each resumes its own `last.ckpt` by default, including
optimizer, scheduler, epoch and update count. Full restoration resume does not
reload the initial encoder artifact, so that artifact is no longer required.
Use a new `experiment=...` for a fresh run or architecture change. Existing
cascade checkpoints cannot initialize this encoder. Masked pretraining is an
experiment, not a demonstrated accuracy gain; compare against the scratch
ablation on held-out scenes.

The refined/calibrated heads cannot resume the original restoration decoder.
New full checkpoints validate decoder settings as well as encoder settings.
To resume an old `spectral.restore.ntire` checkpoint with its original decoder:

```powershell
python main.py flows@_global_=restore task=train experiment=spectral.restore.ntire model.params.layers.spectral.params.skip_fusion=add model.params.layers.spectral.params.full_resolution_blocks=1
```

Changing calibration bounds also changes the model's behavior and requires a
new experiment rather than resuming a checkpoint with different bounds.

Both experiments inherit the NTIRE loader settings described below. If host
RAM is tight, append `data.params.num_workers=0 data.params.pin_memory=false`
to isolate prefetch/worker pressure before tuning workers upward.

## Legacy NTIRE: CM U-Net cascades with an optional Gaussian hypernetwork

`python main.py` without an experiment config still selects the 31-band NTIRE dataset, the `usgs_unet` model,
and its dedicated training pipeline. The configuration is in
`.configs\flows\train.yaml`, `.configs\flows\model\usgs_unet.yaml`, and
`.configs\flows\pipeline\usgs_unet.yaml`. Override the NTIRE source/target paths
in `.configs\flows\data\ntire.yaml` for your machine. Inputs and targets must
be paired, consistently scaled, wavelength-aligned HSI cubes, not RGB images.
Training uses aligned 128x128 crops with batch size 4; validation and test
retain full images with batch size 1 to reduce memory use. Keep validation
scenes separate from the final test set when selecting loss weights.

### NTIRE loading and avoiding stop-go throughput

The reader memory-maps paired HWC `.npy` files read-only, selects the same crop
and flips for both cubes, then copies **only that patch** into writable CHW
tensor storage. It no longer loads and transposes full cubes for 128x128
training patches. Evaluation crops, when configured, are centered and
deterministic; with `crop_size: 0`, evaluation retains the original image
subject to the reader's existing multiple-of-four trimming. Dtype conversion
and source-only `norm` scaling retain the previous behavior.

Source/target filenames must match exactly, and each pair must have matching
HWC shapes and 31 bands. Missing, empty, corrupt, or mismatched inputs fail
explicitly rather than returning `None` or silently pairing different scenes.
Splits are initialized only when their stage is requested.

Defaults are four persistent workers, two prefetched batches per worker, and
pinned transfer memory. Tune `data.params.num_workers` and
`data.params.prefetch_factor` for your storage/RAM; more workers are not always
faster. `num_workers=0` is supported and automatically omits worker-only options.
Workers use PyTorch-seeded random crops and augmentations and stay alive across
epochs. Their startup on Windows no longer triggers circular selector imports.

Previews reuse the first validation prediction without another dataloader
iterator or full-image model pass. They save one bounded 256x256 pseudo-RGB
patch every five epochs (including epoch 0), only on rank zero; sanity
validation never writes a preview. `visualize_freq=0` disables previews.
Validation still evaluates complete images each epoch, and checkpoints still
write synchronously: these remain real work, not data-loader stalls.

```powershell
# Reduce storage contention or prefetch memory.
python main.py data.params.num_workers=2 data.params.prefetch_factor=1
# Less frequent validation/checkpoint I/O, keeping their schedules aligned.
python main.py val_every_n_epochs=5 save_freq=5
# Disable preview I/O.
python main.py visualize_freq=0
```

### Model and training

The default has **three convolutional CM U-Net cascades** (`num_cascades: 3`)
and **no parameter branch** (`use_hypernet: false`). In this mode no hypernetwork
stem, parameter U-Nets, or injection heads are constructed or executed: each
cascade simply adds its predicted residual to the current 31-band image.

Set `use_hypernet: true` to enable the second branch. Its parameter U-Nets use
convolutional residual blocks at full resolution and MST MSAB spectral attention
at half/quarter resolution and the bottleneck (`hyper_num_blocks: 1` per level).
Each cascade has a separate Gaussian injection head predicting centers, widths,
gain amplitudes, and offset amplitudes for six Gaussian components per pixel.
Their wavelength-dependent fields modulate the CM spectrum using
`spectrum * (1 + gain) + offset`, bounded by `gain_scale` and `offset_scale`.
Gaussian amplitudes start at zero. FiLM has been removed.
Odd spatial sizes remain supported by explicit skip-size interpolation.
Constructor initialization is preserved: every cascade initially maps its
input to itself.
The experiment directory is now `usgs-unet.cm3.ntire` to avoid resuming
incompatible FiLM / two-cascade checkpoints. Use a distinct experiment name
when enabling the hypernetwork or changing cascade count; do not resume a
checkpoint from a different architecture. `hyper_full_resolution_msab=true`
restores full-resolution attention when the hypernetwork is enabled.

The default 20-epoch run has two phases using the same model and optimizer:

- Epochs 0-4: masked-channel autoencoder pretraining on **training targets only**.
  Independently mask 16 of 31 whole bands per sample to zero and minimize mean
  absolute error only on the hidden bands. This is masked reconstruction, not
  an efficient token-dropping ViT MAE: dense U-Nets still process all bands.
- Epochs 5-19: paired source-to-target color restoration. The objective is
  all-band MAE + 0.1 MRAE + 0.1 spectral cosine distance + 0.1 spectral
  first-difference error + 0.1 perceptual loss. Perceptual loss is differentiable
  CIE Lab L1 divided by 100, not an ImageNet/VGG feature loss. MRAE uses a 1e-3
  denominator floor to limit the influence of dark bands; the cosine term
  ignores zero target spectra.

Both training and validation progress bars show PSNR, `de` (CIEDE2000)
and `sam` (spectral angle in degrees, with the zero-spectrum convention above).
PSNR averages per-image PSNR across all 31 bands with a fixed data range of 1.
During pretraining, training metrics describe masked reconstruction; validation
metrics always describe paired source-to-target restoration. Training metrics
reuse the current prediction and do not require an extra model forward pass.

Validation masking is repeatable. `val_masked_mae` tracks reconstruction during
pretraining on the first validation batch by default (`masked_val_batches: 1`),
avoiding a second full validation pass. Set that count higher for a larger
diagnostic sample or zero to disable it. `val_mrae` and `val_loss` **always**
measure paired source-to-target restoration, so checkpoint ranking stays
comparable across phases. The
learning-rate cosine schedule restarts for fine-tuning. SWA is disabled to
avoid overriding that schedule. `last.ckpt` resumes model, optimizer, scheduler,
and epoch/phase together; use a new experiment name for a fresh run.

```powershell
# Defaults: masked-channel pretraining, then paired fine-tuning.
python main.py
# Enable the Gaussian parameter branch.
python main.py model.params.layers.pyramid.params.use_hypernet=true experiment=usgs-unet.gaussian3.ntire
# Use two CM-only cascades instead of three.
python main.py model.params.layers.pyramid.params.num_cascades=2 experiment=usgs-unet.cm2.ntire
# Supervised-only ablation.
python main.py pipeline.params.pretrain_epochs=0 experiment=usgs-unet.supervised.ntire
# Longer run with a longer pretraining phase.
python main.py epochs=100 pipeline.params.pretrain_epochs=20 experiment=usgs-unet.mae100.ntire
# CPU execution.
python main.py accelerator=cpu devices=1
```

### Improving 31-band color matching

MAE anchors absolute intensity; spectral cosine distance constrains spectral
shape, and matching adjacent-band differences discourages implausible spectral
wiggles without forcing all spectra to be flat. Compare these loss weights and
masked pretraining against the supervised-only ablation on held-out scenes;
pretraining is a hypothesis, not an established accuracy improvement.

Color loss and DeltaE use the **CIE 1931 2-degree observer with D65**, assuming
31 ordered relative-reflectance bands from **400 to 700 nm in 10 nm steps**.
XYZ is integrated with trapezoidal weights and normalized to unit perfect-
diffuser luminance. Lab uses the corresponding band-limited D65 diffuser white
so a flat spectrum remains neutral. Integration covers only the measured
400-700 nm range; this is not full-visible-spectrum colorimetry. Numeric CIE
observer/D65 tables are embedded; no runtime downloads or new dependencies
are needed. Perceptual loss operates directly on Lab without RGB gamut clipping.

This assumption is not valid for arbitrary camera channels, independently
normalized bands, or radiance already containing an illuminant spectrum.
For those data, adapt the color conversion before interpreting DeltaE or using
the perceptual loss. `perceptual_weight=0` disables that loss but does not change
the colorimetry used for logged metrics.

An optional calibrated linear projection L1 term remains available through
`pipeline.params.spectral_response` (three rows, 31 columns) and `color_weight`;
that auxiliary matrix does **not** replace the CIE/D65 perceptual/metric path.
Keep source/target normalization consistent across all bands (avoid independent
per-image or per-band normalization, which removes the intensity/color
differences the model must learn). The adjacent-band loss assumes equally
spaced bands. Gaussian-only wavelength overrides do not alter colorimetry.
If random-band pretraining transfers poorly, compare contiguous-band masks and
spatial masks separately rather than assuming more masking is better.
