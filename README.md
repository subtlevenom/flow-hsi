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

## NTIRE: CM U-Net cascades with an optional Gaussian hypernetwork

`python main.py` now selects the 31-band NTIRE dataset, the `usgs_unet` model,
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

Both training and validation progress bars show PSNR and `de` (CIEDE2000).
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
