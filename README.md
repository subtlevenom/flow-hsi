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

## NTIRE: MSAB Gaussian U-Net cascades

`python main.py` now selects the 31-band NTIRE dataset, the `usgs_unet` model,
and its dedicated training pipeline. The configuration is in
`.configs\flows\train.yaml`, `.configs\flows\model\usgs_unet.yaml`, and
`.configs\flows\pipeline\usgs_unet.yaml`. Override the NTIRE source/target paths
in `.configs\flows\data\ntire.yaml` for your machine. Inputs and targets must
be paired, consistently scaled, wavelength-aligned HSI cubes, not RGB images.
Training uses aligned 128x128 crops with batch size 4; validation and test
retain full images with batch size 1 to reduce memory use. Keep validation
scenes separate from the final test set when selecting loss weights.

The CM branch retains its convolutional residual U-Nets. Each parameter-branch
U-Net now uses MST MSAB spectral-attention blocks at every encoder, bottleneck,
and decoder level (`hyper_num_blocks: 1` per level). Its cascaded feature state
predicts spatially varying Gaussian gains/offsets that modulate the CM output.
Odd spatial sizes remain supported by explicit skip-size interpolation.
Constructor initialization is preserved: every cascade initially maps its
input to itself.
Older convolution-only `usgs_unet` checkpoints are not architecture-compatible;
use the new experiment directory rather than resuming those weights.

The default 20-epoch run has two phases using the same model and optimizer:

- Epochs 0-4: masked-channel autoencoder pretraining on **training targets only**.
  Independently mask 16 of 31 whole bands per sample to zero and minimize mean
  absolute error only on the hidden bands. This is masked reconstruction, not
  an efficient token-dropping ViT MAE: dense U-Nets still process all bands.
- Epochs 5-19: paired source-to-target color restoration. The objective is
  all-band MAE + 0.1 MRAE + 0.1 spectral cosine distance + 0.1 spectral
  first-difference error. MRAE uses a 1e-3 denominator floor to limit the
  influence of dark bands; the cosine term ignores zero target spectra.

Validation masking is repeatable. `val_masked_mae` tracks reconstruction during
pretraining, while `val_mrae` **always** measures paired source-to-target
restoration, so checkpoint ranking stays comparable across phases. The
learning-rate cosine schedule restarts for fine-tuning. SWA is disabled to
avoid overriding that schedule. `last.ckpt` resumes model, optimizer, scheduler,
and epoch/phase together; use a new experiment name for a fresh run.

```powershell
# Defaults: masked-channel pretraining, then paired fine-tuning.
python main.py
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

For perceptual color fidelity, supply a calibrated **linear** spectral-to-RGB
or spectral-to-XYZ integration matrix as `pipeline.params.spectral_response`
(three rows, 31 columns, including wavelength-bin and illuminant weighting),
then enable `pipeline.params.color_weight`. The pipeline applies that same
matrix to predictions and targets for a differentiable color-projection L1
term. No camera response is guessed and no arbitrary three-band slice is
reported as DeltaE. True DeltaE evaluation additionally requires the correct
illuminant, reference white, and XYZ/Lab conversion.

For nonuniform band spacing, pass physical wavelengths to
`model.params.layers.pyramid.params.wavelengths`; Gaussian modulation will use
their normalized positions. The adjacent-band loss assumes the default evenly
spaced NTIRE bands. Keep source/target normalization consistent across all
bands (avoid independent per-image or per-band normalization, which removes
the intensity/color differences the model must learn). If random-band
pretraining transfers poorly, ablate contiguous-band masks and spatial masks
separately rather than assuming more masking is better.
