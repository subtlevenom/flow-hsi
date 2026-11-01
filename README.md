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

## Spline evaluation performance

Both `SharedCmKAN` and Sprecher now use local-support Cox-de Boor evaluation:
only `spline_order+1` active coefficients are gathered per argument (four for
cubic), instead of materializing every basis coefficient on every edge.
The cubic recurrence is unrolled using the **actual stored knots**, not a
uniform-grid approximation. Other orders retain a local recurrence.
Finite support, knot endpoints, generated-offset/base initialization and model
state names/shapes are preserved; older matching-architecture checkpoints load
unchanged. Numerical rounding may differ at floating-point precision. This
does not alter either model's formula or its activation family.

Bounded CPU measurements (PyTorch 2.4, one thread, batch 1, median of three
timed runs after warmup, float32, default checkpointing/chunking):

| Path | Dense reference | Local evaluator |
|---|---:|---:|
| Shared 31->48->48->31, 32x32, full forward/backward | 9.41 s | 2.36 s |
| Same, parameter application forward/backward | 9.29 s | 2.24 s |
| Same, parameter application inference | 2.86 s | 0.93 s |
| Sprecher 31->31, 8x8, full forward/backward | 6.20 s | 2.24 s |
| Same, parameter application inference | 1.82 s | 0.81 s |

The shared model's MST generator forward alone measured 0.047 s; dense spline
application, not the feature backbone, dominated this CPU experiment.
For an isolated `[64,48,48]` shifted-edge cubic evaluation, saved autograd
tensor payload fell from 56.25 to 33.19 MiB (not process peak memory).
CUDA was unavailable: these are **not GPU throughput predictions** or NTIRE
training/model-quality benchmarks. No active training process was changed.

Local evaluation still has argument/index/gather tensors and launch overhead.
The shared generator still materializes 5,564 values/pixel, including dense W;
local spline evaluation does not eliminate that map. Sprecher still executes
independent scalar blocks in Python and retains all exact branches.
Checkpointing recomputes curves during backward; reducing checkpointing can
improve speed only if memory permits. Chunk size trades launch overhead against
activation memory. Tune these on the actual GPU, not from CPU speed alone.
Existing runs do not hot-reload this implementation; use it for a future run
or an explicitly chosen compatible checkpoint, not a blind resume from an
experiment folder containing multiple architectures.

An **alternative requiring an explicit activation-family change** is a
conditioned Gaussian curve
`g(t)=sum_{r=1}^R a_r(context)*exp(-0.5*((t-mu_r)/sigma_r)^2)`.
With six fixed centers/scales covering the intended domain, only six amplitudes
per curve are generated: a 31-block two-curve Sprecher layer emits 372 values
instead of 620; the three-layer shared model emits 5,552 instead of 5,564
because generated dense weights still dominate. Fixed positive scales avoid
division by tiny values; choose coverage beyond the normalized shifted domain.
Gaussian tails approach zero smoothly rather than having compact support.
If centers/scales must also be generated, six terms require 18 parameters per
curve (1,116 for Sprecher); bound centers to the intended domain and use a
positive lower scale bound plus softplus or bounded sigmoid parameterization.
Normalize/clamp extreme standardized distances before squaring to avoid
overflow, with that intentional tail change documented. A learned linear
baseline can preserve extrapolation at the cost of two additional parameters
per curve. These curves suit context-conditioned local color neighborhoods,
but need coverage/scale regularization and can lose detail or collapse.
Six exponentials may or may not beat four local cubic coefficients on a GPU;
benchmark before switching. No Gaussian family has been substituted here.

### Exact all-block Sprecher batching

Sprecher now evaluates all independent output blocks together by default
(`block_chunk_size: null`). Like the original
[cmKAN implementation](https://github.com/gosha20777/cmKAN/blob/main/cm_kan/ml/layers/cm_kan/cm_kan.py),
it rearranges pixel and curve dimensions for batched tensor operations.
Explicit batched tensor operations handle each block's own knots, generated
Psi/Xi coefficients, Lambda and epsilon; no branches or parameters are tied,
omitted, or reordered. Existing scalar-block state keys and checkpoint shapes
remain unchanged. Batched evaluation works with layer checkpointing, cached
context and pixel chunking. All 63 branches per block remain. Coefficient and
knot gathers flatten arguments sharing the same table rather than expanding
tables over every branch/channel; backward scatters into compact tables.
The existing local Cox-de Boor recurrence still uses actual knots, including
float32 rounding and nonuniform loaded grids. The previous `uniform` and
`shared_shift` options have been removed: there is no curve approximation or
shared epsilon.

RTX 4090 measurements with PyTorch 2.4.0+cu121, batch 1, a 32x32 crop,
31->31 Sprecher layer, full three-stage MST generator and FFN,
bf16-mixed precision, checkpointing enabled, median of three synchronized
samples after warmup:

| Execution | Full forward/backward | Full inference |
|---|---:|---:|
| Original scalar blocks, 64-pixel chunks | 3.047 s | 0.627 s |
| Original evaluator with all 31 blocks batched | 0.455 s | not measured |
| Compact gathers, all 31 blocks batched, 64-pixel chunks | 0.451 s | 0.186 s |

Batching accounts for most of the approximately 6.8x training and 3.4x
inference speedup over the original scalar configuration. These are model
timings, not epoch timings or evidence of a quality change. On a GPU comparison
with independent shifts and nonuniform knots, outputs matched exactly and
the maximum absolute gradient difference was below 5e-7.

The default retains 64-pixel chunks. Float32 full-model measurements with
256/512/1024-pixel chunks took 0.460/0.476/0.503 s for forward/backward and
used approximately 4.2/8.3/16.5 GiB peak allocated memory, versus 0.441 s and
1.1 GiB at 64 pixels. Evaluating all pixels at once was therefore slower and
less memory-efficient. For a lower-memory block batch, set a positive size:

```powershell
python main.py --config-name sprecher task=train model.params.layers.cmkan.params.block_chunk_size=4
```

Per-call inner argument/basis storage scales with pixel and block group sizes,
plus indices and autograd intermediates.
Checkpointing avoids saving all these activations across the image, but
recomputes them during backward. Increasing pixel and block chunks together
can exhaust memory. The 620-values/pixel generated curve map is unchanged.

A separate matched-shape CPU curve microbenchmark (`[64,63,31]` arguments,
six fixed Gaussian centers/scales with generated amplitudes versus local cubic)
measured Gaussian/cubic forward-backward at 0.018/0.030 s and inference at
0.0076/0.0216 s. This is **not** a full Gaussian-network benchmark or evidence
of equivalent approximation quality. Six Gaussians change the curve family;
no activation was replaced. Fixed centers are cheapest but must cover the
shifted spectral domain; context-generated centers can follow the local color
neighborhood, but add channels and can collapse. Bound centers and widths,
keep a strictly positive minimum width, and consider a linear baseline if
linear tails rather than vanishing tails are desired. In either case the exact
Sprecher architecture still needs `n*(2n+1)*m = 60,543` inner evaluations per
pixel. Replacing splines alone cannot eliminate that dominant scaling.
No live training process or automatic resume was changed.

## NTIRE: exact-form Sprecher network

The new `sprecher` run implements the requested scalar block literally:
`f(x) = sum_{q=0}^{2n} Xi(sum_{p=1}^{n} Lambda[p,q] * Psi(x_p + epsilon*q))`.
Inputs use zero-based tensor indexing for the mathematical indices p=1..n;
q includes both endpoints, giving **2n+1 branches**, never a capped
approximation. This is the requested shared-Xi variant, not a claim that every
Sprecher representation in the literature uses the same outer curve.
A block owns exactly **two B-spline curves**: Psi shared across every p/q,
and Xi shared across every q. Lambda is a learned `[n, 2n+1]` matrix
(not generated); one learned epsilon scalar serves every q in that block.
Different blocks have independent Lambda, epsilon and both curve vectors.

A **Sprecher layer** runs m scalar blocks in parallel to map n to m.
`cmkan.SprecherNet` composes layers sequentially, then uses the actual
cmKAN encoder `FFN` from `flows/ml/layers/encoders/cm_encoder.py`:
1x1 projection, 3x3 depthwise convolution, ReLU, 1x1 output projection.
The default is **31 -> 31 -> FFN(32 hidden) -> 31**: one Sprecher layer
with 31 independent scalar blocks, preserving the
HSI-to-HSI NTIRE task. The convolutional FFN and spatial conditioning mean
the complete image network is not solely a pointwise theorem representation.
This single-layer width is an experiment, not a benchmark-selected optimum
or a presumed improvement over the previous two-layer `[8,8]` setting.

MST++ uses the original spectrum as context for **all layers**, not each
intermediate signal. Its feature backbone is the same repository MST adaptation
described below, with one zero-initialized 20-channel head per scalar block:
10 Psi and 10 Xi coefficient offsets, using `grid_size=7`, `spline_order=3`.
Each block has independent learned base coefficients; generated offsets modify
them per conditioned pixel. There are no generated weights/shifts/biases.
`predictor(context)` caches the two curves for each block and applies the same
mapping to another signal on the same spatial grid. Odd sizes, including 1x1,
are supported. No pretrained weights are loaded.

To keep normalized spectra inside meaningful spline support, epsilon is
parameterized as `max_shift * sigmoid(epsilon_logit) / (2n)`, so it is strictly
positive and `epsilon*(2n) < max_shift` (default 0.5; initial total shift 0.25).
Psi's base interval is `[-1,2]`, containing normalized [0,1] inputs plus the
bounded shift; Xi's is `[-2,2]`. Psi initially reproduces identity; Xi initially
reproduces `t/(2n+1)`, and Lambda starts at `1/n`. This scaling is in the
curve coefficients, not an extra averaging operation outside the formula.
The final FFN has its normal initialization: **the full network does not
initialize to identity**. Lambda is unconstrained and intermediate outputs can
leave these intervals during learning. Beyond extended knot support splines
return zero, not extrapolation. Adjust ranges to the data; no implicit clipping
or normalization is inserted. Finite learned splines with a shared Xi do
**not** inherit a universal-representation theorem guarantee.

The 31 default scalar blocks generate **620 values/pixel** in one layer:
float32 maps cost 2.42 MiB at 32x32, 38.75 MiB at 128x128 and about
584 MiB at 482x512. Lambda has 60,543 learned entries plus 31 epsilon scalars
(each block's matrix is `[31,63]`). Exact branches require 60,543 inner
spline evaluations per pixel, about 3.62 times the previous `[8,8]` setting;
all inner 10-coefficient bases would occupy about 2.31 GiB per 32x32 sample
if retained together, before recursion intermediates and outer bases.
Default 64-pixel chunks bound one largest block's inner basis to about 4.8 MiB;
layer checkpointing avoids retaining all these training bases, trading compute
for memory. MST++ activations, output FFN and curve maps still cost memory.
Use smaller validation crops/batches when needed; full-cube inference may be slow.

```powershell
python main.py --config-name sprecher task=train
```

This inherits the existing NTIRE dataset root and training/pipeline defaults
described below, with experiment `cmkan.sprecher.ntire`. The configured root
must exist on the training machine; append `"dataset_root=C:\path\to\NTIRE\dataset"`
for another location. Architecture controls are under
`model.params.layers.cmkan.params`, for example:

```powershell
python main.py --config-name sprecher task=train "model.params.layers.cmkan.params.layer_widths=[8,8,8]"
```

`input_dim` and `output_dim` are configurable in the standalone network;
`spectral_channels=N` updates both and the NTIRE loader/pipeline (the same
non-31 colorimetry opt-outs below apply). `grid_size`, `spline_order`,
`psi_range`, `xi_range`, `max_shift`, `ffn_hidden`, MST widths/stages,
`pixel_chunk_size` and `checkpoint_splines` remain configurable.
The previous `shared_cmkan` run is retained for compatibility, not silently
reinterpreted as a Sprecher block. Synthetic smoke tests do not establish
quality on real NTIRE data.

## NTIRE: conditioned shared-spline cmKAN (previous architecture)

Starting from `usgs-unet`, `cmkan.SharedCmKAN` implements the neuron
`y_j = c_j + sum_i W_ji * psi(x_i + b_j)`. Each layer has **one univariate
B-spline**, shared across its receiving neurons and input edges; `W` is a
full independent edge-weight matrix, and `b` and `c` are receiving-neuron
shift and output-bias vectors. There is no learned input calibration, SiLU
residual, activation after the weighted sum, or output clipping. Successive
layers supply the outer nonlinear transformations. The spline uses the
existing cmKAN knot construction and Cox-de Boor recurrence.

The conditioning generator emits a concatenation of **spline
coefficients, W, b, c for every layer**. A layer with dimensions I -> O
generates `grid_size + spline_order + O*I + 2*O` values per pixel, rather
than independent edge splines. Every layer has its **own generated curve**
(its own coefficient slice and its own base coefficients), shared by all of
that layer's neurons/edges; curves are not tied across layers. Defaults are
`grid_size=7`, `spline_order=3` (cubic) on `grid_range=[0,1]`: 10
coefficients per layer. A learned base parameter vector plus
generated offsets defines each local mapping. Thus a curve is shared within
each layer **at a conditioned pixel**, not forced to be identical across
all image locations. `predictor(context)` caches the local parameter field
for applying the same mapping to another signal on the same batch/spatial
grid, preserving existing cmKAN conditioning semantics.

The default generator (`encoder_type=mst`) adapts **MST++** (Cai et al.,
"MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral
Reconstruction", CVPRW 2022, https://github.com/caiyuanhao1998/MST-plus-plus,
MIT License), reusing this repository's `flows/ml/layers/mst/MST_Plus_Plus.py`
blocks. Instead of the original RGB(3) -> 31-band reconstruction, a 3x3 stem
maps the raw `spectral_channels` input to `mst_features` (32) channels,
`mst_stages` (3) cascaded U-shaped spectral-wise-attention MST blocks with a
global residual form a feature backbone, and a shared 1x1 head
(`mst_head_channels`, 64, GELU) feeds **one zero-initialized 1x1 head per
layer** emitting that layer's full vector. The thousands of generated values
are therefore never MST attention channels. Inputs are replicate-padded to
multiples of four and cropped back, so odd sizes (including 1x1) work. The
previous cmKAN generators remain selectable with `encoder_type=light` (single
wavelet scale) or `encoder_type=full` (two scales); these project the
spectrum to `conditioning_channels` (8) context features, which never replace
the raw neuron inputs.

The default uses three shared-neuron transformations,
`spectral_channels -> 48 -> 48 -> spectral_channels` (31 -> 48 -> 48 -> 31
for NTIRE). This is a cost-conscious engineering starting point, not a
benchmark-validated optimum: the learned splines and conditioning generator
already provide substantial capacity, while every extra or wider layer adds
generated per-pixel dense weights. Per pixel the generator emits
1,594 + 2,410 + 1,560 = **5,564** values.
This float32 parameter field is about 22 MiB for a 32x32 training patch,
348 MiB for a 128x128 validation crop and 5.1 GiB for a full 482x512 cube,
before activations. Learned weights: about 2.09M with `mst` (1.72M backbone +
0.36M heads), 0.28M with `light` or 1.41M with `full`. MST++ attention adds
activation memory and is markedly slower on CPU than `light`. Five
transformations remain selectable, for example
31 -> 48 -> 64 -> 48 -> 32 -> 31 with
`"model.params.layers.cmkan.params.hidden_channels=[48,64,48,32]"`; that emits
10,656 values per pixel (~42 MiB at 32x32, 666 MiB at 128x128, 9.8 GiB at
482x512). Reduce widths or validation/test crops if memory is tight.

The network initializes to identity on [0, 1] using Greville spline
coefficients, rectangular identity W (extra hidden units start at zero), zero
b/c and zero-offset generator heads. All parameters train from scratch; **no Huawei/RGB/MST++ checkpoint
is loaded or frozen**. Padding/cropping handles arbitrary
positive image sizes. Pixel chunking and spline activation checkpointing bound
individual basis evaluations and reduce saved training activations; the full
generated parameter map still costs memory.

This repository's NTIRE task is **paired HSI-to-HSI color correction**, not
RGB-to-HSI reconstruction. Use the existing prepared dataset: paired,
same-named `.npy` files in HWC order, 31 ordered bands at **400:10:700 nm**:

```text
<NTIRE_DATASET_ROOT>
  train/source/<name>.npy
  train/target/<name>.npy
  test/source/<name>.npy
  test/target/<name>.npy
```

The loader checks paired names/shapes, memory-maps cubes, and extracts aligned
random 32x32 training patches with flips. Validation uses deterministic 128x128
center crops to bound the generated parameter field; files must be at least
128 pixels on each spatial axis by default. Set `data.params.val.crop_size=0`
for full-cube validation if memory permits, or a smaller crop for smaller files.
The test split uses full cubes, trimmed to multiples of four by the existing
loader. Float arrays should already use consistent source/target unit-scale
normalization (integer arrays retain the existing dtype scaling). A split's
optional `norm` divides **source only**, not target. This config reuses the
branch's prepared `train`/`test` split convention; it does not download or
guess an NTIRE challenge edition or convert raw challenge files.

After installing the repository dependencies, start training with:

```powershell
python main.py --config-name shared_cmkan task=train
```

The shared cmKAN data config inherits `.configs\flows\data\ntire.yaml`,
including its loader defaults, and uses that config's existing train/validation
dataset root: `/home/korepanov/work/flow-hsi/.data/NTIRE/v8.251224/dataset`.
All shared cmKAN splits use this root. It must exist on the training machine.
To use another location, append `"dataset_root=C:\path\to\NTIRE\dataset"`
(replace the placeholder), or `dataset_root=/your/actual/NTIRE/dataset` on Linux.
Individual `data.params.<split>.source`/`target` overrides remain available.
Defaults are GPU 0,
71 epochs, AdamW at 2e-4 with cosine decay, and paired MAE + floored MRAE +
spectral cosine + adjacent-band gradient + integrated CIE Lab + 0.15 SSIM
loss. Metrics include all-band PSNR/SAM and integrated CIEDE2000.
Checkpoints are ranked by `val_mrae` in
`.experiments\cmkan.shared-spline.ntire\logs\checkpoints`; append `resume=true`
to resume this experiment's `last.ckpt`. Append `accelerator=cpu devices=1`
for CPU execution.

One `spectral_channels=N` override updates model input/output/hidden widths,
the loader's shape check and the pipeline. For a different ordered spectral
grid, explicitly disable the NTIRE-specific color terms with
`pipeline.params.use_ntire_colorimetry=false pipeline.params.perceptual_weight=0`;
31-channel count alone does not establish wavelength/colorimetric compatibility.
The standalone model also accepts unequal input/output dimensions and custom
`hidden_channels`, but the paired NTIRE pipeline requires equal cube shapes.
Configure architecture under `model.params.layers.cmkan.params`, including
`encoder_type`, `mst_features`, `mst_stages`, `mst_head_channels`,
`conditioning_channels`, `grid_size`, `spline_order`, `grid_range`,
`pixel_chunk_size`, and `checkpoint_splines`. B-splines have compact support:
values shifted beyond the extended knot support yield zero spline activation,
not extrapolation; choose a range consistent with dataset normalization.

Windows dependency caveat: the existing `requirements.txt` pins NumPy 2.2.6,
but the torchvision 0.19.0 Windows wheel requires NumPy < 2. This implementation
was exercised in an isolated Python 3.10.12 environment with PyTorch 2.4.0+cpu,
torchvision 0.19.0 and an explicit NumPy 1.26.4 resolver override. The repository
dependency pins are unchanged; use a compatible environment rather than
assuming a clean Windows installation of those pins will resolve.

## NTIRE: shared spectral encoder, two independent experiments

The new encoder-decoder baseline is available through two **separate runs**:

```powershell
# 1. 31 epochs of masked-channel encoder pretraining.
python main.py --config-name pretrain task=train
# 2. 71 epochs of paired NTIRE fine-tuning using the exported encoder.
python main.py --config-name finetune task=train
```

Equivalent flow-group commands are
`python main.py flows@_global_=pretrain task=train` and
`python main.py flows@_global_=finetune task=train`.
Both select NTIRE automatically. Set the paired source/target paths in
`.configs\flows\data\ntire.yaml` for your machine before training. Inputs
are **31-band HSI**, not three-channel RGB; use consistent source/target
normalization and the same ordered wavelengths. Training patch and batch sizes
come from `data.params.train`; validation uses full images. The default device is GPU 0; append
`accelerator=cpu devices=1` for CPU execution.

### Encoder and task heads

The model uses a **GazeTR-inspired, per-pixel receptive-field token transformer**,
not the original image-level gaze regressor. The existing
`ggpir.SpectralEncoderDecoder` factory entrypoint now constructs this model.
The encoder has four **full-resolution** receptive-field branches:

| Branch | Residual-block dilations | Width |
|---|---|---|
| Small | `[1]` | 48 |
| Medium | `[1, 2]` | 96 |
| Large | `[1, 2, 3]` | 128 |
| Extra-large | `[1, 2, 3, 4]` | 160 |
| Independent detail | two dilation-1 blocks | 48 |

A stride-1 3x3 stem feeds all branches. Each residual block has two dense
3x3 convolutions at its listed dilation, GELU between them and a residual
scale of 0.1. Every feature remains on the original pixel grid; larger
receptive fields come from block depth and dilation, not downsampling.
Learned spatial/channel sigmoid gates connect small→medium, small→large,
medium→large, and small/medium/large→extra-large. Gates start with bias -3,
so cross-branch contributions are initially small but trainable. The
independent detail path receives no cross-branch connections.

Encoder residual blocks have two dense 3x3 convolutions with GELU between
them and a residual scale of 0.1. Unlike stock ResNet-18, there is no
max-pooling, aggressive input stride, BatchNorm or image-wide normalization.
Learned pointwise projections adapt the shared stem to each branch width and
adapt gated connections between unequal widths. Pointwise token projections
then align all branch features to 64 token channels. A
separate 31 -> 64 pixel projection preserves the current spectrum. The pixel,
four branch, and detail signals at each location become six tokens, with
learned branch-identity embeddings.
Two four-head query-transformer blocks fuse these tokens independently at
every pixel. The initial query combines the pixel token and detail token;
attention weights therefore depend on the current spectrum and neighborhood.
No attention mixes different output pixels or uses image-wide statistics.
The branch feature vectors are not compressed into candidate output spectra.

Both task heads use two full-resolution gated local blocks and a 64 -> 31
projection. There is **no mandatory spatial bottleneck, absolute coordinate
embedding, Gaussian coefficient head, or output clipping**.
No pooling, resizing, or multiple-of-four padding is used; arbitrary positive
spatial sizes are supported. The largest isolated branch has a nominal
43-pixel receptive field through its four blocks. Hierarchical cross-branch
gates and the two-block task head increase the maximum nominal receptive field
to 87 pixels, so use a conservative 44-pixel input halo when comparing tiled
and full predictions away from true borders. Crop origins need no stride
alignment. This controls convolutional context; it does not guarantee learned
accuracy.

The **pretraining head** predicts all 31 channels directly, without adding the
input to the output. Full-resolution features see only visible values and explicit
missing-band flags; the model masks supplied hidden bands before every path.
Training
alternates source and target cubes between batches, flipping the starting
domain each epoch so even a single-batch dataset uses both domains over time.
Each selected cube is its own reconstruction target, not a source-to-target
translation pair. Training zeroes 16 random whole bands and embeds their explicit
31-band missingness flags into the encoder stem, distinguishing a hidden band
from a genuinely dark one. The objective is **hidden-band MAE + 0.1 visible-band
MAE + 0.15 (1 - SSIM)**. All pretraining epochs do reconstruction; there is no automatic transition to
paired training. This is dense masked-channel autoencoding, not sparse
token-dropping ViT MAE.

The **fine-tuning head** predicts a residual added to the original source HSI.
Small nonzero output weights initialize it near identity without blocking
gradients. It is initialized separately, not transferred as a calibrated-output
mapping from the reconstruction task. The shared encoder includes the branch
extractors, detail path and query transformer.

Activation checkpointing is enabled for local encoder blocks and fusion.
`model.params.layers.spectral.params.pixel_chunk_size=4096` bounds each fusion
operation without cropping the convolutional context; it changes execution,
not the receptive field. `checkpoint_blocks=false` disables recomputation.
Full-resolution training still uses substantial memory; reduce the batch size
if needed. These execution settings do not affect checkpoint compatibility.
For a cheaper fusion ablation, set `fusion_depth=1`; old U-Net options no longer apply.

### Transfer, schedule and resume

Pretraining checkpoints are ranked by deterministic `val_masked_mae`, measured
over target cubes in the entire validation loader with repeatable,
batch-index-seeded masks. Validation remains target-only for comparable
checkpoint selection; training metrics follow the currently selected domain.
The best encoder is exported atomically, on rank zero, to:

```text
.experiments\spectral.pretrain.gazetr.fullrf.wide.ntire\logs\checkpoints\encoder.pt
```

The artifact contains only encoder weights, architecture metadata and selection
metrics, not decoder weights or optimizer state. Fine-tuning requires this
artifact and checks its encoder metadata and state dictionary strictly. Change
`pipeline.params.encoder_checkpoint=...` when pretraining used a different
experiment/output directory. A missing or incompatible artifact fails
explicitly. To deliberately train a random-encoder ablation:

```powershell
python main.py --config-name finetune pipeline.params.encoder_checkpoint=null pipeline.params.allow_random_encoder=true experiment=spectral.scratch.ntire
```

Fine-tuning freezes the encoder, including query fusion, for the first **500 optimizer updates, capped
at the end of epoch 0**, then fine-tunes the complete model. With fewer than
500 updates per epoch, it unfreezes at epoch 1. Gradient accumulation does not
count microbatches as separate optimizer updates. Decoder LR starts at **2e-4**;
encoder LR starts at **2e-5**; both groups are present from the beginning in one
AdamW optimizer, with epoch-wise cosine decay and no optimizer reset.
`pipeline.params.freeze_encoder_steps=0` disables the frozen warmup.
Pretraining uses LR 2e-4 for both encoder and its lightweight head.

Fine-tuning retains all-band MAE, floored MRAE, spectral cosine, adjacent-band
difference and CIE Lab perceptual losses described below and adds
**0.15 (1 - SSIM)** plus a strong spatial sharpness term; checkpoints are
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
Fine-tuning also uses `pipeline.params.sharpness_weight=1.0` to match
horizontal and vertical finite differences across all 31 bands. This
high-frequency edge loss complements SSIM and is distinct from
`spectral_gradient_weight`, which compares adjacent wavelengths at each pixel.
It is logged as `sharpness_loss` / `val_sharpness_loss`; set its weight to 0
to disable it. Pretraining is unchanged.
Checkpoint selection remains masked MAE for pretraining and MRAE for restoration.
SAM logs now use degrees, including CSV logs; do not compare new values
directly with older radian logs (multiply old values by 180/pi).

The runs have separate output directories: `spectral.pretrain.gazetr.fullrf.wide.ntire` and
`spectral.finetune.gazetr.fullrf.wide.ntire`.
Each resumes its own `last.ckpt` by default, including
optimizer, scheduler, epoch and update count. Full fine-tuning resume does not
reload the initial encoder artifact, so that artifact is no longer required.
Use a new `experiment=...` for a fresh run or architecture change. Existing
cascade checkpoints cannot initialize this encoder. Masked pretraining is an
experiment, not a demonstrated accuracy gain; compare against the scratch
ablation on held-out scenes.

**Restart both pretraining and fine-tuning for this model.** Encoder metadata
is now version 6 and identifies the full-resolution dilated query transformer; older encoder artifacts and
full checkpoints are rejected explicitly. Historical experiment files are not
renamed or deleted, and the new directories avoid accidental old-checkpoint
resume. The public model head and pipeline phase are now `finetune`, and all
spectral experiment/config commands use `finetune` rather than the old name.

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
