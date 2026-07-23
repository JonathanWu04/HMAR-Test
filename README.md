# <p align="center">HMAR: Efficient <u>H</u>ierarchical <u>M</u>asked <u>A</u>uto<u>R</u>egressive Image Generation </p>

<p align="center">
  <b>Hermann Kumbong, Xian Liu, Tsung-Yi Lin, Xihui Liu, Ziwei Liu, Daniel Y Fu, Ming-Yu Liu, Christopher Re, David W. Romero</b>
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2506.04421-b31b1b.svg)](https://arxiv.org/abs/2506.04421)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-nvidia/HMAR-yellow)](https://huggingface.co/nvidia/HMAR)&nbsp;
[![Project Page](https://img.shields.io/badge/Project%20Page-NVIDIA%20Research-76B900.svg)](https://research.nvidia.com/labs/dir/hmar/)&nbsp;
</div>

<figure align="center">
    <img src="assets/samples_banner_border.png" width="95%">
    <figcaption align="center"><b>HMAR Samples</b>: Class-conditional ImageNet generated samples at 256×256 and 512×512 resolutions.</figcaption>
</figure>

## Method Overview

<p align="center">
<img src="assets/method_banner.png" width=95%>
<p>
  
## Install

Ensure `torch>=2.0.0` with CUDA is installed.

```bash
# clone
git clone https://github.com/Kumbong/HMAR
cd HMAR

# install dependencies
pip install -r requirements.txt

# Download the vqvae tokenizer from VAR
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth

# Turn on triton autotuning to ensure kernels are tuned for specific hardware
export TRITON_AUTO_TUNING=1
```

## Training
Prepare the [ImageNet](https://cloud.google.com/tpu/docs/imagenet-setup) dataset. It should be in a path `/path/to/imagenet` with subfolders `train` and `validate`.

Train HMAR-{d16, d20, d24, d30, d36-s} on ImageNet 256x256 or 512x512, for next-scale prediction. 

```bash
# d16, 256x256, for d20, d24, d30 etc, change the experiment accordingly
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py  --experiment=hmar-train-d16 --data_path='/path/to/imagenet'
```
**NOTE**:  We provide training configs in e.g `config/experiment/hmar-train-d16.yaml`.


## Finetuning
 
Introduce masked prediction and combine it with next-scale prediction in HMAR-{d16, d20, d24, d30, d36-s} on ImageNet 256x256 or 512x512.

```bash
# d16, 256x256, for d20, d24, d30 etc, change the experiment accordingly
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py  --experiment=hmar-finetune-mask-d16 --data_path='/path/to/imagenet'
```
**NOTE**:  We provide finetuning configs in e.g `config/experiment/hmar-finetune-mask-d16.yaml`.

## Sampling

We provide a sampling script `sample.py` to generate images with HMAR.

```bash
# 1) you can change the sampling configs from config/sampling/hmar-d30.yaml
# 2) you can change the number of masked sampling steps from utils/sampling_arg_util.py 
python sample.py --checkpoint=hmar-d30
```

## Token Uncertainty Heatmaps

This fork adds a diagnostic mode for visualizing token-level probability
distributions during HMAR sampling. The default `HMAR.generate(...)` behavior is
unchanged. When `return_diagnostics=True` is used, generation also returns
per-scale statistics computed from the logits before top-k/top-p truncation:

- entropy, where higher values mean higher token uncertainty;
- top-1 probability, where higher values mean higher confidence;
- selected token probability;
- top-1/top-2 probability margin;
- top-k token ids and probabilities for each spatial token position.

Generate heatmaps for one checkpoint:

```bash
python -m evaluate.generate_uncertainty_heatmaps \
  --checkpoint=hmar-d16 \
  --uncertainty_classes=3_207 \
  --uncertainty_seeds=13_14 \
  --samples_per_class=4
```

Use an explicit finetuned checkpoint path:

```bash
python -m evaluate.generate_uncertainty_heatmaps \
  --checkpoint=hmar-d16 \
  --checkpoint_path=/path/to/ar-ckpt-last.pth \
  --run_name=original-finetuned \
  --uncertainty_classes=3 \
  --uncertainty_seeds=13 \
  --samples_per_class=4 \
  --output_subdir=baseline/original-finetuned
```

The script follows the same config style as evaluation: `--checkpoint=hmar-d16`
loads `config/evaluate/hmar-d16.yaml` for `depth`, `cfg`, `top_k`, `top_p`,
`mask`, and related sampling settings. Extra diagnostic arguments can be passed
from the command line or added to the same YAML file:

```yaml
uncertainty_classes: "3_207"
uncertainty_seeds: "13_14"
samples_per_class: 4
diagnostic_topk: 5
diagnostic_scales: "7_8_9"
overlay_alpha: 0.55
normalize_entropy_per_map: False
run_name: original-finetuned
output_dir: outputs
output_subdir: baseline/original-finetuned
```

Outputs include:

- generated sample PNGs;
- entropy heatmaps and overlays;
- top-1 probability heatmaps and overlays;
- compressed `.npz` files containing the numeric token diagnostics;
- JSON metadata with scale ids, pass names, sampling settings, classes, and seeds;
- `log.txt` with run settings and per-record summary statistics.

Entropy heatmaps use blue for low uncertainty, yellow for middle values, and red
for high uncertainty. Top-1 probability heatmaps use red for low confidence and
green for high confidence.

By default outputs are written under:

```bash
outputs/uncertainty/<checkpoint>/
```

If `--output_subdir=baseline/original-finetuned` is provided, outputs are written
under:

```bash
outputs/uncertainty/baseline/original-finetuned/
```

Inspect a diagnostics `.npz` file:

```bash
python -m evaluate.inspect_uncertainty_npz \
  --npz=outputs/uncertainty/hmar-d16/hmar-d16_class003_seed13_diagnostics.npz
```

Save the per-record summary as CSV:

```bash
python -m evaluate.inspect_uncertainty_npz \
  --npz=outputs/uncertainty/hmar-d16/hmar-d16_class003_seed13_diagnostics.npz \
  --save_csv=outputs/uncertainty/hmar-d16/class003_seed13_summary.csv
```

## Evaluation
 
To compute FID, Inception Score, Precision and Recall, or to reproduce the numbers from our paper

```bash
# generate 50K samples to be used for evaluation 
python -m evaluate.generate_samples --checkpoint=hmar-d16

# compute FID, IS, precision, recall on the generated samples
python -m evaluate.compute_metrics --checkpoint=hmar-d16
```

## Benchmarking

To benchmark the attention kernels, e2e training and inference speedups, or reproduce the efficiency numbers reported in our paper. 

```bash
# Ensure that triton kernels are tuned for specific hardware
export TRITON_AUTO_TUNING=1

# stand alone attention kernels performance
python -m benchmark.attention --sparsity_pattern="block_diagonal"

# end-to-end training performance 
python -m benchmark.training

# inference performance
python -m benchmark.inference
```

We report numbers on `A100 80Gb SXM4`, `CUDA Version: 12.5` and `triton 3.2.0`

## Acknowledgement
We would like to acknowledge the following projects, from which code in this codebase has been derived:
* [VAR](https://github.com/FoundationVision/VAR) 
* [MaskGIT](https://github.com/google-research/maskgit).

## Citation
```bibtex
 @article{kumbong2024hmar,
            title     = {HMAR: Efficient Hierarchical Masked AutoRegressive Image Generation},
            author    = {Kumbong, Hermann and Liu, Xian and Lin, Tsung-Yi and Liu, Xihui and Liu, Ziwei and Fu, Daniel Y and Liu, Ming-Yu and Re, Christopher and Romero, David W},
            journal   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
            year      = {2025},
            url       = {https://arxiv.org/abs/2506.04421}
          }
```
