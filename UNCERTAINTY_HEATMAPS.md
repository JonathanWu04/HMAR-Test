# Token Uncertainty Heatmaps

这份文档说明 uncertainty heatmap 的运行方式、测试内容、指标计算方式，以及相关代码修改位置。

## 基本用法

运行 heatmap 诊断脚本时，只需要指定模型配置和 checkpoint 权重：

```bash
python -m evaluate.generate_uncertainty_heatmaps \
  --checkpoint=hmar-d16 \
  --checkpoint_path=hmar-d16.pth
```

`--checkpoint` 会选择 `config/evaluate/<checkpoint>.yaml`。例如 `--checkpoint=hmar-d16` 会读取 `config/evaluate/hmar-d16.yaml`。

`--checkpoint_path` 指向要测试的 HMAR `.pth` 权重。如果不传，脚本会默认尝试读取 `<checkpoint>.pth`。

VQ-VAE 权重路径和输出根目录保留为 `utils/sampling_arg_util.py` 里的固定默认值：

```text
vae_path = vae_ch160v4096z32.pth
output_dir = outputs
```

这两个参数正常情况下不需要写进 YAML。如果确实需要改，仍然可以通过命令行覆盖：

```bash
python -m evaluate.generate_uncertainty_heatmaps \
  --checkpoint=hmar-d16 \
  --checkpoint_path=experiments/hmar-train-d16/hmar-d16.pth \
  --vae_path=vae_ch160v4096z32.pth \
  --output_dir=outputs
```

## 配置项

heatmap 相关的运行参数放在对应的 evaluate YAML 中：

- `config/evaluate/hmar-d16.yaml`
- `config/evaluate/hmar-d20.yaml`
- `config/evaluate/hmar-d24.yaml`
- `config/evaluate/hmar-d30.yaml`

主要参数如下：

```yaml
uncertainty_classes: "3_40_80_207"
uncertainty_seeds: "13_14_15"
samples_per_class: 2
diagnostic_topk: 5
diagnostic_scales: ""
overlay_alpha: 0.55
normalize_entropy_per_map: False
```

`uncertainty_classes` 是要测试的 ImageNet class id，可以用 `_` 或 `,` 分隔。

`uncertainty_seeds` 是生成图片使用的随机种子。每个 class 和 seed 组合会生成 `samples_per_class` 张图片。

`diagnostic_scales` 用来限制需要可视化的 scale index。留空表示输出所有 scale。

`overlay_alpha` 控制 heatmap overlay 的透明度。

`normalize_entropy_per_map` 控制 entropy heatmap 是否逐图归一化。为了方便跨 scale 比较，默认保持 `False`。

## 输出结构

默认输出目录为：

```text
outputs/uncertainty/experiment/<timestamp>/<checkpoint-path-label>/
```

例如：

```text
outputs/uncertainty/experiment/20260727-153012/hmar-d16/
```

如果 checkpoint 路径是嵌套路径，例如 `experiments/hmar-train-d16/hmar-d16.pth`，输出目录会类似：

```text
outputs/uncertainty/experiment/20260727-153012/experiments_hmar-train-d16_hmar-d16/
```

母目录中会包含：

- `run_metadata.json`：记录 checkpoint、class、seed、采样参数和诊断参数。
- `log.txt`：记录运行日志和每个 record 的统计摘要。
- `<checkpoint-label>_classXXX_seedY_diagnostics.npz`：保存数值诊断结果。
- `<checkpoint-label>_classXXX_seedY_metadata.json`：保存 scale、pass、refinement step 等元数据。
- `<checkpoint-label>_uncertainty_summary.png`：汇总图，按 class 分组；每一行对应一个 sample；从左到右是 scale 从小到大的 entropy overlay，最右侧是原图。

每张生成图片会单独放在自己的文件夹中，例如：

```text
class003_seed13_sample00/
```

sample 文件夹中包含原图、entropy heatmap、entropy overlay、top-1 probability heatmap 和 top-1 probability overlay。

## 测试内容

HMAR 在生成过程中会在每个 scale 为每个空间 token 位置产生 logits。uncertainty 诊断记录的是 top-k/top-p 截断之前的 logits 分布，因此反映的是模型原始 token 分布的不确定性。

每个被记录的位置会保存以下内容：

- `entropy`：token 分布的熵。值越大，模型越不确定。
- `top1_prob`：概率最高 token 的概率。值越大，模型越确定。
- `selected_prob`：最终采样/选择 token 的概率。
- `top1_top2_margin`：top-1 和 top-2 token 概率差。差距越小，说明选择越不稳定。
- `topk_probs` 和 `topk_indices`：top-k token 的概率和 token id。
- `selected_token`：该位置实际选择的 token id。
- `active_mask` 和 `mask_positions`：如果记录来自 masked refinement step，则保存当前被 mask 的位置。

## 计算方式

对每个 token 位置，先将 logits 转成概率：

```text
p = softmax(logits)
```

entropy 的计算方式为：

```text
entropy = -sum(p * log(p))
```

top-1 probability 是 `p` 中最大的概率值。

selected probability 是模型最终采样到的 token 在 `p` 中对应的概率。

top-1/top-2 margin 的计算方式为：

```text
margin = top1_prob - top2_prob
```

summary 中用于横向比较的主图是 entropy overlay。它只使用每个 scale 的 `next_scale` record，不把 masked refinement 的中间 step 混入 scale 列，方便直接比较从小 scale 到大 scale 的不确定性变化。

## 可视化方式

entropy map 会被映射成 RGB heatmap，再 resize 到生成图片大小。颜色含义如下：

- 蓝色：低不确定性。
- 黄色：中等不确定性。
- 红色：高不确定性。

top-1 probability map 使用红色到绿色的渐变：红色表示低置信度，绿色表示高置信度。

overlay 通过 heatmap 和原图的 alpha blending 得到：

```text
overlay = image * (1 - overlay_alpha) + heatmap * overlay_alpha
```

## 代码修改位置

uncertainty heatmap 功能主要涉及以下文件：

- `models/hmar.py`：`HMAR.generate(...)` 支持 `return_diagnostics=True`。生成 next-scale token 和 masked refinement token 时，会调用 `_build_uncertainty_record(...)` 收集 logits 统计结果。
- `models/hmar.py`：`_build_uncertainty_record(...)` 负责计算 entropy、selected probability、top-k probability、top-1/top-2 margin，以及可选的 mask metadata。
- `evaluate/generate_uncertainty_heatmaps.py`：heatmap 生成入口。负责加载 VQ-VAE 和 HMAR checkpoint，生成样本，保存 `.npz` 数值结果、metadata、单样本 heatmap/overlay，以及母目录下的 summary PNG。
- `evaluate/generate_uncertainty_heatmaps.py`：新增 per-sample 子目录输出和 `save_uncertainty_summary(...)` 汇总图逻辑。
- `utils/sampling_arg_util.py`：保留 CLI/default 参数，包括 `checkpoint_path`、`vae_path`、`output_dir`、class/seed/sample 设置、diagnostic top-k、scale 筛选、overlay alpha 和 entropy 归一化开关。
- `config/evaluate/hmar-d16.yaml`、`config/evaluate/hmar-d20.yaml`、`config/evaluate/hmar-d24.yaml`、`config/evaluate/hmar-d30.yaml`：保存各模型的 evaluate 采样配置和 uncertainty 默认配置。
- `evaluate/inspect_uncertainty_npz.py`：用于检查 `.npz` 诊断文件，并可导出 summary CSV。

默认的 sampling/evaluation 行为不受影响。只有调用 `HMAR.generate(..., return_diagnostics=True)` 时才会额外收集 uncertainty 记录。

## 检查数值结果

可以用 inspector 脚本查看 `.npz` 文件摘要：

```bash
python -m evaluate.inspect_uncertainty_npz \
  --npz=outputs/uncertainty/experiment/<timestamp>/<checkpoint-path-label>/<checkpoint-label>_class003_seed13_diagnostics.npz
```

也可以导出 CSV：

```bash
python -m evaluate.inspect_uncertainty_npz \
  --npz=outputs/uncertainty/experiment/<timestamp>/<checkpoint-path-label>/<checkpoint-label>_class003_seed13_diagnostics.npz \
  --save_csv=outputs/uncertainty/experiment/<timestamp>/<checkpoint-path-label>/class003_seed13_summary.csv
```