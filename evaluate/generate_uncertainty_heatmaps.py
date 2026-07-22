# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of the license, please refer to LICENSE.

import argparse
import json
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.utils import save_image

from models import build_vae_hmar
from utils.arg_util import _compile_model, _get_yaml_loader, _seed_everything, _set_tf32


DEFAULT_SAMPLING_ARGS = {
    "checkpoint": "hmar-d16",
    "vfast": 0,
    "tfast": 0,
    "depth": 16,
    "saln": False,
    "anorm": True,
    "fuse": True,
    "pn": "1_2_3_4_5_6_8_10_13_16",
    "patch_size": 16,
    "tf32": True,
    "seed": 42,
    "cfg": 1.5,
    "top_k": 900,
    "top_p": 0.96,
    "more_smooth": False,
    "mask": True,
    "mask_schedule": [[1], [1, 3], [1, 1, 2, 5], [16], [25], [36], [64], [100], [169], [256]],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HMAR token uncertainty heatmaps from pre-sampling logits."
    )
    parser.add_argument("--checkpoint", default="hmar-d16", help="Primary checkpoint name or path.")
    parser.add_argument("--checkpoint_path", default=None, help="Optional explicit primary checkpoint path.")
    parser.add_argument(
        "--config_folder",
        default="evaluate",
        choices=("evaluate", "sample"),
        help="Config folder used for cfg/top-k/top-p/mask settings.",
    )
    parser.add_argument(
        "--config_name",
        default="hmar-d16",
        help="Config file basename, e.g. hmar-d16 reads config/evaluate/hmar-d16.yaml.",
    )
    parser.add_argument("--vae_path", default="vae_ch160v4096z32.pth")
    parser.add_argument("--output_dir", default="uncertainty_heatmaps")
    parser.add_argument("--classes", nargs="+", type=int, default=[3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13])
    parser.add_argument("--samples_per_class", type=int, default=4)
    parser.add_argument("--diagnostic_topk", type=int, default=5)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=None,
        help="Optional scale indices to save. By default all HMAR scales are saved.",
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument(
        "--normalize_entropy_per_map",
        action="store_true",
        help="Use each map's own entropy min/max instead of [0, log(vocab_size)].",
    )
    return parser.parse_args()


def load_sampling_args(config_folder, config_name):
    values = dict(DEFAULT_SAMPLING_ARGS)
    config_path = os.path.join("config", config_folder, f"{config_name}.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=_get_yaml_loader())
        values.update(config)
    values["patch_nums"] = tuple(map(int, values["pn"].replace("-", "_").split("_")))
    _set_tf32(values["tf32"])
    _seed_everything(values["seed"], benchmark=True)
    return SimpleNamespace(**values)


def resolve_checkpoint_path(name, explicit_path=None):
    if explicit_path:
        return explicit_path
    if os.path.exists(name):
        return name
    return f"{name}.pth"


def safe_label(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(value))[0])


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_hmar_state_dict(checkpoint):
    state = checkpoint
    if isinstance(checkpoint, dict):
        if "trainer" in checkpoint and "transformer_wo_ddp" in checkpoint["trainer"]:
            state = checkpoint["trainer"]["transformer_wo_ddp"]
        elif "transformer_wo_ddp" in checkpoint:
            state = checkpoint["transformer_wo_ddp"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        cleaned[key] = value
    return cleaned


def build_model(args, cli_args, checkpoint_path):
    vae_local, hmar = build_vae_hmar(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device="cuda",
        patch_nums=args.patch_nums,
        num_classes=1000,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
    )
    vae_local.load_state_dict(torch_load(cli_args.vae_path), strict=True)
    vae_local = _compile_model(vae_local, args.vfast)
    hmar = _compile_model(hmar, args.tfast)

    state_dict = extract_hmar_state_dict(torch_load(checkpoint_path))
    missing, unexpected = hmar.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warning] missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[warning] unexpected checkpoint keys: {len(unexpected)}")
    hmar.eval()
    return hmar


def to_uint8_image(tensor_chw):
    array = tensor_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def normalize_map(values, vmin, vmax):
    values = values.astype(np.float32)
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def entropy_colormap(norm):
    norm = np.asarray(norm, dtype=np.float32)
    blue = np.array([18, 52, 139], dtype=np.float32)
    yellow = np.array([255, 221, 79], dtype=np.float32)
    red = np.array([213, 45, 45], dtype=np.float32)
    lower = norm <= 0.5
    out = np.empty((*norm.shape, 3), dtype=np.float32)
    out[lower] = blue + (yellow - blue) * (norm[lower][..., None] / 0.5)
    out[~lower] = yellow + (red - yellow) * ((norm[~lower][..., None] - 0.5) / 0.5)
    return np.clip(out, 0, 255).astype(np.uint8)


def confidence_colormap(confidence):
    confidence = np.clip(confidence.astype(np.float32), 0.0, 1.0)
    red = np.array([213, 45, 45], dtype=np.float32)
    green = np.array([38, 150, 83], dtype=np.float32)
    out = red + (green - red) * confidence[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_rgb(rgb, size):
    return np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))


def overlay(image_rgb, heat_rgb, alpha):
    return np.clip(image_rgb * (1.0 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)


def save_rgb(path, array):
    Image.fromarray(array).save(path)


def save_record_images(output_dir, label, class_id, seed, sample_index, image_rgb, record, cli_args):
    scale_index = int(record["scale_index"])
    if cli_args.scales is not None and scale_index not in cli_args.scales:
        return

    stem = (
        f"{label}_class{class_id:03d}_seed{seed}_sample{sample_index:02d}_"
        f"scale{scale_index:02d}_pn{int(record['pn']):02d}_{record['pass']}_step{record['refinement_step']}"
    )
    entropy = record["entropy"][sample_index].numpy()
    top1_prob = record["top1_prob"][sample_index].numpy()
    image_size = (image_rgb.shape[1], image_rgb.shape[0])

    if cli_args.normalize_entropy_per_map:
        entropy_vmin = float(entropy.min())
        entropy_vmax = float(entropy.max())
    else:
        entropy_vmin = 0.0
        entropy_vmax = math.log(4096)

    entropy_rgb = resize_rgb(
        entropy_colormap(normalize_map(entropy, entropy_vmin, entropy_vmax)), image_size
    )
    confidence_rgb = resize_rgb(confidence_colormap(top1_prob), image_size)

    save_rgb(os.path.join(output_dir, f"{stem}_entropy.png"), entropy_rgb)
    save_rgb(
        os.path.join(output_dir, f"{stem}_entropy_overlay.png"),
        overlay(image_rgb, entropy_rgb, cli_args.overlay_alpha),
    )
    save_rgb(os.path.join(output_dir, f"{stem}_top1_prob.png"), confidence_rgb)
    save_rgb(
        os.path.join(output_dir, f"{stem}_top1_prob_overlay.png"),
        overlay(image_rgb, confidence_rgb, cli_args.overlay_alpha),
    )


def diagnostics_to_npz(output_dir, label, class_id, seed, images, diagnostics):
    arrays = {}
    records = []
    for record_index, record in enumerate(diagnostics):
        records.append(
            {
                "record_index": record_index,
                "scale_index": int(record["scale_index"]),
                "pn": int(record["pn"]),
                "pass": record["pass"],
                "refinement_step": int(record["refinement_step"]),
            }
        )
        for key, value in record.items():
            if torch.is_tensor(value):
                arrays[f"record{record_index}_{key}"] = value.numpy()
    arrays["images"] = images.detach().cpu().numpy()
    np.savez_compressed(
        os.path.join(output_dir, f"{label}_class{class_id:03d}_seed{seed}_diagnostics.npz"),
        **arrays,
    )
    with open(
        os.path.join(output_dir, f"{label}_class{class_id:03d}_seed{seed}_metadata.json"),
        "w",
    ) as f:
        json.dump({"records": records}, f, indent=2)


def run_checkpoint(label, checkpoint_path, sampling_args, cli_args):
    hmar = build_model(sampling_args, cli_args, checkpoint_path)
    results = {}
    with torch.inference_mode():
        for class_id in cli_args.classes:
            for seed in cli_args.seeds:
                output = hmar.generate(
                    cli_args.samples_per_class,
                    class_id,
                    g_seed=seed,
                    cfg=sampling_args.cfg,
                    top_k=sampling_args.top_k,
                    top_p=sampling_args.top_p,
                    more_smooth=sampling_args.more_smooth,
                    mask=sampling_args.mask,
                    mask_schedule=sampling_args.mask_schedule,
                    return_diagnostics=True,
                    diagnostic_topk=cli_args.diagnostic_topk,
                )
                images, diagnostics = output
                images = images.detach().cpu()
                results[(class_id, seed)] = (images, diagnostics)
    del hmar
    torch.cuda.empty_cache()
    return results


def main():
    cli_args = parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)
    sampling_args = load_sampling_args(cli_args.config_folder, cli_args.config_name)

    primary_path = resolve_checkpoint_path(cli_args.checkpoint, cli_args.checkpoint_path)
    primary_label = safe_label(cli_args.checkpoint)
    primary_results = run_checkpoint(primary_label, primary_path, sampling_args, cli_args)

    run_metadata = {
        "checkpoint": cli_args.checkpoint,
        "checkpoint_path": primary_path,
        "config_folder": cli_args.config_folder,
        "config_name": cli_args.config_name,
        "classes": cli_args.classes,
        "seeds": cli_args.seeds,
        "samples_per_class": cli_args.samples_per_class,
        "cfg": sampling_args.cfg,
        "top_k": sampling_args.top_k,
        "top_p": sampling_args.top_p,
        "mask": sampling_args.mask,
        "mask_schedule": sampling_args.mask_schedule,
    }
    with open(os.path.join(cli_args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)

    for (class_id, seed), (images, diagnostics) in primary_results.items():
        diagnostics_to_npz(cli_args.output_dir, primary_label, class_id, seed, images, diagnostics)
        for sample_index in range(images.shape[0]):
            image_rgb = to_uint8_image(images[sample_index])
            sample_stem = f"{primary_label}_class{class_id:03d}_seed{seed}_sample{sample_index:02d}"
            save_image(images[sample_index], os.path.join(cli_args.output_dir, f"{sample_stem}.png"))
            for record in diagnostics:
                save_record_images(
                    cli_args.output_dir,
                    primary_label,
                    class_id,
                    seed,
                    sample_index,
                    image_rgb,
                    record,
                    cli_args,
                )


if __name__ == "__main__":
    main()
