# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import datetime
import json
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

from models import HMAR
from utils.sampling_arg_util import Args, get_args


def parse_int_list(value: str) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if isinstance(value, int):
        return [value]
    if value is None or len(str(value).strip()) == 0:
        return []
    value = str(value).replace(",", "_").replace("-", "_")
    return [int(item) for item in value.split("_") if item != ""]


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(value))[0])


def safe_path_label(value: str) -> str:
    normalized = os.path.splitext(os.path.normpath(value))[0]
    normalized = normalized.strip(os.sep).replace(os.sep, "_")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("_")
    return label or safe_label(value)


def resolve_checkpoint_path(args: Args) -> str:
    if args.checkpoint_path is not None and len(args.checkpoint_path):
        return args.checkpoint_path
    if os.path.exists(args.checkpoint):
        return args.checkpoint
    return f"{args.checkpoint}.pth"


def torch_load(path: str, map_location="cpu"):
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


def get_output_dir(args: Args, checkpoint_path: str) -> str:
    if args.output_subdir is not None and len(args.output_subdir):
        return os.path.join(args.output_dir, "uncertainty", args.output_subdir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        args.output_dir,
        "uncertainty",
        "experiment",
        timestamp,
        safe_path_label(checkpoint_path),
    )


def log_line(output_dir: str, message: str):
    timestamp = datetime.datetime.now().strftime("[%m-%d %H:%M:%S]")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(line + "\n")


def log_json(output_dir: str, tag: str, payload: dict):
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(f"{tag}: {json.dumps(payload, sort_keys=True)}\n")


def build_everything(args: Args):
    from models import VQVAE, build_vae_hmar

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

    vae_local.load_state_dict(torch_load(args.vae_path), strict=True)
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    hmar: HMAR = args.compile_model(hmar, args.tfast)

    checkpoint_path = resolve_checkpoint_path(args)
    state_dict = extract_hmar_state_dict(torch_load(checkpoint_path))
    missing, unexpected = hmar.load_state_dict(state_dict, strict=False)
    hmar.eval()

    return hmar, checkpoint_path, missing, unexpected


def to_uint8_image(tensor_chw: torch.Tensor) -> np.ndarray:
    array = tensor_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def normalize_map(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = values.astype(np.float32)
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def entropy_colormap(norm: np.ndarray) -> np.ndarray:
    norm = np.asarray(norm, dtype=np.float32)
    blue = np.array([18, 52, 139], dtype=np.float32)
    yellow = np.array([255, 221, 79], dtype=np.float32)
    red = np.array([213, 45, 45], dtype=np.float32)
    lower = norm <= 0.5
    out = np.empty((*norm.shape, 3), dtype=np.float32)
    out[lower] = blue + (yellow - blue) * (norm[lower][..., None] / 0.5)
    out[~lower] = yellow + (red - yellow) * ((norm[~lower][..., None] - 0.5) / 0.5)
    return np.clip(out, 0, 255).astype(np.uint8)


def confidence_colormap(confidence: np.ndarray) -> np.ndarray:
    confidence = np.clip(confidence.astype(np.float32), 0.0, 1.0)
    red = np.array([213, 45, 45], dtype=np.float32)
    green = np.array([38, 150, 83], dtype=np.float32)
    out = red + (green - red) * confidence[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_rgb(rgb: np.ndarray, size) -> np.ndarray:
    return np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))


def overlay(image_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(image_rgb * (1.0 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)


def save_rgb(path: str, array: np.ndarray):
    Image.fromarray(array).save(path)


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def record_is_enabled(record: dict, scales: Optional[List[int]]) -> bool:
    return scales is None or int(record["scale_index"]) in scales


def summarize_record(record: dict, sample_index: int) -> dict:
    entropy = record["entropy"][sample_index].numpy()
    top1 = record["top1_prob"][sample_index].numpy()
    selected_prob = record["selected_prob"][sample_index].numpy()
    margin = record["top1_top2_margin"][sample_index].numpy()
    return {
        "record_index": int(record["record_index"]),
        "scale_index": int(record["scale_index"]),
        "pn": int(record["pn"]),
        "pass": record["pass"],
        "refinement_step": int(record["refinement_step"]),
        "sample_index": int(sample_index),
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(np.median(entropy)),
        "entropy_max": float(entropy.max()),
        "top1_prob_mean": float(top1.mean()),
        "selected_prob_mean": float(selected_prob.mean()),
        "top1_top2_margin_mean": float(margin.mean()),
        "frac_top1_below_0_5": float((top1 < 0.5).mean()),
        "frac_margin_below_0_1": float((margin < 0.1).mean()),
    }


def add_record_indices(diagnostics: List[dict]) -> List[dict]:
    for record_index, record in enumerate(diagnostics):
        record["record_index"] = record_index
    return diagnostics


def build_record_visuals(
    image_rgb: np.ndarray, record: dict, sample_index: int, args: Args
) -> Dict[str, np.ndarray]:
    entropy = record["entropy"][sample_index].numpy()
    top1_prob = record["top1_prob"][sample_index].numpy()
    image_size = (image_rgb.shape[1], image_rgb.shape[0])

    if args.normalize_entropy_per_map:
        entropy_vmin = float(entropy.min())
        entropy_vmax = float(entropy.max())
    else:
        entropy_vmin = 0.0
        entropy_vmax = math.log(4096)

    entropy_rgb = resize_rgb(
        entropy_colormap(normalize_map(entropy, entropy_vmin, entropy_vmax)), image_size
    )
    confidence_rgb = resize_rgb(confidence_colormap(top1_prob), image_size)

    return {
        "entropy": entropy_rgb,
        "entropy_overlay": overlay(image_rgb, entropy_rgb, args.overlay_alpha),
        "top1_prob": confidence_rgb,
        "top1_prob_overlay": overlay(image_rgb, confidence_rgb, args.overlay_alpha),
    }


def save_record_images(
    output_dir: str,
    label: str,
    class_id: int,
    seed: int,
    sample_index: int,
    image_rgb: np.ndarray,
    record: dict,
    args: Args,
):
    stem = (
        f"{label}_class{class_id:03d}_seed{seed}_sample{sample_index:02d}_"
        f"scale{int(record['scale_index']):02d}_pn{int(record['pn']):02d}_"
        f"{record['pass']}_step{int(record['refinement_step'])}"
    )
    visuals = build_record_visuals(image_rgb, record, sample_index, args)

    save_rgb(os.path.join(output_dir, f"{stem}_entropy.png"), visuals["entropy"])
    save_rgb(
        os.path.join(output_dir, f"{stem}_entropy_overlay.png"),
        visuals["entropy_overlay"],
    )
    save_rgb(os.path.join(output_dir, f"{stem}_top1_prob.png"), visuals["top1_prob"])
    save_rgb(
        os.path.join(output_dir, f"{stem}_top1_prob_overlay.png"),
        visuals["top1_prob_overlay"],
    )
    return visuals


def sample_output_dir(output_dir: str, class_id: int, seed: int, sample_index: int) -> str:
    return os.path.join(
        output_dir, f"class{class_id:03d}_seed{seed}_sample{sample_index:02d}"
    )


def is_scale_summary_record(record: dict) -> bool:
    return record["pass"] == "next_scale" and int(record["refinement_step"]) == 0


def resize_summary_tile(array: np.ndarray, tile_size: int) -> Image.Image:
    return Image.fromarray(array).resize((tile_size, tile_size), Image.Resampling.BILINEAR)


def save_uncertainty_summary(output_dir: str, label: str, summary_rows: List[dict]) -> Optional[str]:
    if len(summary_rows) == 0:
        return None

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (row["class_id"], row["seed"], row["sample_index"]),
    )
    scale_columns = sorted(
        {
            (scale_index, payload["pn"])
            for row in summary_rows
            for scale_index, payload in row["overlays"].items()
        },
        key=lambda item: item[0],
    )
    if len(scale_columns) == 0:
        return None

    tile_size = min(
        summary_rows[0]["original"].shape[0], summary_rows[0]["original"].shape[1]
    )
    label_width = 210
    gap = 12
    margin = 24
    title_height = 48
    header_height = 34
    group_height = 34
    row_gap = 16

    row_count = len(summary_rows)
    class_count = len({row["class_id"] for row in summary_rows})
    column_count = len(scale_columns) + 1
    width = margin * 2 + label_width + column_count * tile_size + (column_count - 1) * gap
    height = (
        margin * 2
        + title_height
        + header_height
        + class_count * group_height
        + row_count * tile_size
        + max(0, row_count - 1) * row_gap
    )

    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24, bold=True)
    header_font = load_font(16, bold=True)
    body_font = load_font(15)
    small_font = load_font(13)

    y = margin
    draw.text(
        (margin, y),
        f"Token uncertainty heatmap overlays - {label}",
        fill=(25, 25, 25),
        font=title_font,
    )
    y += title_height

    x = margin + label_width
    for scale_index, pn in scale_columns:
        draw.text(
            (x, y),
            f"scale{scale_index:02d} / pn{pn:02d}",
            fill=(40, 40, 40),
            font=header_font,
        )
        x += tile_size + gap
    draw.text((x, y), "original", fill=(40, 40, 40), font=header_font)
    y += header_height

    current_class_id = None
    for row in summary_rows:
        if row["class_id"] != current_class_id:
            current_class_id = row["class_id"]
            draw.rectangle(
                (margin, y, width - margin, y + group_height - 8),
                fill=(232, 235, 232),
            )
            draw.text(
                (margin + 10, y + 6),
                f"class{current_class_id:03d}",
                fill=(25, 25, 25),
                font=header_font,
            )
            y += group_height

        row_label = f"seed{row['seed']}\nsample{row['sample_index']:02d}"
        draw.text(
            (margin + 10, y + tile_size // 2 - 18),
            row_label,
            fill=(45, 45, 45),
            font=body_font,
        )

        x = margin + label_width
        for scale_index, _ in scale_columns:
            payload = row["overlays"].get(scale_index)
            if payload is None:
                draw.rectangle(
                    (x, y, x + tile_size, y + tile_size),
                    outline=(210, 210, 210),
                    width=1,
                )
                draw.text(
                    (x + 12, y + 12),
                    "missing",
                    fill=(120, 120, 120),
                    font=small_font,
                )
            else:
                canvas.paste(resize_summary_tile(payload["image"], tile_size), (x, y))
            x += tile_size + gap

        canvas.paste(resize_summary_tile(row["original"], tile_size), (x, y))
        y += tile_size + row_gap

    summary_path = os.path.join(output_dir, f"{label}_uncertainty_summary.png")
    canvas.save(summary_path)
    return summary_path


def diagnostics_to_npz(output_dir: str, label: str, class_id: int, seed: int, images: torch.Tensor, diagnostics: List[dict]):
    arrays = {}
    records = []
    for record in diagnostics:
        record_index = int(record["record_index"])
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


if __name__ == "__main__":
    args: Args = get_args(cfg_folder="evaluate")
    torch.set_default_device("cuda")

    resolved_checkpoint_path = resolve_checkpoint_path(args)
    output_dir = get_output_dir(args, resolved_checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, "log.txt")):
        os.remove(os.path.join(output_dir, "log.txt"))

    checkpoint_label = safe_path_label(resolved_checkpoint_path)
    classes = parse_int_list(args.uncertainty_classes)
    seeds = parse_int_list(args.uncertainty_seeds)
    scales = parse_int_list(args.diagnostic_scales)
    scales = scales if len(scales) else None

    log_line(output_dir, f"[uncertainty] output_dir={output_dir}")
    log_line(output_dir, f"[uncertainty] checkpoint={args.checkpoint}")
    log_line(output_dir, f"[uncertainty] classes={classes}, seeds={seeds}, samples_per_class={args.samples_per_class}")
    log_line(output_dir, f"[uncertainty] cfg={args.cfg}, top_k={args.top_k}, top_p={args.top_p}, mask={args.mask}")
    log_line(output_dir, f"[uncertainty] patch_nums={args.patch_nums}, diagnostic_topk={args.diagnostic_topk}, scales={scales}")

    hmar, checkpoint_path, missing, unexpected = build_everything(args)
    log_line(output_dir, f"[uncertainty] checkpoint_path={checkpoint_path}")
    log_line(output_dir, f"[uncertainty] vae_path={args.vae_path}")
    if missing:
        log_line(output_dir, f"[warning] missing checkpoint keys: {len(missing)}")
    if unexpected:
        log_line(output_dir, f"[warning] unexpected checkpoint keys: {len(unexpected)}")

    run_metadata = {
        "checkpoint": args.checkpoint,
        "checkpoint_path": checkpoint_path,
        "checkpoint_label": checkpoint_label,
        "vae_path": args.vae_path,
        "output_dir": output_dir,
        "classes": classes,
        "seeds": seeds,
        "samples_per_class": args.samples_per_class,
        "cfg": args.cfg,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "mask": args.mask,
        "mask_schedule": args.mask_schedule,
        "patch_nums": args.patch_nums,
        "diagnostic_topk": args.diagnostic_topk,
        "diagnostic_scales": scales,
        "overlay_alpha": args.overlay_alpha,
        "normalize_entropy_per_map": args.normalize_entropy_per_map,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)
    log_json(output_dir, "[run_metadata]", run_metadata)

    summary_rows = []

    with torch.inference_mode():
        for class_id in classes:
            for seed in seeds:
                log_line(output_dir, f"[generate] class={class_id}, seed={seed}")
                images, diagnostics = hmar.generate(
                    args.samples_per_class,
                    class_id,
                    g_seed=seed,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    more_smooth=args.more_smooth,
                    mask=args.mask,
                    mask_schedule=args.mask_schedule,
                    return_diagnostics=True,
                    diagnostic_topk=args.diagnostic_topk,
                )
                images = images.detach().cpu()
                diagnostics = add_record_indices(diagnostics)
                diagnostics_to_npz(output_dir, checkpoint_label, class_id, seed, images, diagnostics)

                for sample_index in range(images.shape[0]):
                    image_rgb = to_uint8_image(images[sample_index])
                    sample_stem = f"{checkpoint_label}_class{class_id:03d}_seed{seed}_sample{sample_index:02d}"
                    sample_dir = sample_output_dir(output_dir, class_id, seed, sample_index)
                    os.makedirs(sample_dir, exist_ok=True)
                    save_image(images[sample_index], os.path.join(sample_dir, f"{sample_stem}.png"))
                    summary_overlays = {}
                    for record in diagnostics:
                        if not record_is_enabled(record, scales):
                            continue
                        visuals = save_record_images(
                            sample_dir,
                            checkpoint_label,
                            class_id,
                            seed,
                            sample_index,
                            image_rgb,
                            record,
                            args,
                        )
                        if is_scale_summary_record(record):
                            summary_overlays[int(record["scale_index"])] = {
                                "pn": int(record["pn"]),
                                "image": visuals["entropy_overlay"],
                            }
                        log_json(
                            output_dir,
                            "[record_summary]",
                            {
                                "class_id": int(class_id),
                                "seed": int(seed),
                                **summarize_record(record, sample_index),
                            },
                        )
                    summary_rows.append(
                        {
                            "class_id": int(class_id),
                            "seed": int(seed),
                            "sample_index": int(sample_index),
                            "original": image_rgb,
                            "overlays": summary_overlays,
                        }
                    )

    summary_path = save_uncertainty_summary(output_dir, checkpoint_label, summary_rows)
    if summary_path is not None:
        log_line(output_dir, f"[summary] saved {summary_path}")

    log_line(output_dir, "[uncertainty] finished")
