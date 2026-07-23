# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the NVIDIA One-Way Noncommercial License v1 (NSCLv1).
# To view a copy of this license, please refer to LICENSE

import argparse
import csv
import json
import os
import re

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and summarize an HMAR uncertainty diagnostics npz file."
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to *_diagnostics.npz.")
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to matching *_metadata.json. If omitted, the script tries to infer it.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Optional path to save per-record summary statistics as CSV.",
    )
    return parser.parse_args()


def infer_metadata_path(npz_path: str) -> str:
    return re.sub(r"_diagnostics\.npz$", "_metadata.json", npz_path)


def load_records(npz_file, metadata_path):
    if metadata_path is not None and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)["records"]

    record_ids = sorted(
        {
            int(match.group(1))
            for key in npz_file.files
            for match in [re.match(r"record(\d+)_", key)]
            if match is not None
        }
    )
    records = []
    for record_id in record_ids:
        entropy_key = f"record{record_id}_entropy"
        pn = npz_file[entropy_key].shape[-1] if entropy_key in npz_file.files else -1
        records.append(
            {
                "record_index": record_id,
                "scale_index": -1,
                "pn": int(pn),
                "pass": "unknown",
                "refinement_step": -1,
            }
        )
    return records


def summarize_record(npz_file, record):
    record_index = int(record["record_index"])
    entropy = npz_file[f"record{record_index}_entropy"]
    top1 = npz_file[f"record{record_index}_top1_prob"]
    selected_prob = npz_file[f"record{record_index}_selected_prob"]
    margin = npz_file[f"record{record_index}_top1_top2_margin"]
    selected_token = npz_file[f"record{record_index}_selected_token"]

    return {
        "record_index": record_index,
        "scale_index": int(record["scale_index"]),
        "pn": int(record["pn"]),
        "pass": record["pass"],
        "refinement_step": int(record["refinement_step"]),
        "num_samples": int(entropy.shape[0]),
        "token_shape": f"{entropy.shape[1]}x{entropy.shape[2]}",
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(np.median(entropy)),
        "entropy_max": float(entropy.max()),
        "top1_prob_mean": float(top1.mean()),
        "selected_prob_mean": float(selected_prob.mean()),
        "top1_top2_margin_mean": float(margin.mean()),
        "frac_top1_below_0_5": float((top1 < 0.5).mean()),
        "frac_margin_below_0_1": float((margin < 0.1).mean()),
        "unique_selected_tokens": int(np.unique(selected_token).shape[0]),
    }


def print_arrays(npz_file):
    rows = [["Array", "Shape", "Dtype"]]
    for key in npz_file.files:
        value = npz_file[key]
        rows.append([key, "x".join(map(str, value.shape)), str(value.dtype)])
    print_table(rows)


def print_summary(rows):
    table = [[
        "rec",
        "scale",
        "pn",
        "pass",
        "step",
        "samples",
        "tokens",
        "H_mean",
        "H_max",
        "top1",
        "sel_prob",
        "margin",
        "top1<0.5",
        "margin<0.1",
        "uniq_tok",
    ]]
    for row in rows:
        table.append(
            [
                row["record_index"],
                row["scale_index"],
                row["pn"],
                row["pass"],
                row["refinement_step"],
                row["num_samples"],
                row["token_shape"],
                f"{row['entropy_mean']:.4f}",
                f"{row['entropy_max']:.4f}",
                f"{row['top1_prob_mean']:.4f}",
                f"{row['selected_prob_mean']:.4f}",
                f"{row['top1_top2_margin_mean']:.4f}",
                f"{row['frac_top1_below_0_5']:.4f}",
                f"{row['frac_margin_below_0_1']:.4f}",
                row["unique_selected_tokens"],
            ]
        )
    print_table(table)


def print_table(rows):
    rows = [[str(item) for item in row] for row in rows]
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    separator = "+".join("-" * (width + 2) for width in widths)
    for row_index, row in enumerate(rows):
        print(" | ".join(item.ljust(widths[col]) for col, item in enumerate(row)))
        if row_index == 0:
            print(separator)


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = parse_args()
    metadata_path = args.metadata or infer_metadata_path(args.npz)
    npz_file = np.load(args.npz)
    records = load_records(npz_file, metadata_path)

    print(f"npz: {args.npz}")
    print(f"metadata: {metadata_path if os.path.exists(metadata_path) else '<not found>'}")
    print_arrays(npz_file)

    rows = [summarize_record(npz_file, record) for record in records]
    print_summary(rows)

    if args.save_csv is not None:
        save_csv(args.save_csv, rows)
        print(f"saved csv: {args.save_csv}")
