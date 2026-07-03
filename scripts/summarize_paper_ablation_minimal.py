#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional


SUMMARY_FIELDS = [
    "split",
    "ablation_mode",
    "seed",
    "speed_mae_mps",
    "curvature_mae_x100",
    "overall_l1_train_scale",
    "ADE",
    "FDE",
    "longitudinal_error",
    "lateral_error",
    "final_lateral_error",
    "checkpoint_path",
    "eval_json_path",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize minimal OpenEMMA-OFT ablation runs.")
    parser.add_argument("--output_root", default=os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/openemma_oft_ablation_minimal"))
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_md", default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def infer_split(path: str, metrics: Dict[str, Any]) -> str:
    if metrics.get("split"):
        return str(metrics["split"])
    text = path.replace(os.sep, "_")
    for split in ("100", "300", "850"):
        if re.search(rf"(^|_){split}(_|scenes)", text):
            return split
    jsonl = str(metrics.get("jsonl") or "")
    for split in ("100", "300", "850"):
        if split in jsonl:
            return split
    return ""


def infer_seed(path: str, metrics: Dict[str, Any]) -> str:
    if metrics.get("seed") is not None:
        return str(metrics["seed"])
    match = re.search(r"seed(\d+)", path)
    return match.group(1) if match else ""


def infer_ablation(path: str, metrics: Dict[str, Any]) -> str:
    if metrics.get("ablation_mode"):
        return str(metrics["ablation_mode"])
    name = os.path.basename(os.path.dirname(path))
    for mode in ("ego_only", "vlm_only", "shuffled_vlm", "zero_history", "constant_motion_last", "constant_motion_mean"):
        if mode in name:
            return mode
    model_type = str(metrics.get("model_type") or "")
    if model_type == "qwen_hidden":
        return "vlm_only"
    return model_type


def find_checkpoint(run_dir: str, metrics: Dict[str, Any]) -> str:
    checkpoint = metrics.get("checkpoint")
    if isinstance(checkpoint, str) and checkpoint:
        return checkpoint
    for filename in ("fusion_action_head.pt", "ego_action_head.pt", "action_head.pt"):
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            return path
    return ""


def load_rollout(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "trajectory_rollout_analysis.json")
    data = load_json(path)
    if not data:
        return {}
    records = data.get("records")
    if not isinstance(records, list) or not records:
        return {}
    groups = records[0].get("groups", {})
    all_group = groups.get("all", {}) if isinstance(groups, dict) else {}
    return {
        "ADE": all_group.get("rollout_ade"),
        "FDE": all_group.get("rollout_fde"),
        "longitudinal_error": all_group.get("longitudinal_mae"),
        "lateral_error": all_group.get("lateral_mae"),
        "final_lateral_error": all_group.get("final_lateral_error"),
    }


def collect_rows(output_root: str) -> List[Dict[str, Any]]:
    rows = []
    for root, _, files in os.walk(output_root):
        if "eval.json" not in files:
            continue
        eval_path = os.path.join(root, "eval.json")
        metrics = load_json(eval_path)
        if not metrics:
            continue
        rollout = load_rollout(root)
        row = {
            "split": infer_split(eval_path, metrics),
            "ablation_mode": infer_ablation(eval_path, metrics),
            "seed": infer_seed(eval_path, metrics),
            "speed_mae_mps": metrics.get("speed_mae_mps"),
            "curvature_mae_x100": metrics.get("curvature_mae_x100"),
            "overall_l1_train_scale": metrics.get("overall_l1_train_scale"),
            "ADE": rollout.get("ADE"),
            "FDE": rollout.get("FDE"),
            "longitudinal_error": rollout.get("longitudinal_error"),
            "lateral_error": rollout.get("lateral_error"),
            "final_lateral_error": rollout.get("final_lateral_error"),
            "checkpoint_path": find_checkpoint(root, metrics),
            "eval_json_path": eval_path,
        }
        rows.append(row)
    rows.sort(key=lambda item: (str(item["split"]), str(item["ablation_mode"]), str(item["seed"])))
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_FIELDS})


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_md(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = [
        "# OpenEMMA-OFT Minimal Ablation Summary",
        "",
        "Lower is better for all numeric metrics. Empty rollout fields mean the corresponding rollout analysis JSON was not found.",
        "",
    ]
    splits = sorted({str(row.get("split") or "unknown") for row in rows})
    for split in splits:
        split_rows = [row for row in rows if str(row.get("split") or "unknown") == split]
        lines.extend(
            [
                f"## {split} scenes",
                "",
                "| ablation | seed | speed | curv. x100 | overall | ADE | FDE | long. | lat. | final lat. |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in split_rows:
            lines.append(
                "| {mode} | {seed} | {speed} | {curv} | {overall} | {ade} | {fde} | {long} | {lat} | {final_lat} |".format(
                    mode=row.get("ablation_mode") or "",
                    seed=row.get("seed") or "",
                    speed=fmt(row.get("speed_mae_mps")),
                    curv=fmt(row.get("curvature_mae_x100")),
                    overall=fmt(row.get("overall_l1_train_scale")),
                    ade=fmt(row.get("ADE")),
                    fde=fmt(row.get("FDE")),
                    long=fmt(row.get("longitudinal_error")),
                    lat=fmt(row.get("lateral_error")),
                    final_lat=fmt(row.get("final_lateral_error")),
                )
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    output_csv = args.output_csv or os.path.join(args.output_root, "ablation_summary.csv")
    output_md = args.output_md or os.path.join(args.output_root, "ablation_summary.md")
    output_json = args.output_json or os.path.join(args.output_root, "ablation_summary.json")
    rows = collect_rows(args.output_root)
    write_csv(output_csv, rows)
    write_md(output_md, rows)
    write_json(output_json, rows)
    print(f"[AblationSummary] rows={len(rows)}")
    print(f"[AblationSummary] wrote csv: {output_csv}")
    print(f"[AblationSummary] wrote md: {output_md}")
    print(f"[AblationSummary] wrote json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
