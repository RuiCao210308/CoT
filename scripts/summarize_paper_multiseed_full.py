#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


METRICS = [
    "speed_mae_mps",
    "curvature_mae_x100",
    "overall_l1_train_scale",
    "ADE",
    "FDE",
    "longitudinal_error",
    "lateral_error",
    "final_lateral_error",
]
ALL_RUN_FIELDS = [
    "family",
    "split",
    "method",
    "seed",
    "run_dir",
] + METRICS + [
    "checkpoint_path",
    "eval_json_path",
]
AGG_FIELDS = ["family", "split", "method", "num_runs"] + METRICS


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize multiseed OpenEMMA-OFT paper experiments.")
    parser.add_argument("--output_root", default=os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/openemma_oft_multiseed_full"))
    return parser.parse_args()


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def infer_split(text: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    if config.get("split"):
        return str(config["split"])
    if metrics.get("split"):
        return str(metrics["split"])
    for split in ("100", "300", "850"):
        if re.search(r"(^|_)" + split + r"(_|scenes|$)", text):
            return split
    jsonl = str(metrics.get("jsonl") or config.get("test_jsonl") or "")
    for split in ("100", "300", "850"):
        if split in jsonl:
            return split
    return ""


def infer_seed(text: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    if config.get("seed") not in (None, ""):
        return str(config["seed"])
    if metrics.get("seed") not in (None, ""):
        return str(metrics["seed"])
    match = re.search(r"seed(\d+)", text)
    return match.group(1) if match else ""


def infer_method(text: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    if config.get("method"):
        return str(config["method"])
    if metrics.get("ablation_mode"):
        mode = str(metrics["ablation_mode"])
        if mode == "full":
            return "fusion"
        return mode
    model_type = str(metrics.get("model_type") or "")
    if model_type == "qwen_hidden":
        return "vlm_only"
    if model_type == "gated_geometry_residual_fusion":
        return "gated_graft"
    if model_type == "constant_motion":
        return str(metrics.get("ablation_mode") or "constant_motion")
    if "gated_graft" in text:
        return "gated_graft"
    if "fusion" in text:
        return "fusion"
    return model_type


def infer_family(rel_path: str, config: Dict[str, Any]) -> str:
    if config.get("family"):
        return str(config["family"])
    first = rel_path.split(os.sep, 1)[0]
    if first in ("main", "ablation", "deterministic"):
        return first
    return "unknown"


def find_checkpoint(run_dir: str, metrics: Dict[str, Any]) -> str:
    checkpoint = metrics.get("checkpoint")
    if isinstance(checkpoint, str) and checkpoint:
        return checkpoint
    for filename in ("fusion_action_head.pt", "gated_geometry_residual_fusion_head.pt", "ego_action_head.pt", "action_head.pt"):
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            return path
    return ""


def load_rollout(run_dir: str) -> Dict[str, Optional[float]]:
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
        "ADE": finite_float(all_group.get("rollout_ade")),
        "FDE": finite_float(all_group.get("rollout_fde")),
        "longitudinal_error": finite_float(all_group.get("longitudinal_mae")),
        "lateral_error": finite_float(all_group.get("lateral_mae")),
        "final_lateral_error": finite_float(all_group.get("final_lateral_error")),
    }


def collect_rows(output_root: str) -> List[Dict[str, Any]]:
    rows = []
    for root, _, files in os.walk(output_root):
        if "eval.json" not in files:
            continue
        eval_path = os.path.join(root, "eval.json")
        config_path = os.path.join(root, "config.json")
        metrics = load_json(eval_path)
        if not metrics:
            continue
        config = load_json(config_path) or {}
        rel_path = os.path.relpath(root, output_root)
        text = rel_path.replace(os.sep, "_")
        rollout = load_rollout(root)
        row = {
            "family": infer_family(rel_path, config),
            "split": infer_split(text, metrics, config),
            "method": infer_method(text, metrics, config),
            "seed": infer_seed(text, metrics, config),
            "run_dir": root,
            "speed_mae_mps": finite_float(metrics.get("speed_mae_mps")),
            "curvature_mae_x100": finite_float(metrics.get("curvature_mae_x100")),
            "overall_l1_train_scale": finite_float(metrics.get("overall_l1_train_scale")),
            "ADE": rollout.get("ADE"),
            "FDE": rollout.get("FDE"),
            "longitudinal_error": rollout.get("longitudinal_error"),
            "lateral_error": rollout.get("lateral_error"),
            "final_lateral_error": rollout.get("final_lateral_error"),
            "checkpoint_path": find_checkpoint(root, metrics),
            "eval_json_path": eval_path,
        }
        rows.append(row)
    rows.sort(key=lambda item: (str(item["family"]), str(item["split"]), str(item["method"]), str(item["seed"])))
    return rows


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [value for value in values if finite_float(value) is not None]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    return mean, math.sqrt(variance)


def aggregate_rows(rows: List[Dict[str, Any]], family: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("family") != family:
            continue
        key = (str(row.get("family") or ""), str(row.get("split") or ""), str(row.get("method") or ""))
        grouped[key].append(row)
    output = []
    for key, items in sorted(grouped.items()):
        family_name, split, method = key
        agg = {"family": family_name, "split": split, "method": method, "num_runs": len(items)}
        for metric in METRICS:
            mean, std = mean_std([item.get(metric) for item in items])
            agg[metric] = {"mean": mean, "std": std}
        output.append(agg)
    return output


def write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, dict):
                    mean = value.get("mean")
                    std = value.get("std")
                    flat[field] = "" if mean is None else f"{mean:.6f}±{std:.6f}"
                else:
                    flat[field] = value
            writer.writerow(flat)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def fmt_value(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "–"
    return f"{number:.3f}"


def fmt_mean_std(value: Any) -> str:
    if not isinstance(value, dict):
        return fmt_value(value)
    mean = finite_float(value.get("mean"))
    std = finite_float(value.get("std"))
    if mean is None:
        return "–"
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def write_all_runs_md(path: str, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Multiseed All Runs",
        "",
        "Lower is better for all numeric metrics. Empty rollout metrics mean trajectory rollout analysis was unavailable.",
        "",
        "| family | split | method | seed | speed | curv. x100 | overall | ADE | FDE | long. | lat. | final lat. |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {family} | {split} | {method} | {seed} | {speed} | {curv} | {overall} | {ade} | {fde} | {long} | {lat} | {final_lat} |".format(
                family=row.get("family") or "",
                split=row.get("split") or "",
                method=row.get("method") or "",
                seed=row.get("seed") or "",
                speed=fmt_value(row.get("speed_mae_mps")),
                curv=fmt_value(row.get("curvature_mae_x100")),
                overall=fmt_value(row.get("overall_l1_train_scale")),
                ade=fmt_value(row.get("ADE")),
                fde=fmt_value(row.get("FDE")),
                long=fmt_value(row.get("longitudinal_error")),
                lat=fmt_value(row.get("lateral_error")),
                final_lat=fmt_value(row.get("final_lateral_error")),
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def write_aggregate_md(path: str, title: str, rows: List[Dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "Values are mean±std over available seeds. Lower is better.",
        "",
        "| split | method | runs | speed | curv. x100 | overall | ADE | FDE | long. | lat. | final lat. |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {split} | {method} | {runs} | {speed} | {curv} | {overall} | {ade} | {fde} | {long} | {lat} | {final_lat} |".format(
                split=row.get("split") or "",
                method=row.get("method") or "",
                runs=row.get("num_runs") or 0,
                speed=fmt_mean_std(row.get("speed_mae_mps")),
                curv=fmt_mean_std(row.get("curvature_mae_x100")),
                overall=fmt_mean_std(row.get("overall_l1_train_scale")),
                ade=fmt_mean_std(row.get("ADE")),
                fde=fmt_mean_std(row.get("FDE")),
                long=fmt_mean_std(row.get("longitudinal_error")),
                lat=fmt_mean_std(row.get("lateral_error")),
                final_lat=fmt_mean_std(row.get("final_lateral_error")),
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def write_outputs(output_root: str, rows: List[Dict[str, Any]]) -> None:
    all_csv = os.path.join(output_root, "multiseed_all_runs.csv")
    all_json = os.path.join(output_root, "multiseed_all_runs.json")
    all_md = os.path.join(output_root, "multiseed_all_runs.md")
    write_csv(all_csv, rows, ALL_RUN_FIELDS)
    write_json(all_json, rows)
    write_all_runs_md(all_md, rows)

    for family, title, prefix in (
        ("main", "Main Multiseed Aggregate", "multiseed_main_aggregate"),
        ("ablation", "Attribution Ablation Aggregate", "multiseed_ablation_aggregate"),
        ("deterministic", "Deterministic Baselines", "multiseed_deterministic"),
    ):
        agg = aggregate_rows(rows, family)
        write_csv(os.path.join(output_root, f"{prefix}.csv"), agg, AGG_FIELDS)
        write_json(os.path.join(output_root, f"{prefix}.json"), agg)
        write_aggregate_md(os.path.join(output_root, f"{prefix}.md"), title, agg)


def main():
    args = parse_args()
    rows = collect_rows(args.output_root)
    write_outputs(args.output_root, rows)
    print(f"[MultiseedSummary] rows={len(rows)}")
    print(f"[MultiseedSummary] wrote outputs under: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
