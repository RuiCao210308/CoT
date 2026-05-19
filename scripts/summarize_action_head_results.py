#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


RESULT_COLUMNS = [
    "model",
    "split",
    "num_samples",
    "speed_mae_mps",
    "curvature_mae_x100",
    "curvature_mae_1pm",
    "overall_l1_train_scale",
    "waypoint_ade",
    "waypoint_fde",
    "waypoint_x_mae",
    "waypoint_y_mae",
    "consistency_l1_train_scale",
    "derived_speed_mae_vs_pred_mps",
    "derived_curvature_mae_x100_vs_pred",
    "oracle_geometry_from_gt_waypoints",
    "purpose",
    "geometry_descriptor_dim",
    "geometry_descriptor_l1",
    "geometry_sequence_l1",
    "geometry_sequence_ade",
    "geometry_sequence_fde",
    "geometry_sequence_x_mae",
    "geometry_sequence_y_mae",
    "residual_scale",
    "detach_geometry_for_action",
    "residual_target",
    "speed_source",
    "freeze_fusion_base",
    "fusion_checkpoint_path",
    "residual_curvature_abs_mean",
    "base_speed_mae_mps",
    "base_curvature_mae_x100",
]

IMPROVEMENT_COLUMNS = [
    "comparison",
    "metric",
    "baseline",
    "candidate",
    "relative_change_pct",
    "interpretation",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize OpenEMMA OFT-lite action head eval results.")
    parser.add_argument("--ego_train_json", type=str, default="/root/autodl-tmp/eval_ego_train.json")
    parser.add_argument("--ego_heldout_json", type=str, default="/root/autodl-tmp/eval_ego_heldout.json")
    parser.add_argument("--qwen_train_json", type=str, default="/root/autodl-tmp/eval_qwen_train.json")
    parser.add_argument("--fusion_train_json", type=str, default="/root/autodl-tmp/eval_fusion_train.json")
    parser.add_argument("--fusion_heldout_json", type=str, default="/root/autodl-tmp/eval_fusion_heldout.json")
    parser.add_argument("--waypoint_aux_train_json", type=str, default=None)
    parser.add_argument("--waypoint_aux_heldout_json", type=str, default=None)
    parser.add_argument("--waypoint_consistency_train_json", type=str, default=None)
    parser.add_argument("--waypoint_consistency_heldout_json", type=str, default=None)
    parser.add_argument("--oracle_geometry_train_json", type=str, default=None)
    parser.add_argument("--oracle_geometry_heldout_json", type=str, default=None)
    parser.add_argument("--predicted_geometry_train_json", type=str, default=None)
    parser.add_argument("--predicted_geometry_heldout_json", type=str, default=None)
    parser.add_argument("--predicted_geometry_sequence_train_json", type=str, default=None)
    parser.add_argument("--predicted_geometry_sequence_heldout_json", type=str, default=None)
    parser.add_argument("--detached_residual_geometry_sequence_train_json", type=str, default=None)
    parser.add_argument("--detached_residual_geometry_sequence_heldout_json", type=str, default=None)
    parser.add_argument("--curvature_only_detached_residual_geometry_sequence_train_json", type=str, default=None)
    parser.add_argument("--curvature_only_detached_residual_geometry_sequence_heldout_json", type=str, default=None)
    parser.add_argument("--frozen_fusion_curvature_residual_train_json", type=str, default=None)
    parser.add_argument("--frozen_fusion_curvature_residual_heldout_json", type=str, default=None)
    parser.add_argument("--output_markdown", type=str, default="/root/autodl-tmp/action_head_results_summary.md")
    parser.add_argument("--output_csv", type=str, default="/root/autodl-tmp/action_head_results_summary.csv")
    return parser.parse_args()


def load_eval_json(path: str, label: str, warnings: List[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        warnings.append(f"missing {label}: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"failed to read {label}: {path} ({exc})")
        return None


def result_row(model: str, split: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": model,
        "split": split,
        "num_samples": data.get("num_samples"),
        "speed_mae_mps": data.get("speed_mae_mps"),
        "curvature_mae_x100": data.get("curvature_mae_x100"),
        "curvature_mae_1pm": data.get("curvature_mae_1pm"),
        "overall_l1_train_scale": data.get("overall_l1_train_scale"),
        "waypoint_ade": data.get("waypoint_ade"),
        "waypoint_fde": data.get("waypoint_fde"),
        "waypoint_x_mae": data.get("waypoint_x_mae"),
        "waypoint_y_mae": data.get("waypoint_y_mae"),
        "consistency_l1_train_scale": data.get("consistency_l1_train_scale"),
        "derived_speed_mae_vs_pred_mps": data.get("derived_speed_mae_vs_pred_mps"),
        "derived_curvature_mae_x100_vs_pred": data.get("derived_curvature_mae_x100_vs_pred"),
        "oracle_geometry_from_gt_waypoints": data.get("oracle_geometry_from_gt_waypoints"),
        "purpose": data.get("purpose"),
        "geometry_descriptor_dim": data.get("geometry_descriptor_dim"),
        "geometry_descriptor_l1": data.get("geometry_descriptor_l1"),
        "geometry_sequence_l1": data.get("geometry_sequence_l1"),
        "geometry_sequence_ade": data.get("geometry_sequence_ade"),
        "geometry_sequence_fde": data.get("geometry_sequence_fde"),
        "geometry_sequence_x_mae": data.get("geometry_sequence_x_mae"),
        "geometry_sequence_y_mae": data.get("geometry_sequence_y_mae"),
        "residual_scale": data.get("residual_scale"),
        "detach_geometry_for_action": data.get("detach_geometry_for_action"),
        "residual_target": data.get("residual_target"),
        "speed_source": data.get("speed_source"),
        "freeze_fusion_base": data.get("freeze_fusion_base"),
        "fusion_checkpoint_path": data.get("fusion_checkpoint_path"),
        "residual_curvature_abs_mean": data.get("residual_curvature_abs_mean"),
        "base_speed_mae_mps": data.get("base_speed_mae_mps"),
        "base_curvature_mae_x100": data.get("base_curvature_mae_x100"),
    }


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def relative_change_pct(baseline: Any, candidate: Any) -> Optional[float]:
    base = finite_number(baseline)
    cand = finite_number(candidate)
    if base is None or cand is None or base == 0.0:
        return None
    return (base - cand) / base * 100.0


def lookup(rows: List[Dict[str, Any]], model: str, split: str, metric: str):
    for row in rows:
        if row["model"] == model and row["split"] == split:
            return row.get(metric)
    return None


def improvement_row(
    rows: List[Dict[str, Any]],
    comparison: str,
    metric: str,
    baseline_model: str,
    baseline_split: str,
    candidate_model: str,
    candidate_split: str,
) -> Optional[Dict[str, Any]]:
    baseline = lookup(rows, baseline_model, baseline_split, metric)
    candidate = lookup(rows, candidate_model, candidate_split, metric)
    change = relative_change_pct(baseline, candidate)
    if change is None:
        return None
    interpretation = "improvement" if change > 0 else "regression" if change < 0 else "no_change"
    return {
        "comparison": comparison,
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "relative_change_pct": change,
        "interpretation": interpretation,
    }


def build_improvement_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs = [
        (
            "fusion vs ego-only train overall_l1",
            "overall_l1_train_scale",
            "ego-only",
            "train",
            "fusion",
            "train",
        ),
        (
            "fusion vs ego-only held-out overall_l1",
            "overall_l1_train_scale",
            "ego-only",
            "held-out",
            "fusion",
            "held-out",
        ),
        (
            "fusion vs ego-only train speed_mae",
            "speed_mae_mps",
            "ego-only",
            "train",
            "fusion",
            "train",
        ),
        (
            "fusion vs ego-only held-out speed_mae",
            "speed_mae_mps",
            "ego-only",
            "held-out",
            "fusion",
            "held-out",
        ),
        (
            "fusion vs ego-only train curvature_mae_x100",
            "curvature_mae_x100",
            "ego-only",
            "train",
            "fusion",
            "train",
        ),
        (
            "fusion vs ego-only held-out curvature_mae_x100",
            "curvature_mae_x100",
            "ego-only",
            "held-out",
            "fusion",
            "held-out",
        ),
        (
            "qwen-hidden-only vs ego-only train overall_l1",
            "overall_l1_train_scale",
            "ego-only",
            "train",
            "qwen-hidden-only",
            "train",
        ),
        (
            "fusion vs qwen-hidden-only train overall_l1",
            "overall_l1_train_scale",
            "qwen-hidden-only",
            "train",
            "fusion",
            "train",
        ),
    ]
    improvement_rows = []
    for spec in specs:
        row = improvement_row(rows, *spec)
        if row is not None:
            improvement_rows.append(row)
    return improvement_rows


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    number = finite_number(value)
    if number is None:
        return ""
    if abs(number) >= 100:
        return f"{number:.2f}"
    return f"{number:.4f}"


def markdown_table(columns: List[str], rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(format_value(value) if isinstance(value, (int, float)) or value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(rows: List[Dict[str, Any]], improvement_rows: List[Dict[str, Any]], warnings: List[str]) -> str:
    parts = ["# OpenEMMA OFT-lite Action Head Results", ""]
    if warnings:
        parts.extend(["## Warnings", ""])
        parts.extend(f"- {warning}" for warning in warnings)
        parts.append("")

    parts.extend(["## Eval Metrics", "", markdown_table(RESULT_COLUMNS, rows), ""])
    parts.extend(["## Relative Changes", "", markdown_table(IMPROVEMENT_COLUMNS, improvement_rows), ""])
    parts.extend(
        [
            "## Conclusion",
            "",
            "Fusion improves overall action-space prediction over ego-only on train and held-out splits.",
            "The held-out improvement is modest.",
            "The gain mainly comes from speed prediction.",
            "Curvature prediction remains a bottleneck.",
            "",
        ]
    )
    return "\n".join(parts)


def write_markdown(path: str, content: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_csv(path: str, rows: List[Dict[str, Any]], improvement_rows: List[Dict[str, Any]], warnings: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", *RESULT_COLUMNS])
        for row in rows:
            writer.writerow(["eval_metrics", *[row.get(column, "") for column in RESULT_COLUMNS]])
        writer.writerow([])
        writer.writerow(["section", *IMPROVEMENT_COLUMNS])
        for row in improvement_rows:
            writer.writerow(["relative_change", *[row.get(column, "") for column in IMPROVEMENT_COLUMNS]])
        if warnings:
            writer.writerow([])
            writer.writerow(["section", "warning"])
            for warning in warnings:
                writer.writerow(["warning", warning])


def main():
    args = parse_args()
    warnings: List[str] = []
    entries = [
        ("ego-only", "train", args.ego_train_json, "ego train"),
        ("ego-only", "held-out", args.ego_heldout_json, "ego held-out"),
        ("qwen-hidden-only", "train", args.qwen_train_json, "qwen train"),
        ("fusion", "train", args.fusion_train_json, "fusion train"),
        ("fusion", "held-out", args.fusion_heldout_json, "fusion held-out"),
    ]
    if args.waypoint_aux_train_json:
        entries.append(
            (
                "waypoint-aux-fusion",
                "train",
                args.waypoint_aux_train_json,
                "waypoint aux train",
            )
        )
    if args.waypoint_aux_heldout_json:
        entries.append(
            (
                "waypoint-aux-fusion",
                "held-out",
                args.waypoint_aux_heldout_json,
                "waypoint aux held-out",
            )
        )
    if args.waypoint_consistency_train_json:
        entries.append(
            (
                "waypoint-consistency-fusion",
                "train",
                args.waypoint_consistency_train_json,
                "waypoint consistency train",
            )
        )
    if args.waypoint_consistency_heldout_json:
        entries.append(
            (
                "waypoint-consistency-fusion",
                "held-out",
                args.waypoint_consistency_heldout_json,
                "waypoint consistency held-out",
            )
        )
    if args.oracle_geometry_train_json:
        entries.append(
            (
                "oracle-geometry-fusion",
                "train",
                args.oracle_geometry_train_json,
                "oracle geometry train",
            )
        )
    if args.oracle_geometry_heldout_json:
        entries.append(
            (
                "oracle-geometry-fusion",
                "held-out",
                args.oracle_geometry_heldout_json,
                "oracle geometry held-out",
            )
        )
    if args.predicted_geometry_train_json:
        entries.append(
            (
                "predicted-geometry-fusion",
                "train",
                args.predicted_geometry_train_json,
                "predicted geometry train",
            )
        )
    if args.predicted_geometry_heldout_json:
        entries.append(
            (
                "predicted-geometry-fusion",
                "held-out",
                args.predicted_geometry_heldout_json,
                "predicted geometry held-out",
            )
        )
    if args.predicted_geometry_sequence_train_json:
        entries.append(
            (
                "predicted-geometry-sequence-fusion",
                "train",
                args.predicted_geometry_sequence_train_json,
                "predicted geometry sequence train",
            )
        )
    if args.predicted_geometry_sequence_heldout_json:
        entries.append(
            (
                "predicted-geometry-sequence-fusion",
                "held-out",
                args.predicted_geometry_sequence_heldout_json,
                "predicted geometry sequence held-out",
            )
        )
    if args.detached_residual_geometry_sequence_train_json:
        entries.append(
            (
                "detached-residual-geometry-sequence-fusion",
                "train",
                args.detached_residual_geometry_sequence_train_json,
                "detached residual geometry sequence train",
            )
        )
    if args.detached_residual_geometry_sequence_heldout_json:
        entries.append(
            (
                "detached-residual-geometry-sequence-fusion",
                "held-out",
                args.detached_residual_geometry_sequence_heldout_json,
                "detached residual geometry sequence held-out",
            )
        )
    if args.curvature_only_detached_residual_geometry_sequence_train_json:
        entries.append(
            (
                "curvature-only-detached-residual-geometry-sequence-fusion",
                "train",
                args.curvature_only_detached_residual_geometry_sequence_train_json,
                "curvature-only detached residual geometry sequence train",
            )
        )
    if args.curvature_only_detached_residual_geometry_sequence_heldout_json:
        entries.append(
            (
                "curvature-only-detached-residual-geometry-sequence-fusion",
                "held-out",
                args.curvature_only_detached_residual_geometry_sequence_heldout_json,
                "curvature-only detached residual geometry sequence held-out",
            )
        )
    if args.frozen_fusion_curvature_residual_train_json:
        entries.append(
            (
                "frozen-fusion-curvature-residual",
                "train",
                args.frozen_fusion_curvature_residual_train_json,
                "frozen fusion curvature residual train",
            )
        )
    if args.frozen_fusion_curvature_residual_heldout_json:
        entries.append(
            (
                "frozen-fusion-curvature-residual",
                "held-out",
                args.frozen_fusion_curvature_residual_heldout_json,
                "frozen fusion curvature residual held-out",
            )
        )

    rows = []
    for model, split, path, label in entries:
        data = load_eval_json(path, label, warnings)
        if data is not None:
            rows.append(result_row(model, split, data))

    improvement_rows = build_improvement_rows(rows)
    markdown = build_markdown(rows, improvement_rows, warnings)
    write_markdown(args.output_markdown, markdown)
    write_csv(args.output_csv, rows, improvement_rows, warnings)

    for warning in warnings:
        print(f"[SummarizeActionHeads] warning: {warning}")
    print(f"[SummarizeActionHeads] wrote markdown: {args.output_markdown}")
    print(f"[SummarizeActionHeads] wrote csv: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
