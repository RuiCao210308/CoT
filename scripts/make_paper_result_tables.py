#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


MODEL_ORDER = [
    "OpenEMMA Text",
    "Ego-only",
    "Qwen-hidden",
    "Fusion",
    "Stage7C",
    "Stage7D",
    "Stage7E",
    "Stage7F",
]
MODEL_ALIASES = {
    "openemma text": "OpenEMMA Text",
    "openemma_text": "OpenEMMA Text",
    "openemma-text": "OpenEMMA Text",
    "text": "OpenEMMA Text",
    "openemma": "OpenEMMA Text",
    "openemma_text_generation": "OpenEMMA Text",
    "ego": "Ego-only",
    "ego_only": "Ego-only",
    "ego-only": "Ego-only",
    "qwen": "Qwen-hidden",
    "qwen_hidden": "Qwen-hidden",
    "qwen-hidden": "Qwen-hidden",
    "qwen-hidden-only": "Qwen-hidden",
    "fusion": "Fusion",
    "egovla_chunk": "Fusion",
    "stage7c": "Stage7C",
    "stage 7c": "Stage7C",
    "predicted_geometry_sequence_fusion": "Stage7C",
    "predicted-geometry-sequence-fusion": "Stage7C",
    "stage7d": "Stage7D",
    "stage 7d": "Stage7D",
    "detached_residual_geometry_sequence_fusion": "Stage7D",
    "detached-residual-geometry-sequence-fusion": "Stage7D",
    "stage7e": "Stage7E",
    "stage 7e": "Stage7E",
    "curvature_only_detached_residual_geometry_sequence_fusion": "Stage7E",
    "curvature-only-detached-residual-geometry-sequence-fusion": "Stage7E",
    "stage7f": "Stage7F",
    "stage 7f": "Stage7F",
    "frozen_fusion_curvature_residual": "Stage7F",
    "frozen-fusion-curvature-residual": "Stage7F",
}
SCENARIO_GROUPS = ["all", "high_curvature_top20", "speed_changing_top20", "steady_low_curvature"]
OVERALL_METRICS = [
    "speed_mae_mps",
    "curvature_mae_x100",
    "overall_l1_train_scale",
    "geometry_sequence_ade",
    "geometry_sequence_fde",
]
SCENARIO_METRICS = ["speed_mae_mps", "curvature_mae_x100", "overall_l1_train_scale"]
ROLLOUT_METRICS = ["rollout_ade", "rollout_fde", "longitudinal_mae", "lateral_mae", "final_lateral_error"]
TEXT_DEGENERACY_COLUMNS = [
    "model",
    "total_records",
    "parse_success",
    "parse_failed",
    "flat_repeated_pair",
    "flat_repeated_pair_rate",
    "reused_history_pairs",
    "reused_history_pairs_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate OpenEMMA-OFT paper result tables.")
    parser.add_argument(
        "--overall_json",
        action="append",
        default=[],
        help="Named overall eval JSON in name:path form. Can be repeated.",
    )
    parser.add_argument("--scenario_json", required=True)
    parser.add_argument("--rollout_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--underline_second", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_named_path(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"expected name:path, got {value}")
    name, path = value.split(":", 1)
    if not name or not path:
        raise ValueError(f"expected name:path, got {value}")
    return name, path


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_model(name: Any) -> str:
    raw = str(name or "").strip()
    key = raw.lower().replace("_", "-")
    if raw in MODEL_ORDER:
        return raw
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    key_underscore = raw.lower().replace("-", "_")
    if key_underscore in MODEL_ALIASES:
        return MODEL_ALIASES[key_underscore]
    return raw


def model_sort_key(name: str) -> Tuple[int, str]:
    if name in MODEL_ORDER:
        return MODEL_ORDER.index(name), name
    return len(MODEL_ORDER), name


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def format_number(value: Any) -> str:
    number = finite_number(value)
    if number is None:
        return ""
    if abs(number) >= 100:
        return f"{number:.2f}"
    return f"{number:.4f}"


def rate(numerator: Any, denominator: Any) -> Optional[float]:
    num = finite_number(numerator)
    den = finite_number(denominator)
    if num is None or den is None or den <= 0:
        return None
    return num / den


def best_ranks(rows: List[Dict[str, Any]], group_keys: Sequence[str], metric: str) -> Dict[Tuple[Any, ...], int]:
    by_value = []
    for idx, row in enumerate(rows):
        value = finite_number(row.get(metric))
        if value is None:
            continue
        key = tuple(row.get(group_key) for group_key in group_keys)
        by_value.append((key, value, idx))

    ranks: Dict[Tuple[Any, ...], int] = {}
    grouped: Dict[Tuple[Any, ...], List[Tuple[float, int]]] = {}
    for key, value, idx in by_value:
        grouped.setdefault(key, []).append((value, idx))
    for key, values in grouped.items():
        values.sort(key=lambda item: item[0])
        distinct_rank = 0
        last_value = None
        for value, idx in values:
            if last_value is None or value != last_value:
                distinct_rank += 1
                last_value = value
            ranks[(idx, metric)] = distinct_rank
    return ranks


def decorated_value(value: Any, rank: Optional[int], fmt: str, underline_second: bool) -> str:
    text = format_number(value) if fmt != "int" else str(value if value is not None else "")
    if not text:
        return text
    if rank == 1:
        return f"**{text}**"
    if rank == 2 and underline_second:
        return f"<u>{text}</u>"
    return text


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def latex_value(value: Any, rank: Optional[int], fmt: str, underline_second: bool) -> str:
    text = format_number(value) if fmt != "int" else str(value if value is not None else "")
    if not text:
        return ""
    if rank == 1:
        return r"\textbf{" + text + "}"
    if rank == 2 and underline_second:
        return r"\underline{" + text + "}"
    return text


def metric_ranks(
    rows: List[Dict[str, Any]],
    metrics: Sequence[str],
    group_keys: Sequence[str],
) -> Dict[Tuple[Any, ...], int]:
    ranks = {}
    for metric in metrics:
        ranks.update(best_ranks(rows, group_keys, metric))
    return ranks


def write_csv(path: str, columns: Sequence[str], rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    path: str,
    columns: Sequence[str],
    rows: List[Dict[str, Any]],
    metric_columns: Sequence[str],
    rank_group_keys: Sequence[str],
    underline_second: bool,
):
    ranks = metric_ranks(rows, metric_columns, rank_group_keys)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---" if column in {"model", "group"} else "---:" for column in columns]) + " |",
    ]
    for idx, row in enumerate(rows):
        values = []
        for column in columns:
            if column in metric_columns:
                values.append(decorated_value(row.get(column), ranks.get((idx, column)), "float", underline_second))
            elif column.endswith("_records") or column in {"parse_success", "parse_failed", "flat_repeated_pair", "reused_history_pairs"}:
                values.append(str(row.get(column, "")))
            elif column.endswith("_rate"):
                value = finite_number(row.get(column))
                values.append("" if value is None else f"{value * 100:.2f}%")
            else:
                values.append(str(row.get(column, "")))
        lines.append("| " + " | ".join(values) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_latex(
    path: str,
    columns: Sequence[str],
    rows: List[Dict[str, Any]],
    metric_columns: Sequence[str],
    rank_group_keys: Sequence[str],
    caption: str,
    label: str,
    underline_second: bool,
):
    ranks = metric_ranks(rows, metric_columns, rank_group_keys)
    align = "l" * sum(1 for column in columns if column in {"model", "group"})
    align += "r" * (len(columns) - len(align))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + latex_escape(caption) + "}",
        r"\label{" + latex_escape(label) + "}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        values = []
        for column in columns:
            if column in metric_columns:
                values.append(latex_value(row.get(column), ranks.get((idx, column)), "float", underline_second))
            elif column.endswith("_rate"):
                value = finite_number(row.get(column))
                values.append("" if value is None else f"{value * 100:.2f}\\%")
            else:
                values.append(latex_escape(row.get(column, "")))
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_table_bundle(
    output_dir: str,
    stem: str,
    columns: Sequence[str],
    rows: List[Dict[str, Any]],
    metric_columns: Sequence[str],
    rank_group_keys: Sequence[str],
    caption: str,
    label: str,
    underline_second: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, f"{stem}.csv"), columns, rows)
    write_markdown(
        os.path.join(output_dir, f"{stem}.md"),
        columns,
        rows,
        metric_columns,
        rank_group_keys,
        underline_second,
    )
    write_latex(
        os.path.join(output_dir, f"{stem}.tex"),
        columns,
        rows,
        metric_columns,
        rank_group_keys,
        caption,
        label,
        underline_second,
    )


def load_overall_rows(named_paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for value in named_paths:
        name, path = parse_named_path(value)
        data = load_json(path)
        model = canonical_model(name or data.get("model_type"))
        row = {"model": model}
        for metric in OVERALL_METRICS:
            row[metric] = data.get(metric)
        rows.append(row)
    rows.sort(key=lambda row: model_sort_key(row["model"]))
    return rows


def load_grouped_report(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} missing records list")
    return records


def grouped_metric_rows(path: str, metrics: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for item in load_grouped_report(path):
        model = canonical_model(item.get("name") or item.get("model"))
        groups = item.get("groups", {})
        if not isinstance(groups, dict):
            continue
        for group in SCENARIO_GROUPS:
            values = groups.get(group, {})
            if not isinstance(values, dict):
                values = {}
            row = {"model": model, "group": group, "count": values.get("count")}
            for metric in metrics:
                row[metric] = values.get(metric)
            rows.append(row)
    rows.sort(key=lambda row: (model_sort_key(row["model"]), SCENARIO_GROUPS.index(row["group"])))
    return rows


def text_degeneracy_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    for item in load_grouped_report(path):
        model = canonical_model(item.get("name") or item.get("model"))
        parse_success = item.get("parse_success")
        flat_count = item.get("flat_repeated_pair")
        reused_count = item.get("reused_history_pairs")
        row = {
            "model": model,
            "total_records": item.get("total_records"),
            "parse_success": parse_success,
            "parse_failed": item.get("parse_failed"),
            "flat_repeated_pair": flat_count,
            "flat_repeated_pair_rate": rate(flat_count, parse_success),
            "reused_history_pairs": reused_count,
            "reused_history_pairs_rate": rate(reused_count, parse_success),
        }
        rows.append(row)
    rows.sort(key=lambda row: model_sort_key(row["model"]))
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    overall_rows = load_overall_rows(args.overall_json)
    scenario_rows = grouped_metric_rows(args.scenario_json, SCENARIO_METRICS)
    rollout_rows = grouped_metric_rows(args.rollout_json, ROLLOUT_METRICS)
    degeneracy_rows = text_degeneracy_rows(args.scenario_json)

    write_table_bundle(
        args.output_dir,
        "table_overall_action_metrics",
        ["model", *OVERALL_METRICS],
        overall_rows,
        OVERALL_METRICS,
        [],
        "Overall action prediction metrics.",
        "tab:overall_action_metrics",
        args.underline_second,
    )
    write_table_bundle(
        args.output_dir,
        "table_scenario_action_metrics",
        ["model", "group", "count", *SCENARIO_METRICS],
        scenario_rows,
        SCENARIO_METRICS,
        ["group"],
        "Scenario-stratified action prediction metrics.",
        "tab:scenario_action_metrics",
        args.underline_second,
    )
    write_table_bundle(
        args.output_dir,
        "table_rollout_metrics",
        ["model", "group", "count", *ROLLOUT_METRICS],
        rollout_rows,
        ROLLOUT_METRICS,
        ["group"],
        "Trajectory rollout metrics.",
        "tab:rollout_metrics",
        args.underline_second,
    )
    write_table_bundle(
        args.output_dir,
        "table_text_degeneracy",
        TEXT_DEGENERACY_COLUMNS,
        degeneracy_rows,
        [],
        [],
        "OpenEMMA text-generation degeneracy diagnostics.",
        "tab:text_degeneracy",
        args.underline_second,
    )

    print(f"[PaperResultTables] wrote tables to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
