#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


PLOT_MODELS = [
    ("openemma_text", "OpenEMMA text"),
    ("fusion", "Fusion"),
    ("stage7d", "Stage 7D"),
    ("stage7e", "Stage 7E"),
]
MODEL_ALIASES = {
    "openemma_text": {"openemma_text", "openemma", "text", "openemma-text"},
    "fusion": {"fusion", "egovla_chunk", "egovla-chunk"},
    "stage7d": {"stage7d", "7d", "detached_residual_geometry_sequence_fusion"},
    "stage7e": {"stage7e", "7e", "curvature_only_detached_residual_geometry_sequence_fusion"},
    "qwen_hidden": {"qwen_hidden", "qwen-hidden", "qwen"},
    "ego_only": {"ego_only", "ego-only", "ego"},
}
COLORS = {
    "gt": "#111111",
    "openemma_text": "#D55E00",
    "fusion": "#0072B2",
    "stage7d": "#CC79A7",
    "stage7e": "#009E73",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize qualitative trajectory rollout cases.")
    parser.add_argument(
        "--samples_jsonl",
        default="/root/autodl-tmp/action_eval_trajectory_rollout_samples.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/openemma_oft_figures/trajectory_cases",
    )
    parser.add_argument("--min_abs_improvement", type=float, default=0.5)
    parser.add_argument("--min_rel_improvement", type=float, default=0.1)
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def canonical_model(name: Any) -> str:
    value = str(name or "").strip().lower()
    for canonical, aliases in MODEL_ALIASES.items():
        if value in aliases:
            return canonical
    return value


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def sample_key(row: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    token = row.get("sample_token")
    if token:
        return ("token", token, None)
    scene = row.get("scene_name")
    frame = row.get("frame_idx")
    if scene is not None and frame is not None:
        return ("scene_frame", scene, frame)
    return ("record", row.get("record_index"), row.get("line_no"))


def group_samples(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, Any, Any], Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    for row in rows:
        model = canonical_model(row.get("model"))
        key = sample_key(row)
        entry = grouped.setdefault(
            key,
            {
                "key": key,
                "scene_name": row.get("scene_name"),
                "frame_idx": row.get("frame_idx"),
                "sample_token": row.get("sample_token"),
                "models": {},
                "target_abs_curvature_x100": finite_number(row.get("target_abs_curvature_x100")),
                "target_speed_change": finite_number(row.get("target_speed_change")),
            },
        )
        if entry.get("scene_name") is None:
            entry["scene_name"] = row.get("scene_name")
        if entry.get("frame_idx") is None:
            entry["frame_idx"] = row.get("frame_idx")
        if entry.get("sample_token") is None:
            entry["sample_token"] = row.get("sample_token")
        if entry.get("target_abs_curvature_x100") is None:
            entry["target_abs_curvature_x100"] = finite_number(row.get("target_abs_curvature_x100"))
        if entry.get("target_speed_change") is None:
            entry["target_speed_change"] = finite_number(row.get("target_speed_change"))
        entry["models"][model] = row
    return grouped


def metric(sample: Dict[str, Any], model: str, key: str = "rollout_ade") -> Optional[float]:
    row = sample["models"].get(model)
    if row is None:
        return None
    return finite_number(row.get(key))


def improvement(sample: Dict[str, Any], baseline: str, candidate: str) -> Optional[float]:
    base = metric(sample, baseline)
    cand = metric(sample, candidate)
    if base is None or cand is None:
        return None
    return base - cand


def clearly_better(sample: Dict[str, Any], baseline: str, candidate: str, min_abs: float, min_rel: float) -> bool:
    base = metric(sample, baseline)
    gain = improvement(sample, baseline, candidate)
    if base is None or gain is None:
        return False
    return gain >= min_abs and gain / max(base, 1e-6) >= min_rel


def quantile_threshold(values: List[float], q: float) -> Optional[float]:
    values = sorted(values)
    if not values:
        return None
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def select_best(
    samples: List[Dict[str, Any]],
    predicate,
    score_fn,
) -> Optional[Dict[str, Any]]:
    candidates = [sample for sample in samples if predicate(sample)]
    if not candidates:
        return None
    candidates.sort(key=lambda sample: score_fn(sample), reverse=True)
    return candidates[0]


def select_cases(samples_by_key: Dict[Tuple[Any, Any, Any], Dict[str, Any]], min_abs: float, min_rel: float):
    samples = list(samples_by_key.values())
    curv_values = [
        value
        for value in (finite_number(sample.get("target_abs_curvature_x100")) for sample in samples)
        if value is not None
    ]
    speed_values = [
        value
        for value in (finite_number(sample.get("target_speed_change")) for sample in samples)
        if value is not None
    ]
    high_curv_threshold = quantile_threshold(curv_values, 0.8)
    low_curv_threshold = quantile_threshold(curv_values, 0.2)
    high_speed_threshold = quantile_threshold(speed_values, 0.8)
    low_speed_threshold = quantile_threshold(speed_values, 0.2)

    def has_models(sample, names):
        return all(name in sample["models"] for name in names)

    selected = {}
    selected["high_curvature"] = select_best(
        samples,
        lambda sample: has_models(sample, ["openemma_text", "stage7e"])
        and (high_curv_threshold is None or sample.get("target_abs_curvature_x100", -1) >= high_curv_threshold)
        and clearly_better(sample, "openemma_text", "stage7e", min_abs, min_rel),
        lambda sample: improvement(sample, "openemma_text", "stage7e") or -1e9,
    )
    selected["speed_changing"] = select_best(
        samples,
        lambda sample: has_models(sample, ["openemma_text", "stage7d"])
        and (high_speed_threshold is None or sample.get("target_speed_change", -1) >= high_speed_threshold)
        and clearly_better(sample, "openemma_text", "stage7d", min_abs, min_rel),
        lambda sample: improvement(sample, "openemma_text", "stage7d") or -1e9,
    )
    selected["steady_low_curvature"] = select_best(
        samples,
        lambda sample: has_models(sample, ["openemma_text", "fusion", "stage7e"])
        and (low_curv_threshold is None or sample.get("target_abs_curvature_x100", 1e9) <= low_curv_threshold)
        and (low_speed_threshold is None or sample.get("target_speed_change", 1e9) <= low_speed_threshold)
        and clearly_better(sample, "fusion", "openemma_text", min_abs, min_rel)
        and clearly_better(sample, "stage7e", "openemma_text", min_abs, min_rel),
        lambda sample: min(
            improvement(sample, "fusion", "openemma_text") or -1e9,
            improvement(sample, "stage7e", "openemma_text") or -1e9,
        ),
    )
    selected["fusion_advantage"] = select_best(
        samples,
        lambda sample: has_models(sample, ["fusion", "qwen_hidden", "ego_only"])
        and clearly_better(sample, "qwen_hidden", "fusion", min_abs, min_rel)
        and clearly_better(sample, "ego_only", "fusion", min_abs, min_rel),
        lambda sample: min(
            improvement(sample, "qwen_hidden", "fusion") or -1e9,
            improvement(sample, "ego_only", "fusion") or -1e9,
        ),
    )

    fallback_specs = {
        "high_curvature": (
            lambda sample: has_models(sample, ["openemma_text", "stage7e"])
            and (high_curv_threshold is None or sample.get("target_abs_curvature_x100", -1) >= high_curv_threshold),
            lambda sample: improvement(sample, "openemma_text", "stage7e") or -1e9,
        ),
        "speed_changing": (
            lambda sample: has_models(sample, ["openemma_text", "stage7d"])
            and (high_speed_threshold is None or sample.get("target_speed_change", -1) >= high_speed_threshold),
            lambda sample: improvement(sample, "openemma_text", "stage7d") or -1e9,
        ),
        "steady_low_curvature": (
            lambda sample: has_models(sample, ["openemma_text", "fusion", "stage7e"])
            and (low_curv_threshold is None or sample.get("target_abs_curvature_x100", 1e9) <= low_curv_threshold)
            and (low_speed_threshold is None or sample.get("target_speed_change", 1e9) <= low_speed_threshold),
            lambda sample: min(
                improvement(sample, "fusion", "openemma_text") or -1e9,
                improvement(sample, "stage7e", "openemma_text") or -1e9,
            ),
        ),
        "fusion_advantage": (
            lambda sample: has_models(sample, ["fusion", "qwen_hidden", "ego_only"]),
            lambda sample: min(
                improvement(sample, "qwen_hidden", "fusion") or -1e9,
                improvement(sample, "ego_only", "fusion") or -1e9,
            ),
        ),
    }
    for group, sample in list(selected.items()):
        if sample is None:
            predicate, score_fn = fallback_specs[group]
            selected[group] = select_best(samples, predicate, score_fn)
    return selected


def as_points(value: Any) -> Optional[List[List[float]]]:
    if not isinstance(value, list) or not value:
        return None
    points = []
    for row in value:
        if not isinstance(row, list) or len(row) < 2:
            return None
        x = finite_number(row[0])
        y = finite_number(row[1])
        if x is None or y is None:
            return None
        points.append([x, y])
    return points


def row_points(row: Dict[str, Any]) -> Optional[List[List[float]]]:
    return as_points(row.get("pred_rollout_waypoints"))


def gt_points(sample: Dict[str, Any]) -> Optional[List[List[float]]]:
    for row in sample["models"].values():
        points = as_points(row.get("gt_waypoints"))
        if points:
            return points
    return None


def selected_case_record(group: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    models = {}
    for model, row in sample["models"].items():
        models[model] = {
            "rollout_ade": row.get("rollout_ade"),
            "rollout_fde": row.get("rollout_fde"),
            "lateral_mae": row.get("lateral_mae"),
            "final_lateral_error": row.get("final_lateral_error"),
        }
    return {
        "group": group,
        "sample_token": sample.get("sample_token"),
        "scene_name": sample.get("scene_name"),
        "frame_idx": sample.get("frame_idx"),
        "target_abs_curvature_x100": sample.get("target_abs_curvature_x100"),
        "target_speed_change": sample.get("target_speed_change"),
        "models": models,
    }


def plot_case(group: str, sample: Dict[str, Any], output_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    gt = gt_points(sample)
    if gt:
        ax.plot(
            [point[0] for point in gt],
            [point[1] for point in gt],
            color=COLORS["gt"],
            linewidth=2.4,
            marker="o",
            markersize=3.2,
            label="GT",
        )

    for model, label in PLOT_MODELS:
        row = sample["models"].get(model)
        if row is None:
            continue
        points = row_points(row)
        if not points:
            continue
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=COLORS.get(model),
            linewidth=1.9,
            marker=".",
            markersize=4,
            label=label,
        )

    ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("forward local x (m)")
    ax.set_ylabel("lateral local y (m)")
    ax.set_title(group.replace("_", " ").title(), fontsize=11)
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    png_path = os.path.join(output_dir, f"{group}.png")
    pdf_path = os.path.join(output_dir, f"{group}.pdf")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def write_json(path: str, data: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_jsonl(args.samples_jsonl)
    samples = group_samples(rows)
    selected = select_cases(samples, args.min_abs_improvement, args.min_rel_improvement)

    selected_records = []
    for group, sample in selected.items():
        if sample is None:
            selected_records.append({"group": group, "selected": False})
            continue
        png_path, pdf_path = plot_case(group, sample, args.output_dir)
        record = selected_case_record(group, sample)
        record.update({"selected": True, "png": png_path, "pdf": pdf_path})
        selected_records.append(record)

    selected_path = os.path.join(args.output_dir, "selected_cases.json")
    write_json(selected_path, selected_records)
    print(f"[TrajectoryCaseViz] wrote selected cases: {selected_path}")
    print(f"[TrajectoryCaseViz] output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
