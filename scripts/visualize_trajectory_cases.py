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
LINE_STYLES = {
    "gt": "-",
    "openemma_text": "--",
    "fusion": "-",
    "stage7d": "-.",
    "stage7e": "-",
}
LINE_WIDTHS = {
    "gt": 2.2,
    "openemma_text": 1.9,
    "fusion": 1.9,
    "stage7d": 1.9,
    "stage7e": 2.0,
}
PANEL_ORDER = [
    ("fusion_advantage", "(a) Fusion advantage"),
    ("high_curvature", "(b) High-curvature"),
    ("speed_changing", "(c) Speed-changing"),
    ("steady_low_curvature", "(d) Steady low-curvature"),
]


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


def sample_score(sample: Dict[str, Any], key: str, default: float) -> float:
    value = finite_number(sample.get(key))
    return default if value is None else value


def sample_id(sample: Dict[str, Any]) -> str:
    if sample.get("sample_token"):
        return str(sample.get("sample_token"))
    return repr(sample.get("key"))


def rank_candidates(
    samples: List[Dict[str, Any]],
    predicate,
    score_fn,
    strict_predicate=None,
) -> List[Dict[str, Any]]:
    candidates = [sample for sample in samples if predicate(sample)]
    candidates.sort(key=lambda sample: score_fn(sample), reverse=True)
    ranked = []
    for rank, sample in enumerate(candidates, start=1):
        ranked.append(
            {
                "rank": rank,
                "sample": sample,
                "sample_token": sample.get("sample_token"),
                "scene_name": sample.get("scene_name"),
                "frame_idx": sample.get("frame_idx"),
                "score": score_fn(sample),
                "strict_match": bool(strict_predicate(sample)) if strict_predicate is not None else None,
            }
        )
    return ranked


def choose_ranked(ranked: List[Dict[str, Any]], used_sample_ids=None) -> Optional[Dict[str, Any]]:
    used_sample_ids = used_sample_ids or set()
    for item in ranked:
        if sample_id(item["sample"]) not in used_sample_ids:
            return item
    return ranked[0] if ranked else None


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

    strict_predicates = {
        "high_curvature": lambda sample: has_models(sample, ["openemma_text", "stage7e"])
        and (high_curv_threshold is None or sample_score(sample, "target_abs_curvature_x100", -1) >= high_curv_threshold)
        and clearly_better(sample, "openemma_text", "stage7e", min_abs, min_rel),
        "speed_changing": lambda sample: has_models(sample, ["openemma_text", "stage7d"])
        and (high_speed_threshold is None or sample_score(sample, "target_speed_change", -1) >= high_speed_threshold)
        and clearly_better(sample, "openemma_text", "stage7d", min_abs, min_rel),
        "steady_low_curvature": lambda sample: has_models(sample, ["openemma_text", "fusion", "stage7e"])
        and (low_curv_threshold is None or sample_score(sample, "target_abs_curvature_x100", 1e9) <= low_curv_threshold)
        and (low_speed_threshold is None or sample_score(sample, "target_speed_change", 1e9) <= low_speed_threshold)
        and clearly_better(sample, "fusion", "openemma_text", min_abs, min_rel)
        and clearly_better(sample, "stage7e", "openemma_text", min_abs, min_rel),
        "fusion_advantage": lambda sample: has_models(sample, ["fusion", "qwen_hidden", "ego_only"])
        and clearly_better(sample, "qwen_hidden", "fusion", min_abs, min_rel)
        and clearly_better(sample, "ego_only", "fusion", min_abs, min_rel),
    }

    fallback_specs = {
        "high_curvature": (
            lambda sample: has_models(sample, ["openemma_text", "stage7e"])
            and (high_curv_threshold is None or sample_score(sample, "target_abs_curvature_x100", -1) >= high_curv_threshold),
            lambda sample: improvement(sample, "openemma_text", "stage7e") or -1e9,
        ),
        "speed_changing": (
            lambda sample: has_models(sample, ["openemma_text", "stage7d"])
            and (high_speed_threshold is None or sample_score(sample, "target_speed_change", -1) >= high_speed_threshold),
            lambda sample: improvement(sample, "openemma_text", "stage7d") or -1e9,
        ),
        "steady_low_curvature": (
            lambda sample: has_models(sample, ["openemma_text", "fusion", "stage7e"])
            and (low_curv_threshold is None or sample_score(sample, "target_abs_curvature_x100", 1e9) <= low_curv_threshold)
            and (low_speed_threshold is None or sample_score(sample, "target_speed_change", 1e9) <= low_speed_threshold),
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
    ranked_by_group = {}
    for group, (predicate, score_fn) in fallback_specs.items():
        strict_ranked = rank_candidates(samples, strict_predicates[group], score_fn, strict_predicates[group])
        fallback_ranked = rank_candidates(samples, predicate, score_fn, strict_predicates[group])
        ranked_by_group[group] = strict_ranked or fallback_ranked

    selected = {}
    selected_items = {}
    used = set()
    for group in ["fusion_advantage", "high_curvature", "speed_changing", "steady_low_curvature"]:
        item = choose_ranked(ranked_by_group[group], used_sample_ids=used)
        selected[group] = item["sample"] if item is not None else None
        selected_items[group] = item
        if item is not None and group in {"high_curvature", "speed_changing"}:
            used.add(sample_id(item["sample"]))
    return selected, ranked_by_group, selected_items


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


def candidate_record(item: Dict[str, Any]) -> Dict[str, Any]:
    sample = item["sample"]
    return {
        "rank": item.get("rank"),
        "sample_token": item.get("sample_token"),
        "scene_name": item.get("scene_name"),
        "frame_idx": item.get("frame_idx"),
        "score": item.get("score"),
        "strict_match": item.get("strict_match"),
        "ade": {
            model: metric(sample, model)
            for model in ["openemma_text", "ego_only", "qwen_hidden", "fusion", "stage7d", "stage7e"]
            if model in sample["models"]
        },
        "improvement": {
            "stage7e_vs_text": improvement(sample, "openemma_text", "stage7e"),
            "stage7d_vs_text": improvement(sample, "openemma_text", "stage7d"),
            "fusion_vs_text": improvement(sample, "openemma_text", "fusion"),
            "text_vs_fusion": improvement(sample, "fusion", "openemma_text"),
            "fusion_vs_qwen_hidden": improvement(sample, "qwen_hidden", "fusion"),
            "fusion_vs_ego_only": improvement(sample, "ego_only", "fusion"),
        },
    }


def selected_case_record(
    group: str,
    sample: Dict[str, Any],
    selected_item: Optional[Dict[str, Any]],
    ranked_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
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
        "selection": candidate_record(selected_item) if selected_item is not None else None,
        "candidate_ranking": [candidate_record(item) for item in ranked_candidates[:20]],
    }


def model_label(model: str) -> str:
    if model == "gt":
        return "GT"
    for key, label in PLOT_MODELS:
        if key == model:
            return label
    return model


def collect_plot_points(sample: Dict[str, Any]) -> List[List[float]]:
    all_points = []
    gt = gt_points(sample)
    if gt:
        all_points.extend(gt)
    for model, _ in PLOT_MODELS:
        row = sample["models"].get(model)
        if row is None:
            continue
        points = row_points(row)
        if points:
            all_points.extend(points)
    return all_points


def set_panel_limits(ax, sample: Dict[str, Any]):
    points = collect_plot_points(sample)
    if not points:
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(1.0, (x_max - x_min) * 0.12)
    y_pad = max(0.5, (y_max - y_min) * 0.18)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal", adjustable="box")


def plot_points(ax, points: List[List[float]], model: str, label: Optional[str] = None):
    if not points:
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.plot(
        xs,
        ys,
        color=COLORS.get(model, "#666666"),
        linestyle=LINE_STYLES.get(model, "-"),
        linewidth=LINE_WIDTHS.get(model, 1.9),
        label=label,
    )
    ax.scatter([xs[0]], [ys[0]], color=COLORS.get(model, "#666666"), s=14, marker="o", zorder=4)
    ax.scatter([xs[-1]], [ys[-1]], color=COLORS.get(model, "#666666"), s=18, marker="s", zorder=4)


def format_ade(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def panel_annotation(group: str, sample: Dict[str, Any]) -> str:
    text_ade = metric(sample, "openemma_text")
    fusion_ade = metric(sample, "fusion")
    stage7d_ade = metric(sample, "stage7d")
    stage7e_ade = metric(sample, "stage7e")
    qwen_ade = metric(sample, "qwen_hidden")
    ego_ade = metric(sample, "ego_only")
    if group == "fusion_advantage":
        return f"Fusion {format_ade(fusion_ade)} vs Text {format_ade(text_ade)}"
    if group == "high_curvature":
        candidates = [("7D", stage7d_ade), ("7E", stage7e_ade)]
        candidates = [(name, ade) for name, ade in candidates if ade is not None]
        best_name, best_ade = min(candidates, key=lambda item: item[1]) if candidates else ("7E", stage7e_ade)
        return f"{best_name} {format_ade(best_ade)} vs Text {format_ade(text_ade)}"
    if group == "speed_changing":
        return f"7D {format_ade(stage7d_ade)} vs Text {format_ade(text_ade)}"
    if group == "steady_low_curvature":
        return f"Text {format_ade(text_ade)} vs Fusion {format_ade(fusion_ade)}"
    if qwen_ade is not None and ego_ade is not None:
        return f"Fusion {format_ade(fusion_ade)} vs Qwen {format_ade(qwen_ade)} / Ego {format_ade(ego_ade)}"
    return ""


def draw_case_panel(ax, group: str, title: str, sample: Dict[str, Any], add_labels: bool = False):
    ax.set_facecolor("white")
    gt = gt_points(sample)
    if gt:
        plot_points(ax, gt, "gt", label="GT" if add_labels else None)
    for model, label in PLOT_MODELS:
        row = sample["models"].get(model)
        if row is None:
            continue
        points = row_points(row)
        if points:
            plot_points(ax, points, model, label=label if add_labels else None)
    ax.axhline(0.0, color="#B0B0B0", linewidth=0.6, linestyle="--", alpha=0.55)
    ax.set_xlabel("Forward x (m)", fontsize=9)
    ax.set_ylabel("Lateral y (m)", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, linewidth=0.45, color="#D0D0D0", alpha=0.55)
    ax.tick_params(axis="both", labelsize=8)
    set_panel_limits(ax, sample)
    annotation = panel_annotation(group, sample)
    if annotation:
        ax.text(
            0.98,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
        )


def plot_case(group: str, sample: Dict[str, Any], output_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=200)
    fig.patch.set_facecolor("white")
    draw_case_panel(ax, group, group.replace("_", " ").title(), sample, add_labels=True)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()

    png_path = os.path.join(output_dir, f"{group}.png")
    pdf_path = os.path.join(output_dir, f"{group}.pdf")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def plot_grid(selected: Dict[str, Optional[Dict[str, Any]]], output_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), dpi=300)
    fig.patch.set_facecolor("white")
    handles = []
    labels = []
    for ax, (group, title) in zip(axes.ravel(), PANEL_ORDER):
        sample = selected.get(group)
        if sample is None:
            ax.axis("off")
            ax.set_title(title, fontsize=10, loc="left")
            continue
        draw_case_panel(ax, group, title, sample, add_labels=True)
        if not handles:
            handles, labels = ax.get_legend_handles_labels()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=5,
            frameon=False,
            fontsize=8.5,
            handlelength=2.6,
            columnspacing=1.2,
        )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    png_path = os.path.join(output_dir, "trajectory_cases_grid.png")
    pdf_path = os.path.join(output_dir, "trajectory_cases_grid.pdf")
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
    selected, ranked_by_group, selected_items = select_cases(
        samples,
        args.min_abs_improvement,
        args.min_rel_improvement,
    )

    selected_records = []
    for group, _ in PANEL_ORDER:
        sample = selected.get(group)
        if sample is None:
            selected_records.append(
                {
                    "group": group,
                    "selected": False,
                    "candidate_ranking": [candidate_record(item) for item in ranked_by_group.get(group, [])[:20]],
                }
            )
            continue
        png_path, pdf_path = plot_case(group, sample, args.output_dir)
        record = selected_case_record(
            group,
            sample,
            selected_items.get(group),
            ranked_by_group.get(group, []),
        )
        record.update({"selected": True, "png": png_path, "pdf": pdf_path})
        selected_records.append(record)

    grid_png, grid_pdf = plot_grid(selected, args.output_dir)
    selected_records.append({"group": "multi_panel_grid", "png": grid_png, "pdf": grid_pdf})
    selected_path = os.path.join(args.output_dir, "selected_cases.json")
    write_json(selected_path, selected_records)
    print(f"[TrajectoryCaseViz] wrote selected cases: {selected_path}")
    print(f"[TrajectoryCaseViz] output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
