#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODELS = ["openemma_text", "fusion", "stage7d", "stage7e", "graft_best"]
DEFAULT_METRICS = ["rollout_ade", "rollout_fde", "lateral_mae", "final_lateral_error"]
SCENARIO_ORDER = ["all", "high_curvature_top20", "speed_changing_top20", "steady_low_curvature"]
ECDF_METRICS = ("rollout_ade", "rollout_fde")
SCATTER_BOXPLOT_METRICS = ("rollout_ade", "rollout_fde")
HIST_METRIC = "rollout_ade"
MODEL_DISPLAY_NAMES = {
    "openemma_text": "OpenEMMA Text",
    "fusion": "Fusion",
    "stage7d": "Stage7D",
    "stage7e": "Stage7E",
    "graft_best": "GRAFT Best",
}
MODEL_COLORS = {
    "openemma_text": "#4C78A8",
    "fusion": "#F58518",
    "stage7d": "#54A24B",
    "stage7e": "#B279A2",
    "graft_best": "#E45756",
}
MODEL_ALIASES = {
    "openemma_text": {
        "openemma_text",
        "openemma",
        "text",
        "qwen_text",
        "qwen",
        "openemma_text_action",
        "openemma_text_only",
        "baseline_text",
        "text_baseline",
    },
    "fusion": {
        "fusion",
        "oft_fusion",
        "openemma_fusion",
        "fusion_action_head",
        "frozen_fusion",
        "stage_fusion",
    },
    "stage7d": {
        "stage7d",
        "stage_7d",
        "stage-7d",
        "s7d",
        "detached_residual_geometry_sequence_fusion",
        "detached_residual",
    },
    "stage7e": {
        "stage7e",
        "stage_7e",
        "stage-7e",
        "s7e",
        "curvature_only_detached_residual_geometry_sequence_fusion",
        "curvature_only_detached_residual",
    },
    "graft_best": {
        "graft_best",
        "graft",
        "graft_cache",
        "graft_cache_best",
        "frozen_fusion_curvature_residual",
        "frozen_fusion_curvature_residual_best",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot rollout error distributions from trajectory rollout samples.")
    parser.add_argument("--samples_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"[RolloutPlots] warning: {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[RolloutPlots] {message}")


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def alias_lookup() -> Dict[str, str]:
    lookup = {}
    for canonical, aliases in MODEL_ALIASES.items():
        lookup[normalize_name(canonical)] = canonical
        for alias in aliases:
            lookup[normalize_name(alias)] = canonical
    return lookup


def canonical_model_name(value: Any) -> Optional[str]:
    normalized = normalize_name(value)
    if not normalized:
        return None
    lookup = alias_lookup()
    if normalized in lookup:
        return lookup[normalized]
    for alias, canonical in lookup.items():
        if alias and (normalized == alias or normalized.endswith(f"_{alias}") or alias in normalized):
            return canonical
    return None


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def first_present(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            return record.get(key)
    return None


def sample_key(record: Dict[str, Any], fallback_index: int) -> str:
    for key in ("sample_token", "record_index", "line_no"):
        value = record.get(key)
        if value is not None and str(value) != "":
            return f"{key}:{value}"
    scene_name = record.get("scene_name")
    frame_idx = record.get("frame_idx")
    if scene_name is not None and frame_idx is not None:
        return f"scene_frame:{scene_name}:{frame_idx}"
    return f"row:{fallback_index}"


def load_samples(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                warn(f"skipping JSON decode error at line {line_no}")
                continue
            raw_model = first_present(record, ("model", "model_name", "name", "method", "variant"))
            canonical = canonical_model_name(raw_model)
            row = dict(record)
            for nested_key in ("metrics", "rollout_metrics"):
                nested = record.get(nested_key)
                if isinstance(nested, dict):
                    for key, value in nested.items():
                        row.setdefault(key, value)
            row["_line_no"] = line_no
            row["_raw_model"] = raw_model
            row["model"] = canonical
            row["sample_key"] = sample_key(record, line_no)
            rows.append(row)
    return pd.DataFrame(rows)


def requested_models(values: Iterable[str]) -> List[str]:
    models = []
    for value in values:
        canonical = canonical_model_name(value) or normalize_name(value)
        if canonical not in models:
            models.append(canonical)
    return models


def filter_models(df: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    unknown = sorted(str(value) for value in df["_raw_model"].dropna().unique() if canonical_model_name(value) is None)
    if unknown:
        warn(f"unrecognized model names skipped: {', '.join(unknown)}")
    available = set(df["model"].dropna().unique())
    for model in models:
        if model not in available:
            aliases = sorted(MODEL_ALIASES.get(model, {model}))
            warn(f"requested model '{model}' not found; aliases tried: {', '.join(aliases)}")
    return df[df["model"].isin(models)].copy()


def numeric_metric_frame(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    result = df.copy()
    for metric in metrics:
        if metric not in result.columns:
            warn(f"metric '{metric}' is missing from samples_jsonl")
            result[metric] = np.nan
        result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return result


def metric_values(df: pd.DataFrame, model: str, metric: str) -> np.ndarray:
    values = df.loc[df["model"] == model, metric].dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: str, stem: str, dpi: int, outputs: List[str]) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)


def display_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def plot_ecdf(df: pd.DataFrame, models: Sequence[str], metric: str, output_dir: str, dpi: int, outputs: List[str]) -> None:
    axis_label = "Rollout ADE (m)" if metric == "rollout_ade" else "Rollout FDE (m)"
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plotted = False
    for model in models:
        values = np.sort(metric_values(df, model, metric))
        if values.size == 0:
            warn(f"no finite values for {model}/{metric}; skipping ECDF curve")
            continue
        y = np.arange(1, values.size + 1, dtype=float) / values.size
        ax.step(values, y, where="post", linewidth=2.2, color=MODEL_COLORS.get(model), label=f"{display_name(model)} (N={values.size})")
        plotted = True
    if not plotted:
        warn(f"skipping {metric} ECDF because no finite values are available")
        plt.close(fig)
        return
    ax.set_title("Rollout ADE ECDF" if metric == "rollout_ade" else "Rollout FDE ECDF")
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Fraction of samples <= threshold")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, f"{metric}_ecdf", dpi, outputs)


def plot_scatter_boxplot(
    df: pd.DataFrame,
    models: Sequence[str],
    metric: str,
    output_dir: str,
    dpi: int,
    outputs: List[str],
) -> None:
    values_by_model = [metric_values(df, model, metric) for model in models]
    valid = [(model, values) for model, values in zip(models, values_by_model) if values.size > 0]
    if not valid:
        warn(f"skipping {metric} scatter boxplot because no finite values are available")
        return
    valid_models, valid_values = zip(*valid)
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    positions = np.arange(1, len(valid_models) + 1)
    box = ax.boxplot(
        valid_values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 1.2},
        whiskerprops={"color": "#333333", "linewidth": 1.1},
        capprops={"color": "#333333", "linewidth": 1.1},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#666666", "markeredgewidth": 0, "alpha": 0.35},
    )
    for patch, model in zip(box["boxes"], valid_models):
        patch.set_facecolor(MODEL_COLORS.get(model, "#CCCCCC"))
        patch.set_alpha(0.18)
    for position, model, values in zip(positions, valid_models, valid_values):
        jitter = rng.uniform(-0.18, 0.18, size=values.size)
        ax.scatter(
            np.full(values.size, position) + jitter,
            values,
            s=18,
            alpha=0.35,
            color=MODEL_COLORS.get(model, "#333333"),
            edgecolors="none",
            rasterized=True,
        )
    ax.set_title("Rollout ADE Distribution" if metric == "rollout_ade" else "Rollout FDE Distribution")
    ax.set_ylabel("Rollout ADE (m)" if metric == "rollout_ade" else "Rollout FDE (m)")
    ax.set_xticks(positions)
    ax.set_xticklabels([display_name(model) for model in valid_models], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    save_figure(fig, output_dir, f"{metric}_scatter_boxplot", dpi, outputs)


def plot_hist(df: pd.DataFrame, models: Sequence[str], metric: str, output_dir: str, dpi: int, outputs: List[str]) -> None:
    all_values = np.concatenate([metric_values(df, model, metric) for model in models if metric_values(df, model, metric).size > 0])
    if all_values.size == 0:
        warn(f"skipping {metric} histogram because no finite values are available")
        return
    bins = np.histogram_bin_edges(all_values, bins="auto")
    if bins.size < 3:
        bins = np.linspace(float(all_values.min()), float(all_values.max()) + 1e-6, 8)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for model in models:
        values = metric_values(df, model, metric)
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, histtype="step", linewidth=2.0, color=MODEL_COLORS.get(model), label=f"{display_name(model)} (N={values.size})")
    ax.set_title("Rollout ADE Histogram")
    ax.set_xlabel("Rollout ADE (m)")
    ax.set_ylabel("Number of samples")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, f"{metric}_hist", dpi, outputs)


def paired_metric(df: pd.DataFrame, metric: str, text_model: str, model: str) -> pd.DataFrame:
    columns = ["sample_key", metric]
    text = df.loc[df["model"] == text_model, columns].dropna().rename(columns={metric: "text_metric"})
    other = df.loc[df["model"] == model, columns].dropna().rename(columns={metric: "model_metric"})
    return text.merge(other, on="sample_key", how="inner")


def plot_improvement_hist(df: pd.DataFrame, models: Sequence[str], output_dir: str, dpi: int, outputs: List[str]) -> None:
    if "openemma_text" not in models:
        warn("skipping improvement_vs_text_hist because openemma_text is not selected")
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    plotted = False
    improvements = []
    for model in models:
        if model == "openemma_text":
            continue
        paired = paired_metric(df, "rollout_ade", "openemma_text", model)
        if paired.empty:
            warn(f"no common samples for openemma_text vs {model}; skipping improvement histogram")
            continue
        improvement = (paired["text_metric"] - paired["model_metric"]).to_numpy(dtype=float)
        improvements.append(improvement)
    if not improvements:
        plt.close(fig)
        warn("skipping improvement_vs_text_hist because no paired improvements are available")
        return
    bins = np.histogram_bin_edges(np.concatenate(improvements), bins="auto")
    for model in models:
        if model == "openemma_text":
            continue
        paired = paired_metric(df, "rollout_ade", "openemma_text", model)
        if paired.empty:
            continue
        improvement = (paired["text_metric"] - paired["model_metric"]).to_numpy(dtype=float)
        ax.hist(
            improvement,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color=MODEL_COLORS.get(model),
            label=f"{display_name(model)} (N={improvement.size})",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--")
    ax.set_title("ADE Improvement vs OpenEMMA Text")
    ax.set_xlabel("ADE improvement over text (m)")
    ax.set_ylabel("Number of samples")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "improvement_vs_text_hist", dpi, outputs)


def explicit_scenario_column(df: pd.DataFrame) -> Optional[str]:
    for column in ("scenario", "group", "scenario_group"):
        if column in df.columns:
            normalized = df[column].map(normalize_name)
            if normalized.isin(SCENARIO_ORDER).any():
                df["scenario"] = normalized
                return column
    return None


def infer_scenarios(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    if df.empty:
        return df, "empty dataframe"
    column = explicit_scenario_column(df)
    if column:
        explicit = df[df["scenario"].isin(SCENARIO_ORDER)].copy()
        all_rows = df.copy()
        all_rows["scenario"] = "all"
        return pd.concat([all_rows, explicit[explicit["scenario"] != "all"]], ignore_index=True), None

    required = ["target_abs_curvature_x100", "target_speed_change"]
    if not all(column in df.columns for column in required):
        return df, "missing explicit scenario/group and missing target_abs_curvature_x100 or target_speed_change"

    sample_scores = (
        df[["sample_key", "target_abs_curvature_x100", "target_speed_change"]]
        .copy()
        .assign(
            target_abs_curvature_x100=lambda x: pd.to_numeric(x["target_abs_curvature_x100"], errors="coerce"),
            target_speed_change=lambda x: pd.to_numeric(x["target_speed_change"], errors="coerce"),
        )
        .dropna()
        .drop_duplicates(subset=["sample_key"])
        .reset_index(drop=True)
    )
    if sample_scores.empty:
        return df, "target_abs_curvature_x100 and target_speed_change are present but not finite"

    top_count = max(1, int(math.ceil(len(sample_scores) * 0.2)))
    high_curvature = set(sample_scores.nlargest(top_count, "target_abs_curvature_x100")["sample_key"])
    speed_changing = set(sample_scores.nlargest(top_count, "target_speed_change")["sample_key"])
    low_curvature = set(sample_scores.nsmallest(top_count, "target_abs_curvature_x100")["sample_key"])
    low_speed_change = set(sample_scores.nsmallest(top_count, "target_speed_change")["sample_key"])
    steady = low_curvature & low_speed_change

    scenario_rows = []
    for key in sample_scores["sample_key"]:
        scenario_rows.append({"sample_key": key, "scenario": "all"})
        if key in high_curvature:
            scenario_rows.append({"sample_key": key, "scenario": "high_curvature_top20"})
        if key in speed_changing:
            scenario_rows.append({"sample_key": key, "scenario": "speed_changing_top20"})
        if key in steady:
            scenario_rows.append({"sample_key": key, "scenario": "steady_low_curvature"})
    scenario_df = pd.DataFrame(scenario_rows)
    return df.merge(scenario_df, on="sample_key", how="inner"), None


def plot_scenario_bars(
    df: pd.DataFrame,
    models: Sequence[str],
    output_dir: str,
    dpi: int,
    outputs: List[str],
) -> Optional[str]:
    scenario_df, skip_reason = infer_scenarios(df)
    if skip_reason:
        warn(f"skipping scenario_rollout_bar: {skip_reason}")
        return skip_reason
    if scenario_df.empty:
        skip_reason = "no samples matched scenario groups"
        warn(f"skipping scenario_rollout_bar: {skip_reason}")
        return skip_reason

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    for ax, metric, ylabel in zip(axes, ("rollout_ade", "rollout_fde"), ("Rollout ADE (m)", "Rollout FDE (m)")):
        means = (
            scenario_df[scenario_df["model"].isin(models)]
            .groupby(["scenario", "model"], as_index=False)[metric]
            .mean()
        )
        x = np.arange(len(SCENARIO_ORDER))
        width = 0.78 / max(1, len(models))
        for idx, model in enumerate(models):
            heights = []
            for scenario in SCENARIO_ORDER:
                value = means.loc[(means["scenario"] == scenario) & (means["model"] == model), metric]
                heights.append(float(value.iloc[0]) if not value.empty and pd.notna(value.iloc[0]) else np.nan)
            offset = (idx - (len(models) - 1) / 2.0) * width
            ax.bar(x + offset, heights, width=width, color=MODEL_COLORS.get(model), label=display_name(model), alpha=0.9)
        ax.set_title("ADE by Scenario" if metric == "rollout_ade" else "FDE by Scenario")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Scenario Rollout Error", y=1.02, fontsize=15)
    save_figure(fig, output_dir, "scenario_rollout_bar", dpi, outputs)
    return None


def write_summary_stats(df: pd.DataFrame, models: Sequence[str], metrics: Sequence[str], output_dir: str, outputs: List[str]) -> None:
    rows = []
    for model in models:
        for metric in metrics:
            values = metric_values(df, model, metric)
            if values.size == 0:
                rows.append({"model": model, "metric": metric, "count": 0})
                continue
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "p75": float(np.percentile(values, 75)),
                    "p90": float(np.percentile(values, 90)),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(np.max(values)),
                }
            )
    path = os.path.join(output_dir, "summary_stats.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    outputs.append(path)


def write_win_rates(df: pd.DataFrame, models: Sequence[str], metrics: Sequence[str], output_dir: str, outputs: List[str]) -> None:
    rows = []
    if "openemma_text" not in models:
        warn("win_rate_vs_text.csv will be empty because openemma_text is not selected")
    for model in models:
        if model == "openemma_text":
            continue
        for metric in metrics:
            paired = paired_metric(df, metric, "openemma_text", model)
            if paired.empty:
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "num_common_samples": 0,
                        "win_rate_vs_text": np.nan,
                        "mean_improvement_vs_text": np.nan,
                        "median_improvement_vs_text": np.nan,
                    }
                )
                continue
            improvement = paired["text_metric"] - paired["model_metric"]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "num_common_samples": int(len(paired)),
                    "win_rate_vs_text": float((paired["model_metric"] < paired["text_metric"]).mean()),
                    "mean_improvement_vs_text": float(improvement.mean()),
                    "median_improvement_vs_text": float(improvement.median()),
                }
            )
    path = os.path.join(output_dir, "win_rate_vs_text.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    outputs.append(path)


def write_readme(output_dir: str, outputs: List[str], scenario_skip_reason: Optional[str]) -> None:
    lines = [
        "Rollout error distribution figures",
        "",
        "Inputs are per-sample trajectory rollout metrics. All labels are in English to avoid font portability issues.",
        "",
        "Figures:",
        "- rollout_ade_ecdf / rollout_fde_ecdf: ECDF curves. Curves closer to the upper-left are better.",
        "- rollout_ade_scatter_boxplot / rollout_fde_scatter_boxplot: each scatter point is one sample; the boxplot shows median, IQR, whiskers, and long-tail outliers.",
        "- rollout_ade_hist: rollout ADE distribution counts for each model.",
        "- improvement_vs_text_hist: per-sample ADE improvement over openemma_text. Values greater than 0 mean the model is better than openemma_text on that sample.",
        "- scenario_rollout_bar: grouped bar chart for rollout ADE and FDE across all/high-curvature/speed-changing/steady-low-curvature scenarios.",
        "",
    ]
    if scenario_skip_reason:
        lines.append(f"scenario_rollout_bar was skipped: {scenario_skip_reason}.")
    else:
        lines.append("scenario_rollout_bar was generated from explicit scenario/group labels or reconstructed target curvature/speed-change scores.")
    path = os.path.join(output_dir, "README.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    outputs.append(path)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_paper_style()

    models = requested_models(args.models)
    metrics = list(OrderedDict.fromkeys(args.metrics))
    plot_metrics = list(OrderedDict.fromkeys(metrics + ["rollout_ade", "rollout_fde"]))
    df = load_samples(args.samples_jsonl)
    if df.empty:
        raise ValueError(f"no usable records found in {args.samples_jsonl}")
    df = filter_models(df, models)
    if df.empty:
        raise ValueError("no records left after model alias filtering")
    df = numeric_metric_frame(df, plot_metrics)

    outputs: List[str] = []
    write_summary_stats(df, models, metrics, args.output_dir, outputs)
    write_win_rates(df, models, metrics, args.output_dir, outputs)

    for metric in ECDF_METRICS:
        plot_ecdf(df, models, metric, args.output_dir, args.dpi, outputs)
    for metric in SCATTER_BOXPLOT_METRICS:
        plot_scatter_boxplot(df, models, metric, args.output_dir, args.dpi, outputs)
    plot_hist(df, models, HIST_METRIC, args.output_dir, args.dpi, outputs)
    plot_improvement_hist(df, models, args.output_dir, args.dpi, outputs)
    scenario_skip_reason = plot_scenario_bars(df, models, args.output_dir, args.dpi, outputs)
    write_readme(args.output_dir, outputs, scenario_skip_reason)

    info(f"wrote {len(outputs)} files to {args.output_dir}")
    for path in outputs:
        info(os.path.basename(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
