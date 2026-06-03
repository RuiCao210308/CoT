#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("generate_paper_figures_850.py requires numpy.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


FUSION_COLOR = "#4D4D4D"
GATED_COLOR = "#8B1A1A"
GREEN = "#1B7837"
PURPLE = "#6A3D9A"
ORANGE = "#E66101"
BLUE_GRAY = "#5E81AC"

DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/paper_figures_850_final"

FUSION_EVAL_JSON = "/root/autodl-tmp/fusion_trainval_850scenes_e5_real/eval.json"
FUSION_RECORDS_JSONL = "/root/autodl-tmp/fusion_trainval_850scenes_e5_real/eval_records.jsonl"
GATED_DIR = "/root/autodl-tmp/gated_graft_trainval_850scenes_e5_rs0.1_gw0.5_gate0_realbase"
GATED_EVAL_JSON = f"{GATED_DIR}/eval.json"
GATED_RECORDS_JSONL = f"{GATED_DIR}/eval_records.jsonl"
GATED_SCENARIO_JSON = f"{GATED_DIR}/scenario_analysis.json"
GATED_ROLLOUT_JSON = f"{GATED_DIR}/trajectory_rollout_analysis.json"
GATED_ROLLOUT_SAMPLES_JSONL = f"{GATED_DIR}/trajectory_rollout_samples.jsonl"
GATED_SEED_EVALS = [
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed0_rs0.1_gw0.5_gate0_realbase/eval.json",
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed1_rs0.1_gw0.5_gate0_realbase/eval.json",
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed2_rs0.1_gw0.5_gate0_realbase/eval.json",
]
GATED_SEED_CKPTS = [
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed0_rs0.1_gw0.5_gate0_realbase/gated_geometry_residual_fusion_head.pt",
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed1_rs0.1_gw0.5_gate0_realbase/gated_geometry_residual_fusion_head.pt",
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed2_rs0.1_gw0.5_gate0_realbase/gated_geometry_residual_fusion_head.pt",
]

ACTION_METRICS = {
    "Fusion": {"Speed MAE": 1.0694, "Curv. MAE x100": 0.8426, "Overall L1": 0.9560},
    "Gated-GRAFT": {"Speed MAE": 1.2113, "Curv. MAE x100": 0.8370, "Overall L1": 1.0241},
}
ROLLOUT_METRICS = {
    "Fusion": {"ADE": 2.4261, "FDE": 5.8065, "Long.": 2.1499, "Lat.": 0.6797, "Final Lat.": 1.7789},
    "Gated-GRAFT": {"ADE": 2.7365, "FDE": 6.4259, "Long.": 2.5270, "Lat.": 0.6101, "Final Lat.": 1.6555},
}
SCENARIO_METRICS = {
    "High-curvature": {
        "Fusion": {"ADE": 3.2442, "Long.": 2.5209, "Lat.": 1.4891, "Final Lat.": 4.1945},
        "Gated-GRAFT": {"ADE": 3.5815, "Long.": 3.0005, "Lat.": 1.3696, "Final Lat.": 3.9565},
    },
    "Speed-changing": {
        "Fusion": {"ADE": 3.7943, "Long.": 3.4787, "Lat.": 0.8539, "Final Lat.": 2.3163},
        "Gated-GRAFT": {"ADE": 3.9223, "Long.": 3.6462, "Lat.": 0.7824, "Final Lat.": 2.1783},
    },
    "Steady low-curv.": {
        "Fusion": {"ADE": 0.8141, "Long.": 0.8039, "Lat.": 0.0749, "Final Lat.": 0.1601},
        "Gated-GRAFT": {"ADE": 0.6785, "Long.": 0.6662, "Lat.": 0.0736, "Final Lat.": 0.1648},
    },
}
GATED_SEED_ACTION = [
    {"Speed MAE": 1.2113, "Curv. MAE x100": 0.8370, "Overall L1": 1.0241},
    {"Speed MAE": 1.1040, "Curv. MAE x100": 0.8291, "Overall L1": 0.9665},
    {"Speed MAE": 1.1367, "Curv. MAE x100": 0.8297, "Overall L1": 0.9832},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-style 850scenes result figures.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--shade_alpha", type=float, default=0.16)
    return parser.parse_args()


def info(message: str) -> None:
    print(f"[PaperFigures850] {message}")


def warn(message: str) -> None:
    print(f"[PaperFigures850] warning: {message}")


def set_paper_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int, outputs: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def blend_with_white(value: str, alpha: float) -> Tuple[int, int, int]:
    rgb = hex_to_rgb(value)
    return tuple(int((1.0 - alpha) * 255 + alpha * channel) for channel in rgb)


def pil_font(size: int, bold: bool = False):
    if ImageFont is None:
        return None
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_size(draw, text: str, font) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(draw, xy: Tuple[float, float], text: str, font, fill, anchor: str = "mm") -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_vertical_label(image, text: str, xy: Tuple[int, int], font, fill) -> None:
    label = Image.new("RGBA", (360, 42), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((180, 21), text, font=font, fill=fill, anchor="mm")
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, (xy[0] - rotated.width // 2, xy[1] - rotated.height // 2), rotated)


def save_pil_figure(image, output_dir: Path, stem: str, outputs: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"{stem}.{ext}"
        if ext == "png":
            image.save(path)
        else:
            image.convert("RGB").save(path, "PDF", resolution=300.0)
        outputs.append(str(path))


def draw_pil_grouped_bar(
    stem: str,
    output_dir: Path,
    labels: Sequence[str],
    fusion_values: Sequence[float],
    gated_values: Sequence[float],
    ylabel: str,
    note: Optional[str],
    outputs: List[str],
    size_inches: Tuple[float, float] = (3.5, 2.45),
) -> None:
    if Image is None or ImageDraw is None:
        raise SystemExit("generate_paper_figures_850.py requires matplotlib or Pillow.")
    scale = 300
    width_px = int(size_inches[0] * scale)
    height_px = int(size_inches[1] * scale)
    image = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(image)
    font = pil_font(24)
    small = pil_font(21)
    tiny = pil_font(19)

    left, right, top, bottom = 115, 34, 36, 98
    plot_w = width_px - left - right
    plot_h = height_px - top - bottom
    max_value = max(max(fusion_values), max(gated_values)) * 1.18
    max_value = max(max_value, 1e-6)

    grid = blend_with_white("#000000", 0.18)
    axis = hex_to_rgb("#333333")
    for frac in np.linspace(0.0, 1.0, 5):
        y = top + plot_h * (1 - frac)
        draw.line((left, y, left + plot_w, y), fill=grid, width=1)
        label = f"{max_value * frac:.1f}"
        tw, th = text_size(draw, label, tiny)
        draw.text((left - tw - 10, y - th / 2), label, font=tiny, fill=axis)
    draw.line((left, top, left, top + plot_h), fill=axis, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis, width=2)

    group_w = plot_w / len(labels)
    bar_w = group_w * 0.26
    fusion_color = hex_to_rgb(FUSION_COLOR)
    gated_color = hex_to_rgb(GATED_COLOR)
    for idx, label in enumerate(labels):
        cx = left + group_w * (idx + 0.5)
        vals = [(fusion_values[idx], -bar_w * 0.62, fusion_color), (gated_values[idx], bar_w * 0.62, gated_color)]
        for value, offset, color in vals:
            x0 = cx + offset - bar_w / 2
            x1 = cx + offset + bar_w / 2
            y0 = top + plot_h * (1 - value / max_value)
            y1 = top + plot_h
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=hex_to_rgb("#222222"), width=1)
        draw_centered(draw, (cx, top + plot_h + 32), label, tiny, axis)

    draw_vertical_label(image, ylabel, (27, int(top + plot_h / 2)), small, axis)
    legend_y = 18
    legend_x = left
    draw.rectangle((legend_x, legend_y, legend_x + 30, legend_y + 14), fill=fusion_color)
    draw.text((legend_x + 38, legend_y - 7), "Fusion", font=small, fill=axis)
    legend_x += 170
    draw.rectangle((legend_x, legend_y, legend_x + 30, legend_y + 14), fill=gated_color)
    draw.text((legend_x + 38, legend_y - 7), "Gated-GRAFT", font=small, fill=axis)
    if note:
        lines = note.split("\n")
        y = top + 8
        for line in lines:
            tw, _ = text_size(draw, line, tiny)
            draw.text((left + plot_w - tw, y), line, font=tiny, fill=hex_to_rgb("#333333"))
            y += 24
    save_pil_figure(image, output_dir, stem, outputs)


def draw_pil_scenario_rollout(output_dir: Path, outputs: List[str]) -> None:
    if Image is None or ImageDraw is None:
        raise SystemExit("generate_paper_figures_850.py requires matplotlib or Pillow.")
    scale = 300
    width_px, height_px = int(7.05 * scale), int(2.35 * scale)
    image = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(image)
    small = pil_font(21)
    tiny = pil_font(18)
    title_font = pil_font(22, bold=True)
    axis = hex_to_rgb("#333333")
    grid = blend_with_white("#000000", 0.18)
    fusion_color = hex_to_rgb(FUSION_COLOR)
    gated_color = hex_to_rgb(GATED_COLOR)
    metric_labels = ["ADE", "Long.", "Lat.", "Final Lat."]

    draw.rectangle((780, 15, 810, 29), fill=fusion_color)
    draw.text((818, 8), "Fusion", font=small, fill=axis)
    draw.rectangle((960, 15, 990, 29), fill=gated_color)
    draw.text((998, 8), "Gated-GRAFT", font=small, fill=axis)

    panel_gap = 38
    left_margin = 78
    panel_w = (width_px - left_margin - 28 - 2 * panel_gap) / 3
    top, bottom = 72, 78
    plot_h = height_px - top - bottom
    for panel_idx, (scenario, model_values) in enumerate(SCENARIO_METRICS.items()):
        panel_left = left_margin + panel_idx * (panel_w + panel_gap)
        panel_right = panel_left + panel_w
        values = [model_values[m][metric] for m in ("Fusion", "Gated-GRAFT") for metric in metric_labels]
        max_value = max(values) * 1.16
        draw.text((panel_left + panel_w / 2, 43), scenario, font=title_font, fill=axis, anchor="mm")
        for frac in np.linspace(0.0, 1.0, 4):
            y = top + plot_h * (1 - frac)
            draw.line((panel_left, y, panel_right, y), fill=grid, width=1)
            if panel_idx == 0:
                label = f"{max_value * frac:.1f}"
                tw, th = text_size(draw, label, tiny)
                draw.text((panel_left - tw - 8, y - th / 2), label, font=tiny, fill=axis)
        draw.line((panel_left, top, panel_left, top + plot_h), fill=axis, width=2)
        draw.line((panel_left, top + plot_h, panel_right, top + plot_h), fill=axis, width=2)
        group_w = panel_w / len(metric_labels)
        bar_w = group_w * 0.25
        for idx, metric in enumerate(metric_labels):
            cx = panel_left + group_w * (idx + 0.5)
            for value, offset, color in (
                (model_values["Fusion"][metric], -bar_w * 0.62, fusion_color),
                (model_values["Gated-GRAFT"][metric], bar_w * 0.62, gated_color),
            ):
                x0 = cx + offset - bar_w / 2
                x1 = cx + offset + bar_w / 2
                y0 = top + plot_h * (1 - value / max_value)
                draw.rectangle((x0, y0, x1, top + plot_h), fill=color, outline=hex_to_rgb("#222222"), width=1)
            draw.text((cx, top + plot_h + 24), metric, font=tiny, fill=axis, anchor="mm")
    save_pil_figure(image, output_dir, "fig_850_scenario_rollout", outputs)


def draw_pil_multiseed_action(output_dir: Path, outputs: List[str]) -> None:
    if Image is None or ImageDraw is None:
        raise SystemExit("generate_paper_figures_850.py requires matplotlib or Pillow.")
    labels = ["Speed MAE", "Curv. MAE x100", "Overall L1"]
    fusion = np.asarray([ACTION_METRICS["Fusion"][label] for label in labels], dtype=float)
    gated_seed_values = np.asarray([[seed[label] for label in labels] for seed in GATED_SEED_ACTION], dtype=float)
    gated_mean = gated_seed_values.mean(axis=0)
    gated_std = gated_seed_values.std(axis=0, ddof=1)
    scale = 300
    width_px, height_px = int(3.5 * scale), int(2.45 * scale)
    image = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(image)
    small = pil_font(21)
    tiny = pil_font(18)
    axis = hex_to_rgb("#333333")
    grid = blend_with_white("#000000", 0.18)
    fusion_color = hex_to_rgb(FUSION_COLOR)
    gated_color = hex_to_rgb(GATED_COLOR)
    left, right, top, bottom = 115, 34, 36, 98
    plot_w = width_px - left - right
    plot_h = height_px - top - bottom
    max_value = max(float(fusion.max()), float((gated_mean + gated_std).max())) * 1.18
    for frac in np.linspace(0.0, 1.0, 5):
        y = top + plot_h * (1 - frac)
        draw.line((left, y, left + plot_w, y), fill=grid, width=1)
        label = f"{max_value * frac:.1f}"
        tw, th = text_size(draw, label, tiny)
        draw.text((left - tw - 10, y - th / 2), label, font=tiny, fill=axis)
    draw.line((left, top, left, top + plot_h), fill=axis, width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis, width=2)
    group_w = plot_w / len(labels)
    bar_w = group_w * 0.26
    for idx, label in enumerate(labels):
        cx = left + group_w * (idx + 0.5)
        for value, offset, color in (
            (fusion[idx], -bar_w * 0.62, fusion_color),
            (gated_mean[idx], bar_w * 0.62, gated_color),
        ):
            x0 = cx + offset - bar_w / 2
            x1 = cx + offset + bar_w / 2
            y0 = top + plot_h * (1 - value / max_value)
            draw.rectangle((x0, y0, x1, top + plot_h), fill=color, outline=hex_to_rgb("#222222"), width=1)
        err_x = cx + bar_w * 0.62
        y_mean = top + plot_h * (1 - gated_mean[idx] / max_value)
        y_low = top + plot_h * (1 - (gated_mean[idx] - gated_std[idx]) / max_value)
        y_high = top + plot_h * (1 - (gated_mean[idx] + gated_std[idx]) / max_value)
        draw.line((err_x, y_high, err_x, y_low), fill=axis, width=3)
        draw.line((err_x - 9, y_high, err_x + 9, y_high), fill=axis, width=3)
        draw.line((err_x - 9, y_low, err_x + 9, y_low), fill=axis, width=3)
        y_fusion = top + plot_h * (1 - fusion[idx] / max_value)
        draw.line((cx - group_w * 0.42, y_fusion, cx + group_w * 0.42, y_fusion), fill=fusion_color, width=3)
        draw_centered(draw, (cx, top + plot_h + 32), label, tiny, axis)
    draw.rectangle((left, 17, left + 30, 31), fill=fusion_color)
    draw.text((left + 38, 10), "Fusion (single run)", font=small, fill=axis)
    draw.rectangle((left + 315, 17, left + 345, 31), fill=gated_color)
    draw.text((left + 353, 10), "Gated-GRAFT (mean +/- std)", font=small, fill=axis)
    draw_vertical_label(image, "Error (lower is better)", (27, int(top + plot_h / 2)), small, axis)
    save_pil_figure(image, output_dir, "fig_850_gated_multiseed_action", outputs)


def grouped_bar(
    ax: plt.Axes,
    labels: Sequence[str],
    fusion_values: Sequence[float],
    gated_values: Sequence[float],
    ylabel: str,
    legend: bool = True,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    ax.bar(
        x - width / 2,
        fusion_values,
        width,
        label="Fusion",
        color=FUSION_COLOR,
        edgecolor="#222222",
        linewidth=0.45,
    )
    ax.bar(
        x + width / 2,
        gated_values,
        width,
        label="Gated-GRAFT",
        color=GATED_COLOR,
        edgecolor="#5A0D0D",
        linewidth=0.45,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.margins(x=0.04)
    if legend:
        ax.legend(frameon=False, loc="best")


def plot_action_metrics(output_dir: Path, dpi: int, outputs: List[str]) -> None:
    labels = ["Speed MAE", "Curv. MAE x100", "Overall L1"]
    fusion = [ACTION_METRICS["Fusion"][label] for label in labels]
    gated = [ACTION_METRICS["Gated-GRAFT"][label] for label in labels]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    grouped_bar(ax, labels, fusion, gated, "Error (lower is better)")
    ax.text(
        0.98,
        0.95,
        "Curvature slightly better;\nspeed/overall worse",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        color="#333333",
    )
    save_figure(fig, output_dir, "fig_850_action_metrics", dpi, outputs)


def plot_rollout_tradeoff(output_dir: Path, dpi: int, outputs: List[str]) -> None:
    labels = ["ADE", "FDE", "Long.", "Lat.", "Final Lat."]
    fusion = [ROLLOUT_METRICS["Fusion"][label] for label in labels]
    gated = [ROLLOUT_METRICS["Gated-GRAFT"][label] for label in labels]
    fig, ax = plt.subplots(figsize=(3.5, 2.45))
    grouped_bar(ax, labels, fusion, gated, "Rollout error (m)")
    ax.text(
        0.98,
        0.95,
        "Trade-off: Fusion better on ADE/FDE/long.;\nGated-GRAFT better laterally",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#333333",
    )
    save_figure(fig, output_dir, "fig_850_rollout_tradeoff", dpi, outputs)


def plot_scenario_rollout(output_dir: Path, dpi: int, outputs: List[str]) -> None:
    metric_labels = ["ADE", "Long.", "Lat.", "Final Lat."]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35), sharey=False)
    for ax, (scenario, model_values) in zip(axes, SCENARIO_METRICS.items()):
        fusion = [model_values["Fusion"][metric] for metric in metric_labels]
        gated = [model_values["Gated-GRAFT"][metric] for metric in metric_labels]
        grouped_bar(ax, metric_labels, fusion, gated, "Error (m)", legend=False)
        ax.set_title(scenario, pad=3)
        ax.tick_params(axis="x", rotation=25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, output_dir, "fig_850_scenario_rollout", dpi, outputs)


def plot_multiseed_action(output_dir: Path, dpi: int, outputs: List[str]) -> None:
    labels = ["Speed MAE", "Curv. MAE x100", "Overall L1"]
    x = np.arange(len(labels), dtype=float)
    width = 0.32
    fusion = np.asarray([ACTION_METRICS["Fusion"][label] for label in labels], dtype=float)
    gated_seed_values = np.asarray([[seed[label] for label in labels] for seed in GATED_SEED_ACTION], dtype=float)
    gated_mean = gated_seed_values.mean(axis=0)
    gated_std = gated_seed_values.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.bar(
        x - width / 2,
        fusion,
        width,
        label="Fusion (single run)",
        color=FUSION_COLOR,
        edgecolor="#222222",
        linewidth=0.45,
    )
    ax.bar(
        x + width / 2,
        gated_mean,
        width,
        yerr=gated_std,
        capsize=3.0,
        label="Gated-GRAFT (mean ± std)",
        color=GATED_COLOR,
        edgecolor="#5A0D0D",
        ecolor="#333333",
        error_kw={"elinewidth": 0.85, "capthick": 0.85},
        linewidth=0.45,
    )
    for idx, value in enumerate(fusion):
        ax.hlines(value, idx - 0.42, idx + 0.42, colors=FUSION_COLOR, linestyles="dashed", linewidth=1.0, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (lower is better)")
    ax.grid(axis="y", alpha=0.20, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, output_dir, "fig_850_gated_multiseed_action", dpi, outputs)


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def load_checkpoint_history(path: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        warn(f"missing checkpoint for optional training curve: {path}")
        return None
    try:
        import torch
    except ImportError:
        warn("torch is not available; skipping optional training curve")
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        warn(f"failed to load checkpoint {path}: {exc}")
        return None
    if not isinstance(checkpoint, dict):
        warn(f"checkpoint is not a dict: {path}")
        return None
    history = checkpoint.get("train_history")
    if not isinstance(history, list) or not history:
        warn(f"checkpoint has no train_history: {path}")
        return None
    return [row for row in history if isinstance(row, dict)]


def select_history_metric(histories: Sequence[List[Dict[str, Any]]]) -> Optional[str]:
    candidates = ["avg_total_loss", "avg_action_loss", "avg_loss"]
    for metric in candidates:
        if all(any(finite_number(row.get(metric)) is not None for row in history) for history in histories):
            return metric
    return None


def history_xy(history: List[Dict[str, Any]], metric: str) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for idx, row in enumerate(history):
        y = finite_number(row.get(metric))
        if y is None:
            continue
        x = finite_number(row.get("epoch"))
        if x is None:
            x = finite_number(row.get("global_step"))
        if x is None:
            x = finite_number(row.get("step"))
        if x is None:
            x = float(idx + 1)
        xs.append(x)
        ys.append(y)
    order = np.argsort(np.asarray(xs, dtype=float))
    return np.asarray(xs, dtype=float)[order], np.asarray(ys, dtype=float)[order]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def align_common(curves: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    common = set(curves[0][0].tolist())
    for xs, _ in curves[1:]:
        common &= set(xs.tolist())
    if not common:
        return None
    x_grid = np.asarray(sorted(common), dtype=float)
    rows = []
    for xs, ys in curves:
        mapping = {float(x): float(y) for x, y in zip(xs, ys)}
        rows.append([mapping[float(x)] for x in x_grid])
    return x_grid, np.asarray(rows, dtype=float)


def plot_training_curve(output_dir: Path, dpi: int, outputs: List[str], smooth_window: int, shade_alpha: float) -> Tuple[bool, str]:
    histories = [history for history in (load_checkpoint_history(path) for path in GATED_SEED_CKPTS) if history]
    if len(histories) != 3:
        return False, f"skipped: expected 3 train_history entries, found {len(histories)}"
    metric = select_history_metric(histories)
    if metric is None:
        return False, "skipped: no shared training metric key among avg_total_loss/avg_action_loss/avg_loss"
    curves = [history_xy(history, metric) for history in histories]
    curves = [(xs, ys) for xs, ys in curves if xs.size and ys.size]
    aligned = align_common(curves)
    if aligned is None:
        return False, "skipped: training histories do not share common epoch/step values"
    x_grid, values = aligned
    values = np.asarray([smooth(row, smooth_window) for row in values], dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    ax.plot(x_grid, mean, color=GATED_COLOR, linewidth=2.25, label="Gated-GRAFT mean")
    ax.fill_between(x_grid, mean - std, mean + std, color=GATED_COLOR, alpha=shade_alpha, linewidth=0)
    ax.set_xlabel("Epoch / step")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, output_dir, "fig_850_gated_training_curve_mean_std", dpi, outputs)
    return True, f"generated from metric={metric}, smooth_window={smooth_window}, n=3"


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_num_samples(path: str) -> Optional[int]:
    data = load_json_if_exists(path)
    if not data:
        return None
    for key in ("num_samples", "evaluated_records", "parse_success"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    return None


def write_readme(output_dir: Path, outputs: Sequence[str], training_curve_status: str) -> None:
    fusion_n = read_num_samples(FUSION_EVAL_JSON)
    gated_n = read_num_samples(GATED_EVAL_JSON)
    lines = [
        "Paper figures for 850scenes results",
        "",
        "All plotted metrics are from the 850scenes evaluation setting. Lower is better for every metric shown.",
        "",
        "Data sources:",
        f"- Fusion eval: {FUSION_EVAL_JSON}",
        f"- Fusion records: {FUSION_RECORDS_JSONL}",
        f"- Gated-GRAFT eval: {GATED_EVAL_JSON}",
        f"- Gated-GRAFT records: {GATED_RECORDS_JSONL}",
        f"- Scenario analysis: {GATED_SCENARIO_JSON}",
        f"- Trajectory rollout analysis: {GATED_ROLLOUT_JSON}",
        f"- Trajectory rollout samples: {GATED_ROLLOUT_SAMPLES_JSONL}",
        f"- Gated-GRAFT seed evals: {', '.join(GATED_SEED_EVALS)}",
        f"- Gated-GRAFT seed checkpoints: {', '.join(GATED_SEED_CKPTS)}",
        "",
        f"850scenes test N: 3584. Detected Fusion num_samples: {fusion_n if fusion_n is not None else 'unavailable'}. Detected Gated-GRAFT num_samples: {gated_n if gated_n is not None else 'unavailable'}.",
        "Fusion is a single-run reference in the multi-seed action summary.",
        "Gated-GRAFT multi-seed action summary reports mean ± standard deviation over seed0/seed1/seed2.",
        "",
        "Interpretation:",
        "- ADE/FDE are not the main advantage of Gated-GRAFT in the 850scenes results.",
        "- Fusion is better on ADE, FDE, and longitudinal rollout error.",
        "- Gated-GRAFT is better on lateral and final-lateral rollout error, indicating a lateral stability trade-off.",
        "- Gated-GRAFT slightly improves curvature MAE but worsens speed MAE and overall action L1 in the seed0 single-run comparison.",
        "",
        "Figure guidance:",
        "- Main text: fig_850_action_metrics, fig_850_rollout_tradeoff, fig_850_gated_multiseed_action.",
        "- Main text or appendix: fig_850_scenario_rollout, depending on space.",
        "- Appendix only: optional fig_850_gated_training_curve_mean_std if generated.",
        "",
        f"Optional training curve status: {training_curve_status}",
        "",
        "Generated files:",
    ]
    lines.extend(f"- {Path(path).name}" for path in outputs)
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    set_paper_style()
    output_dir = Path(args.output_dir)
    outputs: List[str] = []
    if plt is not None:
        plot_action_metrics(output_dir, args.dpi, outputs)
        plot_rollout_tradeoff(output_dir, args.dpi, outputs)
        plot_scenario_rollout(output_dir, args.dpi, outputs)
        plot_multiseed_action(output_dir, args.dpi, outputs)
        generated_curve, curve_status = plot_training_curve(
            output_dir,
            args.dpi,
            outputs,
            smooth_window=args.smooth_window,
            shade_alpha=args.shade_alpha,
        )
    else:
        warn("matplotlib is unavailable; using Pillow fallback for mandatory figures")
        draw_pil_grouped_bar(
            "fig_850_action_metrics",
            output_dir,
            ["Speed MAE", "Curv. MAE x100", "Overall L1"],
            [ACTION_METRICS["Fusion"][label] for label in ["Speed MAE", "Curv. MAE x100", "Overall L1"]],
            [ACTION_METRICS["Gated-GRAFT"][label] for label in ["Speed MAE", "Curv. MAE x100", "Overall L1"]],
            "Error (lower is better)",
            "Curvature slightly better;\nspeed/overall worse",
            outputs,
        )
        draw_pil_grouped_bar(
            "fig_850_rollout_tradeoff",
            output_dir,
            ["ADE", "FDE", "Long.", "Lat.", "Final Lat."],
            [ROLLOUT_METRICS["Fusion"][label] for label in ["ADE", "FDE", "Long.", "Lat.", "Final Lat."]],
            [ROLLOUT_METRICS["Gated-GRAFT"][label] for label in ["ADE", "FDE", "Long.", "Lat.", "Final Lat."]],
            "Rollout error (m)",
            "Fusion better ADE/FDE/long.;\nGated-GRAFT better lateral",
            outputs,
        )
        draw_pil_scenario_rollout(output_dir, outputs)
        draw_pil_multiseed_action(output_dir, outputs)
        generated_curve = False
        curve_status = "skipped: matplotlib unavailable in this environment"
    if not generated_curve:
        warn(curve_status)
    write_readme(output_dir, outputs, curve_status)
    info(f"wrote README: {output_dir / 'README.txt'}")
    for path in outputs:
        info(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
