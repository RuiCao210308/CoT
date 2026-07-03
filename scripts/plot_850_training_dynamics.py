#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


FUSION_COLOR = "#4D4D4D"
GATED_COLOR = "#8B1A1A"
PURPLE = "#6A3D9A"

DEFAULT_FUSION_HISTORY = "/root/autodl-tmp/fusion_trainval_850scenes_e5_logcurve/step_history.jsonl"
DEFAULT_GATED_HISTORIES = [
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed0_logcurve/step_history.jsonl",
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed1_logcurve/step_history.jsonl",
    "/root/autodl-tmp/gated_graft_trainval_850scenes_seed2_logcurve/step_history.jsonl",
]
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/paper_training_curves_850_final"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot 850scenes step-level training dynamics.")
    parser.add_argument("--fusion_history", default=DEFAULT_FUSION_HISTORY)
    parser.add_argument("--gated_history", action="append", default=None)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--shade_alpha", type=float, default=0.16)
    return parser.parse_args()


def info(message: str) -> None:
    print(f"[Plot850TrainingDynamics] {message}")


def load_history(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}") from exc
            if not isinstance(row, dict):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"empty history: {path}")
    return rows


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def history_curve(history: Sequence[Dict[str, Any]], metric: str) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in history:
        step = finite_float(row.get("step"))
        value = finite_float(row.get(metric))
        if step is None or value is None:
            continue
        xs.append(step)
        ys.append(value)
    if not xs:
        raise ValueError(f"metric missing from history: {metric}")
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


def align_seed_curves(curves: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, str]:
    if len(curves) == 0:
        raise ValueError("no curves to align")
    common_steps = set(curves[0][0].tolist())
    for xs, _ in curves[1:]:
        common_steps &= set(xs.tolist())
    if common_steps:
        grid = np.asarray(sorted(common_steps), dtype=float)
        rows = []
        for xs, ys in curves:
            mapping = {float(x): float(y) for x, y in zip(xs, ys)}
            rows.append([mapping[float(x)] for x in grid])
        return grid, np.asarray(rows, dtype=float), "common-step merge"

    start = max(float(xs.min()) for xs, _ in curves)
    end = min(float(xs.max()) for xs, _ in curves)
    if start > end:
        raise ValueError("seed histories have no overlapping step range")
    grid = sorted({float(x) for xs, _ in curves for x in xs if start <= float(x) <= end})
    if not grid:
        grid = np.linspace(start, end, num=min(len(xs) for xs, _ in curves))
    grid_arr = np.asarray(grid, dtype=float)
    rows = [np.interp(grid_arr, xs, ys) for xs, ys in curves]
    return grid_arr, np.asarray(rows, dtype=float), "linear interpolation on overlapping step grid"


def set_paper_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8.5,
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


def plot_with_band(ax, x: np.ndarray, values: np.ndarray, label: str, color: str, shade_alpha: float, smooth_window: int):
    values = np.asarray([smooth(row, smooth_window) for row in values], dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    ax.plot(x, mean, color=color, linewidth=2.2, label=label)
    if values.shape[0] > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=shade_alpha, linewidth=0)


def write_readme(
    output_dir: Path,
    fusion_history: str,
    gated_histories: Sequence[str],
    outputs: Sequence[str],
    alignment_notes: Sequence[str],
    smooth_window: int,
) -> None:
    lines = [
        "850scenes training dynamics",
        "",
        "These curves are from logging-only experiments rerun on the 850scenes train split.",
        "They use existing Qwen hidden cache files and do not call Qwen generation or rebuild cache.",
        "Each plotted point is the average over the most recent log_interval_steps=200 training batches.",
        "Fusion is a single-run reference. Gated-GRAFT is mean ± standard deviation over seed0/seed1/seed2.",
        "Lower is better for Speed L1 and Curvature L1. Gate/residual diagnostics are not accuracy metrics.",
        f"Optional smoothing window applied by this plotting script: {smooth_window}.",
        "",
        "Input histories:",
        f"- Fusion: {fusion_history}",
    ]
    lines.extend(f"- Gated-GRAFT seed: {path}" for path in gated_histories)
    lines.extend(
        [
            "",
            "Step alignment:",
        ]
    )
    lines.extend(f"- {note}" for note in alignment_notes)
    lines.extend(
        [
            "",
            "Generated files:",
        ]
    )
    lines.extend(f"- {Path(path).name}" for path in outputs)
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def blend_with_white(value: str, alpha: float) -> Tuple[int, int, int]:
    rgb = hex_to_rgb(value)
    return tuple(int((1.0 - alpha) * 255 + alpha * channel) for channel in rgb)


def pil_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def save_pil_image(image, output_dir: Path, stem: str) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "pdf"):
        path = output_dir / f"{stem}.{ext}"
        if ext == "png":
            image.save(path)
        else:
            image.convert("RGB").save(path, "PDF", resolution=300.0)
        outputs.append(str(path))
    return outputs


def draw_curve_panel(draw, box, x, curves, ylabel: str, labels: Sequence[Tuple[str, str]], font, small_font):
    left, top, right, bottom = box
    axis = hex_to_rgb("#333333")
    grid = blend_with_white("#000000", 0.18)
    all_x = np.concatenate([np.asarray(item["x"], dtype=float) for item in curves])
    all_y = np.concatenate(
        [
            np.asarray(item["mean"], dtype=float)
            if item.get("std") is None
            else np.concatenate([np.asarray(item["mean"]) - np.asarray(item["std"]), np.asarray(item["mean"]) + np.asarray(item["std"])])
            for item in curves
        ]
    )
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.12
    y_min -= y_pad
    y_max += y_pad

    def sx(value):
        return left + (float(value) - x_min) / (x_max - x_min) * (right - left)

    def sy(value):
        return bottom - (float(value) - y_min) / (y_max - y_min) * (bottom - top)

    for frac in np.linspace(0.0, 1.0, 4):
        y_pos = top + (bottom - top) * frac
        draw.line((left, y_pos, right, y_pos), fill=grid, width=1)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)

    for item in curves:
        color = hex_to_rgb(item["color"])
        x_values = np.asarray(item["x"], dtype=float)
        mean = np.asarray(item["mean"], dtype=float)
        std = item.get("std")
        if std is not None:
            std = np.asarray(std, dtype=float)
            upper = [(sx(xv), sy(yv)) for xv, yv in zip(x_values, mean + std)]
            lower = [(sx(xv), sy(yv)) for xv, yv in zip(x_values[::-1], (mean - std)[::-1])]
            draw.polygon(upper + lower, fill=blend_with_white(item["color"], 0.18))
        points = [(sx(xv), sy(yv)) for xv, yv in zip(x_values, mean)]
        if len(points) > 1:
            draw.line(points, fill=color, width=5, joint="curve")
    draw.text(((left + right) / 2, bottom + 34), "Training step", font=small_font, fill=axis, anchor="mm")
    draw.text((left - 38, (top + bottom) / 2), ylabel, font=small_font, fill=axis, anchor="mm")
    legend_x = left
    for label, color in labels:
        draw.line((legend_x, top - 25, legend_x + 34, top - 25), fill=hex_to_rgb(color), width=5)
        draw.text((legend_x + 42, top - 36), label, font=font, fill=axis)
        legend_x += 190


def plot_with_pillow(
    output_dir: Path,
    fusion_speed: Tuple[np.ndarray, np.ndarray],
    fusion_curv: Tuple[np.ndarray, np.ndarray],
    gated_speed: Tuple[np.ndarray, np.ndarray],
    gated_curv: Tuple[np.ndarray, np.ndarray],
    gated_gate: Tuple[np.ndarray, np.ndarray],
    gated_resid: Tuple[np.ndarray, np.ndarray],
    smooth_window: int,
) -> List[str]:
    from PIL import Image, ImageDraw

    width, height = int(7.05 * 300), int(2.25 * 300)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = pil_font(21)
    small = pil_font(20)
    panel_w = 565
    gap = 85
    top, bottom = 76, 570
    boxes = [
        (82, top, 82 + panel_w, bottom),
        (82 + panel_w + gap, top, 82 + 2 * panel_w + gap, bottom),
        (82 + 2 * (panel_w + gap), top, 82 + 3 * panel_w + 2 * gap, bottom),
    ]

    gated_speed_mean = smooth(gated_speed[1].mean(axis=0), smooth_window)
    gated_speed_std = gated_speed[1].std(axis=0, ddof=1)
    gated_curv_mean = smooth(gated_curv[1].mean(axis=0), smooth_window)
    gated_curv_std = gated_curv[1].std(axis=0, ddof=1)
    gated_gate_mean = smooth(gated_gate[1].mean(axis=0), smooth_window)
    gated_gate_std = gated_gate[1].std(axis=0, ddof=1)
    gated_resid_mean = smooth(gated_resid[1].mean(axis=0), smooth_window)
    gated_resid_std = gated_resid[1].std(axis=0, ddof=1)

    draw_curve_panel(
        draw,
        boxes[0],
        None,
        [
            {"x": fusion_speed[0], "mean": smooth(fusion_speed[1], smooth_window), "color": FUSION_COLOR},
            {"x": gated_speed[0], "mean": gated_speed_mean, "std": gated_speed_std, "color": GATED_COLOR},
        ],
        "Speed L1",
        [("Fusion", FUSION_COLOR), ("Gated-GRAFT", GATED_COLOR)],
        font,
        small,
    )
    draw_curve_panel(
        draw,
        boxes[1],
        None,
        [
            {"x": fusion_curv[0], "mean": smooth(fusion_curv[1], smooth_window), "color": FUSION_COLOR},
            {"x": gated_curv[0], "mean": gated_curv_mean, "std": gated_curv_std, "color": GATED_COLOR},
        ],
        "Curvature L1",
        [("Fusion", FUSION_COLOR), ("Gated-GRAFT", GATED_COLOR)],
        font,
        small,
    )
    draw_curve_panel(
        draw,
        boxes[2],
        None,
        [
            {"x": gated_gate[0], "mean": gated_gate_mean, "std": gated_gate_std, "color": GATED_COLOR},
            {"x": gated_resid[0], "mean": gated_resid_mean, "std": gated_resid_std, "color": PURPLE},
        ],
        "Diagnostic value",
        [("Gate mean", GATED_COLOR), ("Residual abs.", PURPLE)],
        font,
        small,
    )
    return save_pil_image(image, output_dir, "fig_850_training_dynamics")


def main():
    args = parse_args()
    gated_paths = args.gated_history or DEFAULT_GATED_HISTORIES
    fusion_history = load_history(args.fusion_history)
    gated_histories = [load_history(path) for path in gated_paths]

    fusion_speed_x, fusion_speed_y = history_curve(fusion_history, "speed_l1")
    fusion_curv_x, fusion_curv_y = history_curve(fusion_history, "curvature_l1")

    alignment_notes: List[str] = []
    gated_speed_grid, gated_speed_values, speed_alignment = align_seed_curves(
        [history_curve(history, "speed_l1") for history in gated_histories]
    )
    alignment_notes.append(f"speed_l1: {speed_alignment}")
    gated_curv_grid, gated_curv_values, curv_alignment = align_seed_curves(
        [history_curve(history, "curvature_l1") for history in gated_histories]
    )
    alignment_notes.append(f"curvature_l1: {curv_alignment}")
    gated_gate_grid, gated_gate_values, gate_alignment = align_seed_curves(
        [history_curve(history, "gate_mean") for history in gated_histories]
    )
    alignment_notes.append(f"gate_mean: {gate_alignment}")
    gated_resid_grid, gated_resid_values, resid_alignment = align_seed_curves(
        [history_curve(history, "residual_abs_mean") for history in gated_histories]
    )
    alignment_notes.append(f"residual_abs_mean: {resid_alignment}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        info("matplotlib unavailable; using Pillow fallback")
        outputs = plot_with_pillow(
            output_dir,
            fusion_speed=(fusion_speed_x, fusion_speed_y),
            fusion_curv=(fusion_curv_x, fusion_curv_y),
            gated_speed=(gated_speed_grid, gated_speed_values),
            gated_curv=(gated_curv_grid, gated_curv_values),
            gated_gate=(gated_gate_grid, gated_gate_values),
            gated_resid=(gated_resid_grid, gated_resid_values),
            smooth_window=args.smooth_window,
        )
        for path in outputs:
            info(f"wrote {path}")
    else:
        set_paper_style(plt)
        fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25))
        axes[0].plot(
            fusion_speed_x,
            smooth(fusion_speed_y, args.smooth_window),
            color=FUSION_COLOR,
            linewidth=2.2,
            label="Fusion",
        )
        plot_with_band(
            axes[0],
            gated_speed_grid,
            gated_speed_values,
            label="Gated-GRAFT",
            color=GATED_COLOR,
            shade_alpha=args.shade_alpha,
            smooth_window=args.smooth_window,
        )
        axes[0].set_xlabel("Training step")
        axes[0].set_ylabel("Speed L1")

        axes[1].plot(
            fusion_curv_x,
            smooth(fusion_curv_y, args.smooth_window),
            color=FUSION_COLOR,
            linewidth=2.2,
            label="Fusion",
        )
        plot_with_band(
            axes[1],
            gated_curv_grid,
            gated_curv_values,
            label="Gated-GRAFT",
            color=GATED_COLOR,
            shade_alpha=args.shade_alpha,
            smooth_window=args.smooth_window,
        )
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Curvature L1")

        plot_with_band(
            axes[2],
            gated_gate_grid,
            gated_gate_values,
            label="Gate mean",
            color=GATED_COLOR,
            shade_alpha=args.shade_alpha,
            smooth_window=args.smooth_window,
        )
        plot_with_band(
            axes[2],
            gated_resid_grid,
            gated_resid_values,
            label="Residual abs. mean",
            color=PURPLE,
            shade_alpha=0.12,
            smooth_window=args.smooth_window,
        )
        axes[2].set_xlabel("Training step")
        axes[2].set_ylabel("Diagnostic value")

        for ax in axes:
            ax.grid(True, alpha=0.20, linewidth=0.6)
            ax.legend(frameon=False, loc="best")
        fig.tight_layout(w_pad=1.1)

        outputs = []
        for ext in ("png", "pdf"):
            path = output_dir / f"fig_850_training_dynamics.{ext}"
            fig.savefig(path, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
            outputs.append(str(path))
            info(f"wrote {path}")
        plt.close(fig)
    write_readme(
        output_dir,
        fusion_history=args.fusion_history,
        gated_histories=gated_paths,
        outputs=outputs,
        alignment_notes=alignment_notes,
        smooth_window=args.smooth_window,
    )
    info(f"wrote {output_dir / 'README.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
