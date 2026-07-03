#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_METRICS = ["train/speed_l1", "train/curvature_l1", "train/gate_mean"]
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/paper_training_curves_850_tensorboard"
COLORS = {
    "Fusion-850": "#4D4D4D",
    "Gated-GRAFT-850": "#8B1A1A",
}
FALLBACK_COLORS = ["#1B7837", "#6A3D9A", "#E66101", "#5E81AC"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot paper-style training curves from TensorBoard scalars.")
    parser.add_argument(
        "--runs",
        action="append",
        required=True,
        help="Run spec in MethodName:/path/to/tensorboard/run_dir format. Repeat for multiple runs/seeds.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--alignment", choices=("common", "interpolate"), default="interpolate")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def info(message: str) -> None:
    print(f"[TensorBoardCurves] {message}")


def warn(message: str) -> None:
    print(f"[TensorBoardCurves] warning: {message}")


def parse_run_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"--runs must use MethodName:/path format, got: {spec}")
    method, path = spec.split(":", 1)
    method = method.strip()
    path = path.strip()
    if not method or not path:
        raise ValueError(f"--runs must use MethodName:/path format, got: {spec}")
    return method, path


def load_scalars(run_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise ImportError(
            "TensorBoard curve plotting requires tensorboard. Please run: pip install tensorboard"
        ) from exc

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"TensorBoard run_dir does not exist: {run_dir}")
    accumulator = EventAccumulator(run_dir)
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        steps = np.asarray([event.step for event in events], dtype=float)
        values = np.asarray([event.value for event in events], dtype=float)
        order = np.argsort(steps)
        curves[tag] = (steps[order], values[order])
    return curves


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(window, int(values.size))
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def align_curves(curves: Sequence[Tuple[np.ndarray, np.ndarray]], mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
    if len(curves) == 1:
        x, y = curves[0]
        return x, y.reshape(1, -1), "single run; no alignment needed"
    if mode == "common":
        common = set(curves[0][0].tolist())
        for xs, _ in curves[1:]:
            common &= set(xs.tolist())
        if not common:
            raise ValueError("no common scalar steps across runs; use --alignment interpolate")
        grid = np.asarray(sorted(common), dtype=float)
        rows = []
        for xs, ys in curves:
            mapping = {float(x): float(y) for x, y in zip(xs, ys)}
            rows.append([mapping[float(x)] for x in grid])
        return grid, np.asarray(rows, dtype=float), "common scalar step merge"

    start = max(float(xs.min()) for xs, _ in curves)
    end = min(float(xs.max()) for xs, _ in curves)
    if start > end:
        raise ValueError("runs have no overlapping scalar step range")
    grid_values = sorted({float(x) for xs, _ in curves for x in xs if start <= float(x) <= end})
    if not grid_values:
        raise ValueError("no scalar points inside overlapping step range")
    grid = np.asarray(grid_values, dtype=float)
    rows = [np.interp(grid, xs, ys) for xs, ys in curves]
    return grid, np.asarray(rows, dtype=float), "linear interpolation on existing scalar step grid"


def metric_label(tag: str) -> str:
    labels = {
        "train/speed_l1": "Speed L1",
        "train/curvature_l1": "Curvature L1",
        "train/gate_mean": "Gate mean",
        "train/residual_abs_mean": "Residual abs. mean",
        "train/loss": "Loss",
        "train/total_loss": "Total loss",
        "train/action_loss": "Action loss",
    }
    return labels.get(tag, tag.replace("train/", "").replace("_", " "))


def color_for_method(method: str, index: int) -> str:
    if method in COLORS:
        return COLORS[method]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


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


def write_readme(
    output_dir: Path,
    run_specs: Sequence[Tuple[str, str]],
    metrics: Sequence[str],
    method_run_counts: Dict[str, int],
    alignment_notes: Sequence[str],
    smooth_window: int,
    alignment: str,
    outputs: Sequence[str],
) -> None:
    lines = [
        "TensorBoard-first 850scenes training curves",
        "",
        "All curves are read from TensorBoard scalar event files. Checkpoint train_history is not used.",
        "Training scripts log one scalar point every log_interval_steps=200 using the average of the most recent 200 batches.",
        "Single-run methods are plotted as a line without a standard-deviation band.",
        "Methods with multiple runs are plotted as mean ± standard deviation.",
        f"Moving-average smoothing window: {smooth_window}. If a run has fewer points, smoothing is automatically reduced.",
        f"Requested alignment mode: {alignment}.",
        "",
        "Run sources:",
    ]
    lines.extend(f"- {method}: {path}" for method, path in run_specs)
    lines.extend(["", "Run counts:"])
    lines.extend(f"- {method}: {count}" for method, count in sorted(method_run_counts.items()))
    lines.extend(["", "Metrics:"])
    lines.extend(f"- {metric}" for metric in metrics)
    lines.extend(["", "Alignment notes:"])
    lines.extend(f"- {note}" for note in alignment_notes)
    lines.extend(["", "Generated files:"])
    lines.extend(f"- {Path(path).name}" for path in outputs)
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_tensorboard_training_curves.py requires matplotlib.") from exc

    run_specs = [parse_run_spec(spec) for spec in args.runs]
    grouped_paths: Dict[str, List[str]] = defaultdict(list)
    for method, path in run_specs:
        grouped_paths[method].append(path)

    loaded = {(method, path): load_scalars(path) for method, paths in grouped_paths.items() for path in paths}
    set_paper_style(plt)
    fig, axes = plt.subplots(1, len(args.metrics), figsize=(2.35 * len(args.metrics), 2.25))
    if len(args.metrics) == 1:
        axes = [axes]

    alignment_notes: List[str] = []
    for metric_index, (ax, metric) in enumerate(zip(axes, args.metrics)):
        plotted_any = False
        for method_index, (method, paths) in enumerate(grouped_paths.items()):
            curves = []
            for path in paths:
                run_curves = loaded[(method, path)]
                if metric not in run_curves:
                    warn(f"metric missing for {method}: {metric} in {path}")
                    continue
                curves.append(run_curves[metric])
            if not curves:
                continue
            x_grid, values, align_note = align_curves(curves, args.alignment)
            values = np.asarray([smooth(row, args.smooth_window) for row in values], dtype=float)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else None
            color = color_for_method(method, method_index)
            ax.plot(x_grid, mean, color=color, linewidth=2.2, label=method)
            if std is not None:
                ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
            alignment_notes.append(f"{metric} / {method}: {align_note}; plotted_runs={values.shape[0]}")
            plotted_any = True
        if not plotted_any:
            warn(f"no runs contained metric: {metric}")
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric_label(metric))
        ax.grid(True, alpha=0.20, linewidth=0.6)
        ax.legend(frameon=False, loc="best")
    if args.title:
        fig.suptitle(args.title, fontsize=9.0)
    fig.tight_layout(w_pad=1.1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "pdf"):
        path = output_dir / f"fig_850_tensorboard_training_curves.{ext}"
        fig.savefig(path, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
        outputs.append(str(path))
        info(f"wrote {path}")
    plt.close(fig)
    write_readme(
        output_dir=output_dir,
        run_specs=run_specs,
        metrics=args.metrics,
        method_run_counts={method: len(paths) for method, paths in grouped_paths.items()},
        alignment_notes=alignment_notes,
        smooth_window=args.smooth_window,
        alignment=args.alignment,
        outputs=outputs,
    )
    info(f"wrote {output_dir / 'README.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
