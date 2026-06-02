#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "plot_paper_training_curves.py requires numpy. "
        "Install it in the plotting environment, then rerun the command."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "plot_paper_training_curves.py requires matplotlib. "
        "Install it in the plotting environment, then rerun the command."
    ) from exc


PAPER_COLORS = ["#8B1A1A", "#4D4D4D", "#1B7837", "#6A3D9A", "#E66101", "#5E81AC"]
DEFAULT_METRICS = [
    "avg_total_loss",
    "avg_action_loss",
    "avg_loss",
    "avg_speed_l1",
    "avg_curvature_l1",
    "avg_geometry_descriptor_loss",
    "avg_gate_loss",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot paper-style multi-run training curves with mean and standard-deviation bands."
    )
    parser.add_argument(
        "--runs",
        action="append",
        required=True,
        help="Named run path in model:path form. Repeat for each seed/run. Path can be a checkpoint, JSON, or directory.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--x_key", default=None, help="History x key. Defaults to epoch, then global_step, then step.")
    parser.add_argument("--alignment", choices=("common", "interpolate"), default="common")
    parser.add_argument("--smooth_window", type=int, default=1, help="Moving-average window. 1 disables smoothing.")
    parser.add_argument("--shade_alpha", type=float, default=0.16)
    parser.add_argument("--linewidth", type=float, default=2.25)
    parser.add_argument("--fig_width", type=float, default=3.35, help="Single-column width in inches.")
    parser.add_argument("--fig_height", type=float, default=2.35)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_named_path(value: str) -> Tuple[str, Path]:
    if ":" not in value:
        raise ValueError(f"--runs must be model:path, got {value}")
    name, path = value.split(":", 1)
    if not name or not path:
        raise ValueError(f"--runs must be model:path, got {value}")
    return name, Path(path)


def warn(message: str) -> None:
    print(f"[PaperTrainingCurves] warning: {message}")


def info(message: str) -> None:
    print(f"[PaperTrainingCurves] {message}")


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"loading checkpoint {path} requires torch") from exc
    return torch.load(str(path), map_location="cpu")


def candidate_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        warn(f"path does not exist: {path}")
        return []
    candidates = []
    for pattern in ("*.json", "*.pt", "*.pth"):
        candidates.extend(sorted(path.glob(pattern)))
    preferred = [item for item in candidates if item.name in {"fusion_action_head.pt", "action_head.pt", "gated_geometry_residual_fusion_head.pt"}]
    return preferred or candidates


def extract_history_from_data(data: Dict[str, Any], source: Path) -> Optional[List[Dict[str, Any]]]:
    history = data.get("train_history")
    if isinstance(history, list) and history:
        return [item for item in history if isinstance(item, dict)]
    if isinstance(data.get("history"), list):
        return [item for item in data["history"] if isinstance(item, dict)]
    warn(f"no train_history found in {source}")
    return None


def load_history(path: Path) -> Optional[List[Dict[str, Any]]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = load_json(path)
        elif suffix in {".pt", ".pth"}:
            data = load_torch_checkpoint(path)
        else:
            warn(f"unsupported run file suffix: {path}")
            return None
    except Exception as exc:
        warn(f"failed to load {path}: {exc}")
        return None
    if not isinstance(data, dict):
        warn(f"run file does not contain a dict: {path}")
        return None
    return extract_history_from_data(data, path)


def load_runs(named_paths: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for value in named_paths:
        name, path = parse_named_path(value)
        files = candidate_files(path)
        if not files:
            warn(f"no run files found for {name}: {path}")
            continue
        loaded_any = False
        for file_path in files:
            history = load_history(file_path)
            if history:
                grouped.setdefault(name, []).append({"path": str(file_path), "history": history})
                loaded_any = True
        if not loaded_any:
            warn(f"no usable train_history for {name}: {path}")
    return grouped


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def history_xy(history: List[Dict[str, Any]], metric: str, x_key: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for idx, row in enumerate(history):
        y = finite_number(row.get(metric))
        if y is None:
            continue
        if x_key:
            x = finite_number(row.get(x_key))
        else:
            x = finite_number(row.get("epoch"))
            if x is None:
                x = finite_number(row.get("global_step"))
            if x is None:
                x = finite_number(row.get("step"))
            if x is None:
                x = float(idx + 1)
        if x is None:
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(np.asarray(xs, dtype=float))
    return np.asarray(xs, dtype=float)[order], np.asarray(ys, dtype=float)[order]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def align_common(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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


def align_interpolate(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    start = max(float(xs.min()) for xs, _ in curves)
    end = min(float(xs.max()) for xs, _ in curves)
    if start > end:
        return None
    lengths = [len(xs[(xs >= start) & (xs <= end)]) for xs, _ in curves]
    grid_len = max(2, min(lengths) if lengths else 0)
    if grid_len <= 0:
        return None
    x_grid = np.linspace(start, end, grid_len)
    rows = [np.interp(x_grid, xs, ys) for xs, ys in curves]
    return x_grid, np.asarray(rows, dtype=float)


def aligned_metric(
    runs: List[Dict[str, Any]],
    metric: str,
    x_key: Optional[str],
    alignment: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    curves = []
    for run in runs:
        xs, ys = history_xy(run["history"], metric, x_key)
        if xs.size and ys.size:
            curves.append((xs, ys))
    if not curves:
        return None
    if len(curves) == 1:
        return curves[0][0], curves[0][1][None, :], "single-run"
    if alignment == "common":
        aligned = align_common(curves)
        if aligned is not None:
            return aligned[0], aligned[1], "common-step merge"
        info(f"{metric}: no common x values across runs; falling back to interpolation")
    aligned = align_interpolate(curves)
    if aligned is None:
        warn(f"{metric}: unable to align curves")
        return None
    return aligned[0], aligned[1], "interpolation to shared x range"


def metric_label(metric: str) -> str:
    labels = {
        "avg_total_loss": "Total loss",
        "avg_action_loss": "Action loss",
        "avg_loss": "Loss",
        "avg_speed_l1": "Speed L1",
        "avg_curvature_l1": "Curvature L1",
        "avg_geometry_descriptor_loss": "Geometry descriptor loss",
        "avg_gate_loss": "Gate loss",
    }
    return labels.get(metric, metric.replace("_", " "))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> List[str]:
    paths = []
    for ext in ("png", "pdf"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        paths.append(str(out))
    plt.close(fig)
    return paths


def plot_metric(
    grouped: Dict[str, List[Dict[str, Any]]],
    metric: str,
    args,
    color_map: Dict[str, str],
) -> List[str]:
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    plotted = False
    for name, runs in grouped.items():
        aligned = aligned_metric(runs, metric, args.x_key, args.alignment)
        if aligned is None:
            continue
        x_grid, values, mode = aligned
        values = np.asarray([smooth(row, args.smooth_window) for row in values], dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
        color = color_map[name]
        ax.plot(x_grid, mean, color=color, linewidth=args.linewidth, label=f"{name} (n={values.shape[0]})")
        if values.shape[0] > 1:
            ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=args.shade_alpha, linewidth=0)
        info(f"{metric}/{name}: {mode}, runs={values.shape[0]}, points={x_grid.size}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return []
    x_label = args.x_key or "epoch / step"
    ax.set_xlabel(x_label.replace("_", " "))
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.02)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return save_figure(fig, output_dir, f"training_curve_{metric}", args.dpi)


def main():
    args = parse_args()
    set_paper_style()
    grouped = load_runs(args.runs)
    if not grouped:
        raise SystemExit("no usable runs found")
    color_map = {name: PAPER_COLORS[idx % len(PAPER_COLORS)] for idx, name in enumerate(grouped.keys())}
    outputs = []
    for metric in args.metrics:
        outputs.extend(plot_metric(grouped, metric, args, color_map))
    if not outputs:
        raise SystemExit("no figures were generated; check --metrics and train_history keys")
    for path in outputs:
        info(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
