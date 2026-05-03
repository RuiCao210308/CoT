#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from typing import Any, Iterable

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Quickly inspect future waypoint targets in action chunk JSONL.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max_preview", type=int, default=3)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def numeric_stats(values: Iterable[Any]):
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def shape_key(value):
    try:
        shape = tuple(np.asarray(value, dtype=object).shape)
    except Exception:
        shape = ()
    if not shape:
        return "missing_or_invalid"
    return "x".join(str(dim) for dim in shape)


def inspect_waypoints(jsonl_path: str, max_preview: int):
    shape_counts = Counter()
    x_values = []
    y_values = []
    distance_values = []
    previews = []
    records = 0
    missing = 0
    nonfinite = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records += 1
            target = record.get("target", {})
            waypoints = target.get("future_waypoints_local")
            shape_counts[shape_key(waypoints)] += 1
            if waypoints is None:
                missing += 1
                continue
            try:
                arr = np.asarray(waypoints, dtype=np.float64)
            except (TypeError, ValueError):
                missing += 1
                continue
            if arr.ndim != 2 or arr.shape[-1] != 2:
                continue
            finite_mask = np.isfinite(arr).all(axis=1)
            nonfinite += int(arr.shape[0] - np.count_nonzero(finite_mask))
            finite = arr[finite_mask]
            if finite.size:
                distance = np.linalg.norm(finite, axis=1)
                x_values.extend(finite[:, 0].tolist())
                y_values.extend(finite[:, 1].tolist())
                distance_values.extend(distance.tolist())
            if len(previews) < max_preview:
                metadata = record.get("metadata", {})
                previews.append(
                    {
                        "line": line_no,
                        "scene_name": metadata.get("scene_name"),
                        "frame_idx": metadata.get("frame_idx"),
                        "sample_token": metadata.get("sample_token"),
                        "future_waypoints_local": arr.tolist(),
                    }
                )

    return {
        "jsonl": jsonl_path,
        "records": records,
        "missing_waypoints": missing,
        "waypoint_nonfinite_rows": nonfinite,
        "waypoint_shape_counts": dict(shape_counts),
        "x_m_local": numeric_stats(x_values),
        "y_m_local": numeric_stats(y_values),
        "distance_m": numeric_stats(distance_values),
        "preview": previews,
    }


def main():
    args = parse_args()
    report = inspect_waypoints(args.jsonl, args.max_preview)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved waypoint inspect report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
