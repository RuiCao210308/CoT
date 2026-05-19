#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


METRIC_KEYS = [
    "rollout_ade",
    "rollout_fde",
    "longitudinal_mae",
    "lateral_mae",
    "final_lateral_error",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trajectory rollout errors from action eval records.")
    parser.add_argument(
        "--records",
        action="append",
        required=True,
        help="Named records JSONL in name:path form. Can be repeated.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_txt", required=True)
    parser.add_argument("--output_samples_jsonl", type=str, default=None)
    parser.add_argument("--dt", type=float, default=0.5)
    return parser.parse_args()


def parse_named_path(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"--records must be name:path, got {value}")
    name, path = value.split(":", 1)
    if not name or not path:
        raise ValueError(f"--records must be name:path, got {value}")
    return name, path


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = {"parse_success": False, "failure_reason": "json_decode_error"}
            record["_line_no"] = line_no
            records.append(record)
    return records


def finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def as_matrix(value: Any) -> Optional[List[List[float]]]:
    if not isinstance(value, list) or not value:
        return None
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) < 2:
            return None
        x = finite_number(row[0])
        y = finite_number(row[1])
        if x is None or y is None:
            return None
        rows.append([x, y])
    return rows


def action_train_scale(record: Dict[str, Any], pred: bool) -> Optional[List[List[float]]]:
    if pred:
        return as_matrix(record.get("pred_action") or record.get("parsed_action"))
    target_train = as_matrix(record.get("target_action_train_scale"))
    if target_train is not None:
        return target_train
    target = as_matrix(record.get("target_action"))
    if target is None:
        return None
    return [[row[0], row[1] * 100.0] for row in target]


def target_waypoints(record: Dict[str, Any]) -> Optional[List[List[float]]]:
    return as_matrix(record.get("target_waypoints_local"))


def rollout_action(action_train_scale_value: List[List[float]], dt: float) -> List[List[float]]:
    """Roll out [speed_mps, curvature_x100] into local [x_m, y_m] waypoints."""
    x = 0.0
    y = 0.0
    heading = 0.0
    points = []
    for speed_mps, curvature_x100 in action_train_scale_value:
        curvature_1pm = curvature_x100 / 100.0
        distance = speed_mps * dt
        heading_mid = heading + 0.5 * curvature_1pm * distance
        x += distance * math.cos(heading_mid)
        y += distance * math.sin(heading_mid)
        heading += curvature_1pm * distance
        points.append([x, y])
    return points


def target_scores(target_action: Optional[List[List[float]]]) -> Dict[str, Optional[float]]:
    scores = {"target_abs_curvature_x100": None, "target_speed_change": None}
    if not target_action:
        return scores
    speeds = [row[0] for row in target_action]
    curvatures = [row[1] for row in target_action]
    scores["target_abs_curvature_x100"] = mean([abs(value) for value in curvatures])
    scores["target_speed_change"] = max(speeds) - min(speeds) if speeds else None
    return scores


def sample_identity(record: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    return {
        "model": model_name,
        "scene_name": record.get("scene_name"),
        "frame_idx": record.get("frame_idx"),
        "sample_token": record.get("sample_token"),
        "record_index": record.get("record_index"),
        "line_no": record.get("line_no") or record.get("_line_no"),
    }


def compute_rollout_errors(record: Dict[str, Any], model_name: str, dt: float) -> Optional[Dict[str, Any]]:
    if "parse_success" in record and not bool(record.get("parse_success")):
        return None
    pred_action = action_train_scale(record, pred=True)
    target_action = action_train_scale(record, pred=False)
    if not pred_action or not target_action:
        return None

    pred_waypoints = rollout_action(pred_action, dt=dt)
    gt_waypoints = target_waypoints(record)
    gt_source = "target_waypoints_local"
    if not gt_waypoints:
        gt_waypoints = rollout_action(target_action, dt=dt)
        gt_source = "target_action_rollout"

    n = min(len(pred_waypoints), len(gt_waypoints))
    if n <= 0:
        return None
    pred_waypoints = pred_waypoints[:n]
    gt_waypoints = gt_waypoints[:n]
    dx = [pred_waypoints[idx][0] - gt_waypoints[idx][0] for idx in range(n)]
    dy = [pred_waypoints[idx][1] - gt_waypoints[idx][1] for idx in range(n)]
    distances = [math.sqrt(dx[idx] * dx[idx] + dy[idx] * dy[idx]) for idx in range(n)]

    result = {
        **sample_identity(record, model_name),
        "gt_waypoint_source": gt_source,
        "rollout_ade": mean(distances),
        "rollout_fde": distances[-1],
        "longitudinal_mae": mean([abs(value) for value in dx]),
        "lateral_mae": mean([abs(value) for value in dy]),
        "final_lateral_error": abs(dy[-1]),
        "pred_rollout_waypoints": pred_waypoints,
        "gt_waypoints": gt_waypoints,
    }
    result.update(target_scores(target_action))
    return result


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def top_indices(records: List[Dict[str, Any]], score_key: str, fraction: float = 0.2) -> set:
    scored = [
        (idx, finite_number(record.get(score_key)))
        for idx, record in enumerate(records)
        if finite_number(record.get(score_key)) is not None
    ]
    if not scored:
        return set()
    scored.sort(key=lambda item: item[1], reverse=True)
    count = max(1, math.ceil(len(scored) * fraction))
    return {idx for idx, _ in scored[:count]}


def bottom_indices(records: List[Dict[str, Any]], score_key: str, fraction: float = 0.2) -> set:
    scored = [
        (idx, finite_number(record.get(score_key)))
        for idx, record in enumerate(records)
        if finite_number(record.get(score_key)) is not None
    ]
    if not scored:
        return set()
    scored.sort(key=lambda item: item[1])
    count = max(1, math.ceil(len(scored) * fraction))
    return {idx for idx, _ in scored[:count]}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {"count": len(records)}
    for key in METRIC_KEYS:
        values = [finite_number(record.get(key)) for record in records]
        values = [value for value in values if value is not None]
        result[key] = mean(values)
    return result


def build_groups(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    high_curvature = top_indices(records, "target_abs_curvature_x100")
    speed_changing = top_indices(records, "target_speed_change")
    low_curvature = bottom_indices(records, "target_abs_curvature_x100")
    low_speed_change = bottom_indices(records, "target_speed_change")
    steady_indices = low_curvature & low_speed_change
    return {
        "all": records,
        "high_curvature_top20": [record for idx, record in enumerate(records) if idx in high_curvature],
        "speed_changing_top20": [record for idx, record in enumerate(records) if idx in speed_changing],
        "steady_low_curvature": [record for idx, record in enumerate(records) if idx in steady_indices],
    }


def analyze_one(name: str, path: str, dt: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw_records = load_jsonl(path)
    sample_errors = []
    skipped = 0
    for record in raw_records:
        errors = compute_rollout_errors(record, name, dt=dt)
        if errors is None:
            skipped += 1
            continue
        sample_errors.append(errors)
    groups = build_groups(sample_errors)
    return (
        {
            "name": name,
            "path": path,
            "total_records": len(raw_records),
            "evaluated_records": len(sample_errors),
            "skipped_records": skipped,
            "groups": {group_name: summarize(group_records) for group_name, group_records in groups.items()},
        },
        sample_errors,
    )


def fmt(value: Any) -> str:
    number = finite_number(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def build_text(report: Dict[str, Any]) -> str:
    lines = ["Trajectory Rollout Analysis", "", f"dt={report['dt']}", ""]
    for item in report["records"]:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"path: {item['path']}",
                f"evaluated_records={item['evaluated_records']} skipped_records={item['skipped_records']}",
                "",
                "| group | count | rollout_ade | rollout_fde | longitudinal_mae | lateral_mae | final_lateral_error |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group_name, group in item["groups"].items():
            lines.append(
                "| {name} | {count} | {ade} | {fde} | {longitudinal} | {lateral} | {final_lat} |".format(
                    name=group_name,
                    count=group["count"],
                    ade=fmt(group.get("rollout_ade")),
                    fde=fmt(group.get("rollout_fde")),
                    longitudinal=fmt(group.get("longitudinal_mae")),
                    lateral=fmt(group.get("lateral_mae")),
                    final_lat=fmt(group.get("final_lateral_error")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: str, content: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = parse_args()
    analyses = []
    all_sample_errors = []
    for value in args.records:
        name, path = parse_named_path(value)
        analysis, sample_errors = analyze_one(name, path, dt=args.dt)
        analyses.append(analysis)
        all_sample_errors.extend(sample_errors)
    report = {"dt": args.dt, "records": analyses}
    write_json(args.output_json, report)
    write_text(args.output_txt, build_text(report))
    if args.output_samples_jsonl:
        write_jsonl(args.output_samples_jsonl, all_sample_errors)
    print(f"[TrajectoryRollout] wrote json: {args.output_json}")
    print(f"[TrajectoryRollout] wrote txt: {args.output_txt}")
    if args.output_samples_jsonl:
        print(f"[TrajectoryRollout] wrote samples: {args.output_samples_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
