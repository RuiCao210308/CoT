#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


METRIC_KEYS = [
    "speed_mae_mps",
    "curvature_mae_x100",
    "overall_l1_train_scale",
    "speed_delta_mae",
    "curvature_delta_mae",
]
TEXT_FLAT_KEYWORDS = ("flat_repeated_pair", "flat_repeat", "flat repeated")
TEXT_REUSED_KEYWORDS = (
    "reused_history_pairs",
    "mostly_reused_history_pairs",
    "copied_history_prefix",
    "copied_history_suffix",
    "reused history",
    "copied history",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze action eval records by scenario groups.")
    parser.add_argument(
        "--records",
        action="append",
        required=True,
        help="Named records JSONL in name:path form. Can be repeated.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_txt", required=True)
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
        speed = finite_number(row[0])
        curvature = finite_number(row[1])
        if speed is None or curvature is None:
            return None
        rows.append([speed, curvature])
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


def flatten_reason(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_reason(item) for item in value)
    if isinstance(value, dict):
        pieces = []
        for key, item in value.items():
            pieces.append(str(key))
            pieces.append(flatten_reason(item))
        return " ".join(pieces)
    return str(value)


def text_flag(record: Dict[str, Any], direct_keys: List[str], keywords) -> bool:
    for key in direct_keys:
        value = record.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
    reason_text = flatten_reason(record.get("failure_reason")) + " " + flatten_reason(record.get("planner_guard"))
    reason_text = reason_text.lower()
    return any(keyword in reason_text for keyword in keywords)


def is_flat(record: Dict[str, Any]) -> bool:
    return text_flag(record, ["flat_repeated_pair", "flat_repeat", "flat_repeat_detected"], TEXT_FLAT_KEYWORDS)


def is_reused(record: Dict[str, Any]) -> bool:
    return text_flag(
        record,
        ["reused_history_pairs", "mostly_reused_history_pairs", "copied_history"],
        TEXT_REUSED_KEYWORDS,
    )


def parse_success(record: Dict[str, Any]) -> bool:
    if "parse_success" in record:
        return bool(record.get("parse_success"))
    return all(finite_number(record.get(key)) is not None for key in METRIC_KEYS[:3])


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def action_scores(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
    target = action_train_scale(record, pred=False)
    pred = action_train_scale(record, pred=True)
    scores = {
        "target_abs_curvature_x100": None,
        "target_speed_change": None,
        "speed_delta_mae": None,
        "curvature_delta_mae": None,
    }
    if target:
        curvatures = [row[1] for row in target]
        speeds = [row[0] for row in target]
        scores["target_abs_curvature_x100"] = mean([abs(value) for value in curvatures])
        scores["target_speed_change"] = max(speeds) - min(speeds) if speeds else None
    if pred and target and len(pred) >= 2 and len(target) >= 2:
        n = min(len(pred), len(target))
        pred_speed_delta = [pred[idx][0] - pred[idx - 1][0] for idx in range(1, n)]
        target_speed_delta = [target[idx][0] - target[idx - 1][0] for idx in range(1, n)]
        pred_curv_delta = [pred[idx][1] - pred[idx - 1][1] for idx in range(1, n)]
        target_curv_delta = [target[idx][1] - target[idx - 1][1] for idx in range(1, n)]
        scores["speed_delta_mae"] = mean(
            [abs(pred_speed_delta[idx] - target_speed_delta[idx]) for idx in range(n - 1)]
        )
        scores["curvature_delta_mae"] = mean(
            [abs(pred_curv_delta[idx] - target_curv_delta[idx]) for idx in range(n - 1)]
        )
    return scores


def complete_record(record: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(record)
    pred = action_train_scale(record, pred=True)
    target = action_train_scale(record, pred=False)
    if pred and target and len(pred) and len(target):
        n = min(len(pred), len(target))
        speed_abs = [abs(pred[idx][0] - target[idx][0]) for idx in range(n)]
        curvature_abs = [abs(pred[idx][1] - target[idx][1]) for idx in range(n)]
        record.setdefault("speed_mae_mps", mean(speed_abs))
        record.setdefault("curvature_mae_x100", mean(curvature_abs))
        if speed_abs and curvature_abs:
            record.setdefault("overall_l1_train_scale", (sum(speed_abs) + sum(curvature_abs)) / (2 * n))
    scores = action_scores(record)
    for key, value in scores.items():
        record[key] = value
    record["_parse_success"] = parse_success(record)
    record["_flat_repeated_pair"] = is_flat(record)
    record["_reused_history_pairs"] = is_reused(record)
    return record


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
    success = [record for record in records if record["_parse_success"]]
    high_curvature = top_indices(success, "target_abs_curvature_x100")
    speed_changing = top_indices(success, "target_speed_change")
    low_curvature = bottom_indices(success, "target_abs_curvature_x100")
    low_speed_change = bottom_indices(success, "target_speed_change")
    steady_indices = low_curvature & low_speed_change
    return {
        "all": success,
        "high_curvature_top20": [record for idx, record in enumerate(success) if idx in high_curvature],
        "speed_changing_top20": [record for idx, record in enumerate(success) if idx in speed_changing],
        "steady_low_curvature": [record for idx, record in enumerate(success) if idx in steady_indices],
        "text_non_flat": [record for record in success if not record["_flat_repeated_pair"]],
        "text_non_reused": [record for record in success if not record["_reused_history_pairs"]],
        "text_clean": [
            record
            for record in success
            if not record["_flat_repeated_pair"] and not record["_reused_history_pairs"]
        ],
    }


def analyze_one(name: str, path: str) -> Dict[str, Any]:
    records = [complete_record(record) for record in load_jsonl(path)]
    groups = build_groups(records)
    return {
        "name": name,
        "path": path,
        "total_records": len(records),
        "parse_success": sum(1 for record in records if record["_parse_success"]),
        "parse_failed": sum(1 for record in records if not record["_parse_success"]),
        "flat_repeated_pair": sum(1 for record in records if record["_parse_success"] and record["_flat_repeated_pair"]),
        "reused_history_pairs": sum(1 for record in records if record["_parse_success"] and record["_reused_history_pairs"]),
        "groups": {group_name: summarize(group_records) for group_name, group_records in groups.items()},
    }


def fmt(value: Any) -> str:
    number = finite_number(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def build_text(analysis: Dict[str, Any]) -> str:
    lines = ["Action Eval Scenario Analysis", ""]
    for item in analysis["records"]:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"path: {item['path']}",
                f"parse_success={item['parse_success']} parse_failed={item['parse_failed']}",
                f"flat_repeated_pair={item['flat_repeated_pair']} reused_history_pairs={item['reused_history_pairs']}",
                "",
                "| group | count | speed_mae_mps | curvature_mae_x100 | overall_l1_train_scale | speed_delta_mae | curvature_delta_mae |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group_name, group in item["groups"].items():
            lines.append(
                "| {name} | {count} | {speed} | {curv} | {overall} | {speed_delta} | {curv_delta} |".format(
                    name=group_name,
                    count=group["count"],
                    speed=fmt(group.get("speed_mae_mps")),
                    curv=fmt(group.get("curvature_mae_x100")),
                    overall=fmt(group.get("overall_l1_train_scale")),
                    speed_delta=fmt(group.get("speed_delta_mae")),
                    curv_delta=fmt(group.get("curvature_delta_mae")),
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


def main():
    args = parse_args()
    analyses = []
    for value in args.records:
        name, path = parse_named_path(value)
        analyses.append(analyze_one(name, path))
    report = {"records": analyses}
    write_json(args.output_json, report)
    write_text(args.output_txt, build_text(report))
    print(f"[ActionEvalScenario] wrote json: {args.output_json}")
    print(f"[ActionEvalScenario] wrote txt: {args.output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
