import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_SPEED_RANGE = (-0.5, 45.0)
DEFAULT_CURVATURE_WARN_ABS_1PM = 0.2
EXTREME_CURVATURE_ABS_1PM = 0.5


def _empty_stats() -> Dict[str, Any]:
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


def numeric_stats(values: Iterable[Any]) -> Dict[str, Any]:
    """Return compact distribution stats for finite numeric values."""
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)

    if not finite:
        return _empty_stats()

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


def _shape_of(value: Any) -> Tuple[int, ...]:
    try:
        return tuple(np.asarray(value, dtype=object).shape)
    except Exception:
        return ()


def _array_or_none(value: Any, dtype=np.float64) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2:
        return None
    return arr


def _count_nonfinite(arr: Optional[np.ndarray]) -> int:
    if arr is None:
        return 0
    return int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))


def _range_violations(arr: Optional[np.ndarray], low: float, high: float) -> int:
    if arr is None:
        return 0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0
    return int(np.count_nonzero((finite < low) | (finite > high)))


def _abs_gt_violations(arr: Optional[np.ndarray], threshold: float) -> int:
    if arr is None:
        return 0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(finite) > threshold))


def _shape_key(shape: Tuple[int, ...]) -> str:
    if not shape:
        return "invalid"
    return "x".join(str(dim) for dim in shape)


def _read_jsonl(path: str) -> Iterable[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped), None
            except json.JSONDecodeError as exc:
                yield line_no, None, str(exc)


def validate_action_chunk_record_basic(record: Dict[str, Any]) -> bool:
    """Validate required action-chunk fields without importing model-side modules."""
    required_top = ["metadata", "input", "target", "prediction", "metrics"]
    missing = [key for key in required_top if key not in record]
    if missing:
        raise ValueError(f"missing top-level fields: {missing}")

    input_section = record["input"]
    target_section = record["target"]
    prediction_section = record["prediction"]

    ego_history = input_section.get("ego_history_array")
    future_gt = target_section.get("future_action_gt")
    pred = prediction_section.get("qwen_predicted_action")

    if len(input_section.get("ego_history_schema", [])) != 3:
        raise ValueError("ego_history_schema must have 3 fields")
    if len(target_section.get("future_action_schema", [])) != 2:
        raise ValueError("future_action_schema must have 2 fields")
    if not ego_history or any(len(row) != 3 for row in ego_history):
        raise ValueError("ego_history_array must have shape [T_hist, 3]")
    if not future_gt or any(len(row) != 2 for row in future_gt):
        raise ValueError("future_action_gt must have shape [T_fut, 2]")
    if pred is not None and (not pred or any(len(row) != 2 for row in pred)):
        raise ValueError("qwen_predicted_action must have shape [T_pred, 2] when present")

    return True


def check_action_chunk_jsonl(
    jsonl_path: str,
    expected_future_len: int = 10,
    speed_range: Tuple[float, float] = DEFAULT_SPEED_RANGE,
    curvature_warn_abs_1pm: float = DEFAULT_CURVATURE_WARN_ABS_1PM,
    curvature_extreme_abs_1pm: float = EXTREME_CURVATURE_ABS_1PM,
    max_examples: int = 20,
) -> Dict[str, Any]:
    """Read action chunk JSONL and summarize training-readiness issues."""
    summary: Dict[str, Any] = {
        "path": jsonl_path,
        "records": 0,
        "json_errors": 0,
        "validation_errors": 0,
        "curvature_units": {
            "target.future_action_gt": "curvature_1pm",
            "prediction.qwen_predicted_action": "curvature_1pm",
            "prediction.raw_qwen_text": "curvature_x100",
        },
        "curvature_thresholds_1pm": {
            "warning_abs_gt": curvature_warn_abs_1pm,
            "extreme_abs_gt": curvature_extreme_abs_1pm,
        },
        "shape_counts": {
            "ego_history_array": Counter(),
            "future_action_gt": Counter(),
            "qwen_predicted_action": Counter(),
        },
        "nonfinite": {
            "ego_history_array": 0,
            "future_action_gt": 0,
            "qwen_predicted_action": 0,
        },
        "prompt": {
            "present": 0,
            "missing": 0,
            "prompt_type_counts": Counter(),
        },
        "range_violations": {
            "target_speed": 0,
            "target_curvature_abs_gt_warn": 0,
            "prediction_speed": 0,
            "prediction_curvature_abs_gt_warn": 0,
            "extreme_target_curvature_abs_gt": 0,
            "extreme_prediction_curvature_abs_gt": 0,
        },
        "planner_guard": {
            "present": 0,
            "invalid": 0,
            "retry_used": 0,
            "retry_count": [],
            "final_invalid": 0,
            "retry_exhausted_invalid": 0,
            "reason_counts": Counter(),
            "repeated_pair_ratio": [],
            "history_overlap_ratio": [],
            "flat_repeat_detected": 0,
        },
        "metrics": defaultdict(list),
        "examples": [],
    }

    for line_no, record, json_error in _read_jsonl(jsonl_path):
        if json_error:
            summary["json_errors"] += 1
            if len(summary["examples"]) < max_examples:
                summary["examples"].append(
                    {"line": line_no, "kind": "json_error", "message": json_error}
                )
            continue

        assert record is not None
        summary["records"] += 1
        metadata = record.get("metadata", {})
        record_id = {
            "line": line_no,
            "scene_name": metadata.get("scene_name"),
            "frame_idx": metadata.get("frame_idx"),
            "sample_token": metadata.get("sample_token"),
        }

        try:
            validate_action_chunk_record_basic(record)
        except Exception as exc:
            summary["validation_errors"] += 1
            if len(summary["examples"]) < max_examples:
                summary["examples"].append(
                    {**record_id, "kind": "validation_error", "message": str(exc)}
                )

        ego = record.get("input", {}).get("ego_history_array")
        input_section = record.get("input", {})
        target = record.get("target", {}).get("future_action_gt")
        pred = record.get("prediction", {}).get("qwen_predicted_action")

        planning_prompt = input_section.get("planning_prompt")
        system_message = input_section.get("system_message")
        if isinstance(planning_prompt, str) and planning_prompt.strip() and isinstance(system_message, str) and system_message.strip():
            summary["prompt"]["present"] += 1
        else:
            summary["prompt"]["missing"] += 1
        prompt_type = input_section.get("prompt_type") or "missing"
        summary["prompt"]["prompt_type_counts"][str(prompt_type)] += 1

        summary["shape_counts"]["ego_history_array"][_shape_key(_shape_of(ego))] += 1
        summary["shape_counts"]["future_action_gt"][_shape_key(_shape_of(target))] += 1
        summary["shape_counts"]["qwen_predicted_action"][_shape_key(_shape_of(pred))] += 1

        ego_arr = _array_or_none(ego)
        target_arr = _array_or_none(target)
        pred_arr = _array_or_none(pred)

        summary["nonfinite"]["ego_history_array"] += _count_nonfinite(ego_arr)
        summary["nonfinite"]["future_action_gt"] += _count_nonfinite(target_arr)
        summary["nonfinite"]["qwen_predicted_action"] += _count_nonfinite(pred_arr)

        if target_arr is not None:
            if target_arr.shape != (expected_future_len, 2) and len(summary["examples"]) < max_examples:
                summary["examples"].append(
                    {
                        **record_id,
                        "kind": "unexpected_target_shape",
                        "shape": list(target_arr.shape),
                    }
                )
            summary["range_violations"]["target_speed"] += _range_violations(
                target_arr[:, 0], speed_range[0], speed_range[1]
            )
            summary["range_violations"]["target_curvature_abs_gt_warn"] += _abs_gt_violations(
                target_arr[:, 1], curvature_warn_abs_1pm
            )
            summary["range_violations"]["extreme_target_curvature_abs_gt"] += _abs_gt_violations(
                target_arr[:, 1], curvature_extreme_abs_1pm
            )

        if pred_arr is not None:
            if pred_arr.shape != (expected_future_len, 2) and len(summary["examples"]) < max_examples:
                summary["examples"].append(
                    {
                        **record_id,
                        "kind": "unexpected_prediction_shape",
                        "shape": list(pred_arr.shape),
                    }
                )
            summary["range_violations"]["prediction_speed"] += _range_violations(
                pred_arr[:, 0], speed_range[0], speed_range[1]
            )
            summary["range_violations"]["prediction_curvature_abs_gt_warn"] += _abs_gt_violations(
                pred_arr[:, 1], curvature_warn_abs_1pm
            )
            summary["range_violations"]["extreme_prediction_curvature_abs_gt"] += _abs_gt_violations(
                pred_arr[:, 1], curvature_extreme_abs_1pm
            )

        prediction = record.get("prediction", {})
        guard = prediction.get("planner_guard") or {}
        if guard:
            summary["planner_guard"]["present"] += 1
            if guard.get("is_valid") is False:
                summary["planner_guard"]["invalid"] += 1
            final_prediction_valid = guard.get("final_prediction_valid", guard.get("is_valid"))
            if final_prediction_valid is False:
                summary["planner_guard"]["final_invalid"] += 1
            for reason in guard.get("reasons") or []:
                summary["planner_guard"]["reason_counts"][str(reason)] += 1
            if guard.get("flat_repeat_detected"):
                summary["planner_guard"]["flat_repeat_detected"] += 1
            retry_count = guard.get("retry_count")
            if retry_count is None:
                retry_count = 1 if (guard.get("retry_used") or prediction.get("retry_used")) else 0
            try:
                retry_count = int(retry_count)
            except (TypeError, ValueError):
                retry_count = 0
            summary["planner_guard"]["retry_count"].append(retry_count)
            if retry_count > 0 and final_prediction_valid is False:
                summary["planner_guard"]["retry_exhausted_invalid"] += 1
            for key in ("repeated_pair_ratio", "history_overlap_ratio"):
                if guard.get(key) is not None:
                    summary["planner_guard"][key].append(guard.get(key))
        if prediction.get("retry_used") or guard.get("retry_used"):
            summary["planner_guard"]["retry_used"] += 1

        for key, value in (record.get("metrics") or {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                summary["metrics"][key].append(float(value))

    return finalize_action_chunk_summary(summary)


def finalize_action_chunk_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    records = summary["records"]
    for key, counter in list(summary["shape_counts"].items()):
        summary["shape_counts"][key] = dict(counter)
    summary["prompt"]["prompt_type_counts"] = dict(summary["prompt"]["prompt_type_counts"])

    guard = summary["planner_guard"]
    guard["trigger_rate"] = (guard["invalid"] / records) if records else None
    guard["retry_rate"] = (guard["retry_used"] / records) if records else None
    guard["final_invalid_rate"] = (guard["final_invalid"] / records) if records else None
    guard["retry_exhausted_invalid_rate"] = (
        guard["retry_exhausted_invalid"] / records
    ) if records else None
    guard["reason_counts"] = dict(guard["reason_counts"])
    guard["retry_count"] = numeric_stats(guard["retry_count"])
    guard["repeated_pair_ratio"] = numeric_stats(guard["repeated_pair_ratio"])
    guard["history_overlap_ratio"] = numeric_stats(guard["history_overlap_ratio"])

    metric_stats = {}
    for key, values in summary["metrics"].items():
        metric_stats[key] = numeric_stats(values)
    summary["metrics"] = metric_stats

    range_issues = sum(summary["range_violations"].values())
    nonfinite_issues = sum(summary["nonfinite"].values())
    summary["status"] = "pass"
    if summary["json_errors"] or summary["validation_errors"] or nonfinite_issues:
        summary["status"] = "fail"
    elif range_issues:
        summary["status"] = "warn"
    return summary


def format_action_chunk_summary(summary: Dict[str, Any]) -> str:
    guard = summary["planner_guard"]
    lines = [
        "Action chunk sanity report",
        f"  path: {summary['path']}",
        f"  status: {summary['status']}",
        f"  records: {summary['records']}",
        f"  json_errors: {summary['json_errors']}",
        f"  validation_errors: {summary['validation_errors']}",
        f"  curvature_units: {summary['curvature_units']}",
        f"  curvature_thresholds_1pm: {summary['curvature_thresholds_1pm']}",
        f"  shapes.ego_history_array: {summary['shape_counts']['ego_history_array']}",
        f"  shapes.future_action_gt: {summary['shape_counts']['future_action_gt']}",
        f"  shapes.qwen_predicted_action: {summary['shape_counts']['qwen_predicted_action']}",
        f"  prompt: present={summary['prompt']['present']}, missing={summary['prompt']['missing']}, "
        f"prompt_type_counts={summary['prompt']['prompt_type_counts']}",
        f"  nonfinite: {summary['nonfinite']}",
        f"  range_violations: {summary['range_violations']}",
        (
            "  planner_guard: "
            f"present={guard['present']}, invalid={guard['invalid']}, "
            f"trigger_rate={_format_rate(guard['trigger_rate'])}, retry_rate={_format_rate(guard['retry_rate'])}, "
            f"final_invalid={guard['final_invalid']}, final_invalid_rate={_format_rate(guard['final_invalid_rate'])}, "
            f"retry_exhausted_invalid={guard['retry_exhausted_invalid']}, "
            f"retry_exhausted_invalid_rate={_format_rate(guard['retry_exhausted_invalid_rate'])}, "
            f"flat_repeat_detected={guard['flat_repeat_detected']}, reasons={guard['reason_counts']}"
        ),
        (
            "  planner_guard.retry_count: "
            f"count={guard['retry_count']['count']}, mean={_fmt(guard['retry_count']['mean'])}, "
            f"p50={_fmt(guard['retry_count']['p50'])}, p90={_fmt(guard['retry_count']['p90'])}, "
            f"max={_fmt(guard['retry_count']['max'])}"
        ),
    ]
    for metric in ("ade", "ade_1s", "ade_2s", "ade_3s"):
        if metric in summary["metrics"]:
            stats = summary["metrics"][metric]
            lines.append(
                f"  {metric}: count={stats['count']}, mean={_fmt(stats['mean'])}, "
                f"p50={_fmt(stats['p50'])}, p90={_fmt(stats['p90'])}, p95={_fmt(stats['p95'])}, max={_fmt(stats['max'])}"
            )
    if summary["examples"]:
        lines.append("  examples:")
        for example in summary["examples"]:
            lines.append(f"    - {example}")
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def save_action_chunk_report(summary: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanity-check OpenEMMA action chunk JSONL files.")
    parser.add_argument("jsonl_path", help="Path to action_chunks.jsonl generated by --save_action_chunks.")
    parser.add_argument("--report_output", type=str, default=None, help="Optional JSON report output path.")
    parser.add_argument("--expected_future_len", type=int, default=10, help="Expected future chunk length.")
    parser.add_argument("--min_speed_mps", type=float, default=DEFAULT_SPEED_RANGE[0])
    parser.add_argument("--max_speed_mps", type=float, default=DEFAULT_SPEED_RANGE[1])
    parser.add_argument(
        "--max_abs_curvature_1pm",
        type=float,
        default=DEFAULT_CURVATURE_WARN_ABS_1PM,
        help="Warning threshold for abs(curvature_1pm). Defaults to 0.2.",
    )
    parser.add_argument(
        "--extreme_abs_curvature_1pm",
        type=float,
        default=EXTREME_CURVATURE_ABS_1PM,
        help="Extreme threshold for abs(curvature_1pm). Defaults to 0.5.",
    )
    parser.add_argument("--max_examples", type=int, default=20)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    max_abs_curv = abs(float(args.max_abs_curvature_1pm))
    summary = check_action_chunk_jsonl(
        args.jsonl_path,
        expected_future_len=args.expected_future_len,
        speed_range=(args.min_speed_mps, args.max_speed_mps),
        curvature_warn_abs_1pm=max_abs_curv,
        curvature_extreme_abs_1pm=abs(float(args.extreme_abs_curvature_1pm)),
        max_examples=args.max_examples,
    )
    print(format_action_chunk_summary(summary))
    if args.report_output:
        save_action_chunk_report(summary, args.report_output)
        print(f"Saved JSON report: {args.report_output}")
    return 1 if summary["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
