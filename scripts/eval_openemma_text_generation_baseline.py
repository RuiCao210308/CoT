#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from base_planner import (
    PlannerInput,
    build_speed_curvature_prompt,
    build_speed_curvature_retry_prompt,
    build_speed_curvature_sys_message,
    evaluate_speed_curvature_prediction,
    parse_speed_curvature_text,
)
from qwen_planner import build_qwen_inputs, generate_with_qwen
from train_action_head import load_qwen_model_and_processor


SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
TEXT_PRED_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]
TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]
PAIR_PATTERN = re.compile(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate original OpenEMMA Qwen text-generation baseline.")
    parser.add_argument("--jsonl", required=True, help="Path to action chunk JSONL.")
    parser.add_argument("--model-path", type=str, default="qwen", help="Qwen local path or qwen alias.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means no limit after start_index.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="openemma_text")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_planner_guard", type=str2bool, default=False)
    parser.add_argument("--retry_on_invalid", type=str2bool, default=False)
    parser.add_argument("--max_retries", type=int, default=0)
    parser.add_argument("--save_raw_text", type=str2bool, default=True)
    return parser.parse_args()


def load_jsonl_records(jsonl_path: str):
    records = []
    skipped = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped["json_decode_error"] += 1
                continue
            record["_line_no"] = line_no
            records.append(record)
    return records, skipped


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if "cuda" in device_arg and not torch.cuda.is_available():
        print("[OpenEMMATextBaseline] cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_image_path(record: Dict[str, Any], dataroot: str) -> Optional[str]:
    metadata = record.get("metadata", {})
    candidates = [
        metadata.get("image_path"),
        metadata.get("camera_path"),
        metadata.get("cam_front_path"),
        metadata.get("CAM_FRONT"),
        record.get("image_path"),
        record.get("camera_path"),
    ]
    for image_path in candidates:
        if not isinstance(image_path, str) or not image_path:
            continue
        if os.path.exists(image_path):
            return image_path
        if not os.path.isabs(image_path):
            candidate = os.path.join(dataroot, image_path)
            if os.path.exists(candidate):
                return candidate
    return None


def tensor_or_none(value: Any, shape) -> Optional[np.ndarray]:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if tuple(array.shape) != tuple(shape):
        return None
    if not np.isfinite(array).all():
        return None
    return array


def make_target_train(record: Dict[str, Any]) -> Optional[np.ndarray]:
    target = tensor_or_none(record.get("target", {}).get("future_action_gt"), (10, 2))
    if target is None:
        return None
    target_train = target.copy()
    target_train[:, 1] *= 100.0
    return target_train


def get_prompt(record: Dict[str, Any]) -> Optional[str]:
    prompt = record.get("input", {}).get("planning_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt

    ego_history = tensor_or_none(record.get("input", {}).get("ego_history_array"), (10, 3))
    if ego_history is None:
        return None
    obs_speeds = ego_history[:, 1]
    obs_curvatures_1pm = ego_history[:, 2] / 100.0
    obs_velocities = np.stack([obs_speeds, np.zeros_like(obs_speeds)], axis=1)
    planner_input = PlannerInput(
        images=None,
        obs_velocities=obs_velocities,
        obs_curvatures=obs_curvatures_1pm,
        scene_description=record.get("metadata", {}).get("scene_description"),
        object_description=None,
        intent_description=None,
        method="openemma",
        reasoning_mode="cot",
    )
    prompt, _ = build_speed_curvature_prompt(planner_input)
    return prompt


def get_system_message(record: Dict[str, Any]) -> str:
    system_message = record.get("input", {}).get("system_message")
    if isinstance(system_message, str) and system_message.strip():
        return system_message
    return build_speed_curvature_sys_message(article="an")


def get_obs_motion(record: Dict[str, Any]):
    ego_history = tensor_or_none(record.get("input", {}).get("ego_history_array"), (10, 3))
    if ego_history is None:
        return None, None
    obs_speeds = ego_history[:, 1]
    obs_curvatures_1pm = ego_history[:, 2] / 100.0
    obs_velocities = np.stack([obs_speeds, np.zeros_like(obs_speeds)], axis=1)
    return obs_velocities, obs_curvatures_1pm


def qwen_text_message_builder(system_message: str):
    def _get_message(prompt, image=None, args=None):
        return [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    return _get_message


def count_raw_pairs(raw_text: Any) -> int:
    if not isinstance(raw_text, str):
        return 0
    cleaned = raw_text.replace("Future speeds and curvatures:", "")
    return len(PAIR_PATTERN.findall(cleaned))


def parse_prediction(raw_text: str):
    raw_pair_count = count_raw_pairs(raw_text)
    pred = parse_speed_curvature_text(raw_text, max_len=10)
    if pred is None:
        return None, "parse_failed", raw_pair_count
    if raw_pair_count < 10:
        return None, "too_few_pairs", raw_pair_count
    if raw_pair_count > 10:
        return None, "too_many_pairs", raw_pair_count
    if tuple(pred.shape) != (10, 2):
        return None, "invalid_shape", raw_pair_count
    if not np.isfinite(pred).all():
        return None, "nonfinite_prediction", raw_pair_count
    return pred.astype(np.float32), None, raw_pair_count


def generate_text(prompt: str, image_path: str, system_message: str, processor: Any, model: Any, args) -> str:
    inputs = build_qwen_inputs(
        prompt=prompt,
        images=image_path,
        processor=processor,
        model=model,
        args=args,
        get_message_fn=qwen_text_message_builder(system_message),
    )
    return generate_with_qwen(
        inputs=inputs,
        processor=processor,
        model=model,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0.0,
        temperature=max(args.temperature, 1e-5),
    )


def evaluate_prediction(pred_train: np.ndarray, target_train: np.ndarray) -> Dict[str, float]:
    speed_abs = np.abs(pred_train[:, 0] - target_train[:, 0])
    curvature_abs_x100 = np.abs(pred_train[:, 1] - target_train[:, 1])
    overall_abs = np.abs(pred_train - target_train)
    return {
        "speed_mae_mps": float(speed_abs.mean()),
        "curvature_mae_x100": float(curvature_abs_x100.mean()),
        "curvature_mae_1pm": float((curvature_abs_x100 / 100.0).mean()),
        "overall_l1_train_scale": float(overall_abs.mean()),
    }


def sample_identity(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata", {})
    return {
        "scene_name": metadata.get("scene_name"),
        "frame_idx": metadata.get("frame_idx"),
        "sample_token": metadata.get("sample_token"),
        "line_no": record.get("_line_no"),
    }


def collect_samples(records: List[Dict[str, Any]], args):
    samples = []
    skipped = Counter()
    start_index = max(args.start_index, 0)
    if args.start_index < 0:
        skipped["negative_start_index"] += 1
    for record_index, record in enumerate(records[start_index:], start=start_index):
        if args.max_samples > 0 and len(samples) >= args.max_samples:
            break
        target_train = make_target_train(record)
        if target_train is None:
            skipped["invalid_target_future_action_gt"] += 1
            continue
        image_path = resolve_image_path(record, args.dataroot)
        if image_path is None:
            skipped["missing_camera_file"] += 1
            continue
        prompt = get_prompt(record)
        if prompt is None:
            skipped["missing_prompt_or_ego_history"] += 1
            continue
        obs_velocities, obs_curvatures = get_obs_motion(record)
        if obs_velocities is None or obs_curvatures is None:
            skipped["invalid_input_ego_history_array"] += 1
            continue
        samples.append(
            {
                "record_index": record_index,
                "record": record,
                "image_path": image_path,
                "prompt": prompt,
                "system_message": get_system_message(record),
                "target_train": target_train,
                "target_action": record.get("target", {}).get("future_action_gt"),
                "obs_velocities": obs_velocities,
                "obs_curvatures": obs_curvatures,
            }
        )
    return samples, skipped


def write_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl_row(handle, row: Dict[str, Any]):
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(args):
    records, skipped = load_jsonl_records(args.jsonl)
    samples, sample_skipped = collect_samples(records, args)
    skipped.update(sample_skipped)
    print(
        f"[OpenEMMATextBaseline] loaded_records={len(records)} usable_samples={len(samples)} "
        f"skipped={sum(skipped.values())}"
    )
    if skipped:
        print(f"[OpenEMMATextBaseline] skipped_reasons={dict(skipped)}")
    if not samples:
        metrics = finalize_metrics(args, len(records), skipped, Counter(), Counter(), 0, [])
        write_json(args.output_json, metrics)
        return 1

    device = resolve_device(args.device)
    model, processor = load_qwen_model_and_processor(args.model_path, str(device))

    sums = Counter()
    failure_reasons = Counter()
    retry_used_count = 0
    record_rows = []
    output_handle = None
    if args.output_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
        output_handle = open(args.output_jsonl, "w", encoding="utf-8")

    try:
        for sample in samples:
            record = sample["record"]
            raw_text = generate_text(
                sample["prompt"],
                sample["image_path"],
                sample["system_message"],
                processor,
                model,
                args,
            )
            pred_train, failure_reason, raw_pair_count = parse_prediction(raw_text)
            retry_used = False
            guard_details = None

            if pred_train is not None:
                guard_details = evaluate_speed_curvature_prediction(
                    pred_train,
                    sample["obs_velocities"],
                    sample["obs_curvatures"],
                    return_details=True,
                )
                if guard_details.get("flat_repeat_detected"):
                    failure_reasons["flat_repeated_pair"] += 1
                if "mostly_reused_history_pairs" in guard_details.get("reasons", []):
                    failure_reasons["reused_history_pairs"] += 1
                if "copied_history_prefix" in guard_details.get("reasons", []):
                    failure_reasons["reused_history_pairs"] += 1
                if "copied_history_suffix" in guard_details.get("reasons", []):
                    failure_reasons["reused_history_pairs"] += 1

            if (
                pred_train is not None
                and args.use_planner_guard
                and guard_details is not None
                and not guard_details.get("is_valid", False)
            ):
                failure_reason = ",".join(guard_details.get("reasons", [])) or "planner_guard_rejected"
                failure_reasons["planner_guard_triggered"] += 1
                pred_train = None

            retries_remaining = max(args.max_retries, 0) if args.retry_on_invalid else 0
            while pred_train is None and retries_remaining > 0:
                retry_used = True
                retry_prompt = build_speed_curvature_retry_prompt(sample["prompt"], raw_text, [failure_reason])
                raw_text = generate_text(
                    retry_prompt,
                    sample["image_path"],
                    sample["system_message"],
                    processor,
                    model,
                    args,
                )
                pred_train, failure_reason, raw_pair_count = parse_prediction(raw_text)
                if pred_train is not None and args.use_planner_guard:
                    guard_details = evaluate_speed_curvature_prediction(
                        pred_train,
                        sample["obs_velocities"],
                        sample["obs_curvatures"],
                        return_details=True,
                    )
                    if not guard_details.get("is_valid", False):
                        failure_reason = ",".join(guard_details.get("reasons", [])) or "planner_guard_rejected"
                        failure_reasons["planner_guard_triggered"] += 1
                        pred_train = None
                retries_remaining -= 1

            row = {
                **sample_identity(record),
                "record_index": sample["record_index"],
                "parse_success": pred_train is not None,
                "failure_reason": failure_reason,
                "raw_pair_count": raw_pair_count,
                "retry_used": retry_used,
                "planner_guard": guard_details,
            }
            if args.save_raw_text:
                row["raw_qwen_text"] = raw_text
            if retry_used:
                retry_used_count += 1
                failure_reasons["retry_used"] += 1

            if pred_train is None:
                failure_reasons[failure_reason or "parse_failed"] += 1
                row.update(
                    {
                        "parsed_action": None,
                        "target_action": sample["target_action"],
                        "speed_mae_mps": None,
                        "curvature_mae_x100": None,
                        "overall_l1_train_scale": None,
                    }
                )
            else:
                target_train = sample["target_train"]
                sample_metrics = evaluate_prediction(pred_train, target_train)
                sums["samples"] += 1
                sums["points"] += 10
                sums["speed_abs"] += sample_metrics["speed_mae_mps"] * 10
                sums["curvature_abs_x100"] += sample_metrics["curvature_mae_x100"] * 10
                sums["curvature_abs_1pm"] += sample_metrics["curvature_mae_1pm"] * 10
                sums["overall_abs_train_scale"] += sample_metrics["overall_l1_train_scale"] * 20
                sums["action_values"] += 20
                row.update(
                    {
                        "parsed_action": pred_train.tolist(),
                        "target_action": sample["target_action"],
                        **sample_metrics,
                    }
                )

            record_rows.append(row)
            if output_handle is not None:
                write_jsonl_row(output_handle, row)
    finally:
        if output_handle is not None:
            output_handle.close()

    metrics = finalize_metrics(args, len(records), skipped, failure_reasons, sums, retry_used_count, record_rows)
    write_json(args.output_json, metrics)
    print(f"[OpenEMMATextBaseline] saved report: {args.output_json}")
    if args.output_jsonl:
        print(f"[OpenEMMATextBaseline] saved records: {args.output_jsonl}")
    print_summary(metrics)
    return 0


def finalize_metrics(
    args,
    loaded_records: int,
    skipped: Counter,
    failure_reasons: Counter,
    sums: Counter,
    retry_used_count: int,
    record_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    requested = len(record_rows)
    evaluated = int(sums["samples"])
    parse_failed = requested - evaluated
    points = int(sums["points"])
    action_values = int(sums["action_values"])
    metrics = {
        "model_type": "openemma_text_generation",
        "method": args.method,
        "jsonl": args.jsonl,
        "model_path": args.model_path,
        "dataroot": args.dataroot,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "use_planner_guard": args.use_planner_guard,
        "retry_on_invalid": args.retry_on_invalid,
        "max_retries": args.max_retries,
        "save_raw_text": args.save_raw_text,
        "loaded_records": loaded_records,
        "num_samples_requested": requested,
        "num_samples_evaluated": evaluated,
        "parse_success": evaluated,
        "parse_failed": parse_failed,
        "parse_failed_rate": parse_failed / requested if requested else None,
        "speed_mae_mps": None,
        "curvature_mae_x100": None,
        "curvature_mae_1pm": None,
        "overall_l1_train_scale": None,
        "raw_curvature_unit": "curvature_x100",
        "target_curvature_unit": "curvature_1pm",
        "eval_curvature_unit": "curvature_x100",
        "prediction_schema": TEXT_PRED_ACTION_SCHEMA,
        "target_schema": SOURCE_ACTION_SCHEMA,
        "train_scale_schema": TRAIN_ACTION_SCHEMA,
        "skipped": dict(skipped),
        "failure_reasons": dict(failure_reasons),
        "invalid_shape": int(failure_reasons["invalid_shape"]),
        "nonfinite_prediction": int(failure_reasons["nonfinite_prediction"]),
        "too_few_pairs": int(failure_reasons["too_few_pairs"]),
        "too_many_pairs": int(failure_reasons["too_many_pairs"]),
        "flat_repeated_pair": int(failure_reasons["flat_repeated_pair"]),
        "reused_history_pairs": int(failure_reasons["reused_history_pairs"]),
        "retry_used": retry_used_count,
        "planner_guard_triggered": int(failure_reasons["planner_guard_triggered"]),
    }
    if points > 0 and action_values > 0:
        metrics["speed_mae_mps"] = sums["speed_abs"] / points
        metrics["curvature_mae_x100"] = sums["curvature_abs_x100"] / points
        metrics["curvature_mae_1pm"] = sums["curvature_abs_1pm"] / points
        metrics["overall_l1_train_scale"] = sums["overall_abs_train_scale"] / action_values
    return metrics


def print_summary(metrics: Dict[str, Any]):
    print("[OpenEMMATextBaseline] model_type=openemma_text_generation")
    print(
        f"requested={metrics['num_samples_requested']} evaluated={metrics['num_samples_evaluated']} "
        f"parse_failed={metrics['parse_failed']} parse_failed_rate={metrics['parse_failed_rate']}"
    )
    print(f"speed_mae_mps={metrics['speed_mae_mps']}")
    print(f"curvature_mae_x100={metrics['curvature_mae_x100']}")
    print(f"curvature_mae_1pm={metrics['curvature_mae_1pm']}")
    print(f"overall_l1_train_scale={metrics['overall_l1_train_scale']}")
    print(f"failure_reasons={metrics['failure_reasons']}")


def main():
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
