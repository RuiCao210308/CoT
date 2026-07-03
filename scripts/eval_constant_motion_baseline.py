#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Optional

import torch


SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate constant speed-curvature motion baselines.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--mode", choices=("last", "mean"), default="last")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means no limit after start_index.")
    parser.add_argument("--split", type=str, default=None)
    return parser.parse_args()


def tensor_or_none(value: Any, shape) -> Optional[torch.Tensor]:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tuple(tensor.shape) != tuple(shape):
        return None
    if not torch.isfinite(tensor).all():
        return None
    return tensor


def pred_to_physical(pred_train: torch.Tensor) -> torch.Tensor:
    pred_physical = pred_train.clone()
    pred_physical[..., 1] /= 100.0
    return pred_physical


def tensor_to_list(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    tensor = tensor.detach().float().cpu()
    if tensor.ndim >= 1 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor.tolist()


def make_target(record: Dict[str, Any]) -> Optional[torch.Tensor]:
    target = tensor_or_none(record.get("target", {}).get("future_action_gt"), (10, 2))
    if target is None:
        return None
    target = target.clone()
    target[:, 1] *= 100.0
    return target.unsqueeze(0)


def make_prediction(record: Dict[str, Any], mode: str) -> Optional[torch.Tensor]:
    ego = tensor_or_none(record.get("input", {}).get("ego_history_array"), (10, 3))
    if ego is None:
        return None
    if mode == "last":
        pair = ego[-1, 1:3]
    else:
        pair = ego[:, 1:3].mean(dim=0)
    return pair.view(1, 1, 2).repeat(1, 10, 1)


def make_waypoints(record: Dict[str, Any]) -> Optional[torch.Tensor]:
    target = record.get("target", {})
    if target.get("future_waypoints_schema") != ["x_m_local", "y_m_local"]:
        return None
    waypoints = tensor_or_none(target.get("future_waypoints_local"), (10, 2))
    if waypoints is None:
        return None
    return waypoints.unsqueeze(0)


def update_sums(sums: Counter, pred_train: torch.Tensor, target_train: torch.Tensor) -> None:
    pred_train = pred_train.detach().float().cpu()
    target_train = target_train.detach().float().cpu()
    pred_physical = pred_to_physical(pred_train)
    target_physical = pred_to_physical(target_train)
    speed_err = pred_physical[..., 0] - target_physical[..., 0]
    curvature_err_x100 = pred_train[..., 1] - target_train[..., 1]
    curvature_err_1pm = pred_physical[..., 1] - target_physical[..., 1]
    overall_abs = torch.abs(pred_train - target_train)
    sums["samples"] += int(pred_train.shape[0])
    sums["points"] += int(pred_train.shape[0] * pred_train.shape[1])
    sums["action_values"] += int(overall_abs.numel())
    sums["speed_abs"] += float(torch.abs(speed_err).sum())
    sums["curvature_abs_x100"] += float(torch.abs(curvature_err_x100).sum())
    sums["curvature_abs_1pm"] += float(torch.abs(curvature_err_1pm).sum())
    sums["overall_abs_train_scale"] += float(overall_abs.sum())


def write_record(handle, args, record: Dict[str, Any], record_index: int, pred_train: torch.Tensor, target_train: torch.Tensor):
    pred_train_cpu = pred_train.detach().float().cpu()
    target_train_cpu = target_train.detach().float().cpu()
    pred_physical = pred_to_physical(pred_train_cpu)
    target_physical = pred_to_physical(target_train_cpu)
    speed_abs = torch.abs(pred_physical[..., 0] - target_physical[..., 0])
    curvature_abs_x100 = torch.abs(pred_train_cpu[..., 1] - target_train_cpu[..., 1])
    overall_abs = torch.abs(pred_train_cpu - target_train_cpu)
    metadata = record.get("metadata", {})
    target_waypoints = make_waypoints(record)
    row = {
        "scene_name": metadata.get("scene_name"),
        "frame_idx": metadata.get("frame_idx"),
        "sample_token": metadata.get("sample_token"),
        "record_index": record_index,
        "line_no": record.get("_line_no"),
        "model_type": "constant_motion",
        "ablation_mode": f"constant_motion_{args.mode}",
        "parse_success": True,
        "pred_action": tensor_to_list(pred_train_cpu),
        "pred_action_schema": TRAIN_ACTION_SCHEMA,
        "target_action": tensor_to_list(target_physical),
        "target_action_schema": SOURCE_ACTION_SCHEMA,
        "target_action_train_scale": tensor_to_list(target_train_cpu),
        "target_action_train_scale_schema": TRAIN_ACTION_SCHEMA,
        "target_waypoints_local": tensor_to_list(target_waypoints),
        "target_waypoints_schema": ["x_m_local", "y_m_local"] if target_waypoints is not None else None,
        "speed_mae_mps": float(speed_abs.mean()),
        "curvature_mae_x100": float(curvature_abs_x100.mean()),
        "overall_l1_train_scale": float(overall_abs.mean()),
    }
    handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    loaded = 0
    skipped = Counter()
    sums = Counter()
    with open(args.jsonl, "r", encoding="utf-8") as f, open(args.output_jsonl, "w", encoding="utf-8") as out:
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
            record_index = loaded
            loaded += 1
            if record_index < args.start_index:
                continue
            if args.max_samples > 0 and sums["samples"] >= args.max_samples:
                break
            target_train = make_target(record)
            pred_train = make_prediction(record, args.mode)
            if target_train is None:
                skipped["invalid_target_future_action_gt"] += 1
                continue
            if pred_train is None:
                skipped["invalid_input_ego_history_array"] += 1
                continue
            update_sums(sums, pred_train, target_train)
            write_record(out, args, record, record_index, pred_train, target_train)

    points = int(sums["points"])
    action_values = int(sums["action_values"])
    metrics = {
        "model_type": "constant_motion",
        "ablation_mode": f"constant_motion_{args.mode}",
        "split": args.split,
        "jsonl": args.jsonl,
        "checkpoint": None,
        "loaded_records": loaded,
        "num_samples": int(sums["samples"]),
        "num_points": points,
        "skipped": dict(skipped),
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "train_action_schema": TRAIN_ACTION_SCHEMA,
        "speed_mae_mps": sums["speed_abs"] / points if points else None,
        "curvature_mae_1pm": sums["curvature_abs_1pm"] / points if points else None,
        "curvature_mae_x100": sums["curvature_abs_x100"] / points if points else None,
        "overall_l1_train_scale": sums["overall_abs_train_scale"] / action_values if action_values else None,
    }
    write_json(args.output_json, metrics)
    print(f"[ConstantMotion] mode={args.mode} samples={metrics['num_samples']} skipped={metrics['skipped']}")
    print(f"[ConstantMotion] saved report: {args.output_json}")
    print(f"[ConstantMotion] saved records: {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
