#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from action_head import ContinuousActionHead, EgoOnlyActionHead
from qwen_planner import (
    build_qwen_inputs,
    extract_qwen_hidden_states,
    select_planning_hidden,
)
from train_action_head import load_qwen_model_and_processor


SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]
MODEL_TYPES = ("ego_only", "qwen_hidden")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OpenEMMA OFT-lite action heads.")
    parser.add_argument("--jsonl", required=True, help="Path to stable GT action chunk JSONL.")
    parser.add_argument("--model_type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to action head checkpoint.")
    parser.add_argument("--model-path", type=str, default="qwen", help="Qwen local path or qwen alias.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100, help="0 means no limit after start_index.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default=None)
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
        print("[EvalActionHead] cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_image_path(record: Dict[str, Any], dataroot: str) -> Optional[str]:
    metadata = record.get("metadata", {})
    image_path = metadata.get("image_path")
    if isinstance(image_path, str) and os.path.exists(image_path):
        return image_path
    if isinstance(image_path, str) and not os.path.isabs(image_path):
        candidate = os.path.join(dataroot, image_path)
        if os.path.exists(candidate):
            return candidate
    return None


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


def make_target_tensors(record: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    target_physical = tensor_or_none(record.get("target", {}).get("future_action_gt"), (10, 2))
    if target_physical is None:
        return None
    target_train = target_physical.clone()
    target_train[:, 1] *= 100.0
    return {
        "physical": target_physical.unsqueeze(0),
        "train_scale": target_train.unsqueeze(0),
    }


def get_prompt_fields(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    input_section = record.get("input", {})
    system_message = input_section.get("system_message")
    planning_prompt = input_section.get("planning_prompt")
    if not isinstance(system_message, str) or not system_message.strip():
        return None
    if not isinstance(planning_prompt, str) or not planning_prompt.strip():
        return None
    return {"system_message": system_message, "planning_prompt": planning_prompt}


def collect_samples(records: List[Dict[str, Any]], args):
    samples = []
    skipped = Counter()
    if args.start_index < 0:
        skipped["negative_start_index"] += 1
        start_index = 0
    else:
        start_index = args.start_index

    for record_index, record in enumerate(records[start_index:], start=start_index):
        if args.max_samples > 0 and len(samples) >= args.max_samples:
            break

        targets = make_target_tensors(record)
        if targets is None:
            skipped["invalid_target_future_action_gt"] += 1
            continue

        sample = {
            "record_index": record_index,
            "line_no": record.get("_line_no"),
            "target_train": targets["train_scale"],
            "target_physical": targets["physical"],
        }

        if args.model_type == "ego_only":
            ego_history = tensor_or_none(record.get("input", {}).get("ego_history_array"), (10, 3))
            if ego_history is None:
                skipped["invalid_input_ego_history_array"] += 1
                continue
            sample["ego_history"] = ego_history.unsqueeze(0)
        else:
            prompt_fields = get_prompt_fields(record)
            if prompt_fields is None:
                skipped["missing_input_system_or_planning_prompt"] += 1
                continue
            image_path = resolve_image_path(record, args.dataroot)
            if image_path is None:
                skipped["missing_image_path"] += 1
                continue
            sample.update(prompt_fields)
            sample["image_path"] = image_path

        samples.append(sample)
    return samples, skipped


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def load_ego_head(checkpoint: Dict[str, Any], device: torch.device) -> EgoOnlyActionHead:
    config = checkpoint.get("config", {})
    head = EgoOnlyActionHead(
        history_steps=int(config.get("history_steps", 10)),
        ego_dim=int(config.get("ego_dim", 3)),
        chunk_size=int(config.get("chunk_size", 10)),
        action_dim=int(config.get("action_dim", 2)),
        hidden_size=int(config.get("hidden_size", 256)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint.get("ego_action_head_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("checkpoint missing ego_action_head_state_dict")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def load_qwen_hidden_head(checkpoint: Dict[str, Any], device: torch.device) -> ContinuousActionHead:
    config = checkpoint.get("config", {})
    head = ContinuousActionHead(
        hidden_dim=int(config.get("hidden_dim", 3584)),
        chunk_size=int(config.get("chunk_size", 10)),
        action_dim=int(config.get("action_dim", 2)),
        hidden_size=int(config.get("hidden_size", 1024)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint.get("action_head_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("checkpoint missing action_head_state_dict")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def qwen_eval_message_builder(system_message: str):
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


def pred_to_physical(pred_train: torch.Tensor) -> torch.Tensor:
    pred_physical = pred_train.clone()
    pred_physical[..., 1] /= 100.0
    return pred_physical


def update_metric_sums(sums: Dict[str, float], pred_train: torch.Tensor, target_train: torch.Tensor):
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
    sums["speed_abs"] += float(torch.abs(speed_err).sum())
    sums["curvature_abs_x100"] += float(torch.abs(curvature_err_x100).sum())
    sums["curvature_abs_1pm"] += float(torch.abs(curvature_err_1pm).sum())
    sums["overall_abs_train_scale"] += float(overall_abs.sum())
    sums["speed_sq"] += float(torch.square(speed_err).sum())
    sums["curvature_sq_x100"] += float(torch.square(curvature_err_x100).sum())
    sums["action_values"] += int(overall_abs.numel())


def finalize_metrics(args, loaded_records: int, skipped: Counter, sums: Dict[str, float]) -> Dict[str, Any]:
    num_points = int(sums["points"])
    num_action_values = int(sums["action_values"])
    metrics = {
        "model_type": args.model_type,
        "jsonl": args.jsonl,
        "checkpoint": args.checkpoint,
        "model_path": args.model_path,
        "dataroot": args.dataroot,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "train_action_schema": TRAIN_ACTION_SCHEMA,
        "loaded_records": loaded_records,
        "num_samples": int(sums["samples"]),
        "num_points": num_points,
        "skipped": dict(skipped),
        "speed_mae_mps": None,
        "curvature_mae_1pm": None,
        "curvature_mae_x100": None,
        "overall_l1_train_scale": None,
        "speed_rmse_mps": None,
        "curvature_rmse_x100": None,
    }
    if num_points == 0:
        return metrics

    metrics["speed_mae_mps"] = sums["speed_abs"] / num_points
    metrics["curvature_mae_1pm"] = sums["curvature_abs_1pm"] / num_points
    metrics["curvature_mae_x100"] = sums["curvature_abs_x100"] / num_points
    metrics["overall_l1_train_scale"] = sums["overall_abs_train_scale"] / num_action_values
    metrics["speed_rmse_mps"] = (sums["speed_sq"] / num_points) ** 0.5
    metrics["curvature_rmse_x100"] = (sums["curvature_sq_x100"] / num_points) ** 0.5
    return metrics


def print_summary(metrics: Dict[str, Any]):
    print(f"[EvalActionHead] model_type={metrics['model_type']}")
    print(
        f"samples={metrics['num_samples']} points={metrics['num_points']} "
        f"skipped={metrics['skipped']}"
    )
    print(f"speed_mae_mps={metrics['speed_mae_mps']}")
    print(f"curvature_mae_x100={metrics['curvature_mae_x100']}")
    print(f"curvature_mae_1pm={metrics['curvature_mae_1pm']}")
    print(f"overall_l1_train_scale={metrics['overall_l1_train_scale']}")


def save_report(metrics: Dict[str, Any], output_json: Optional[str]):
    if not output_json:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[EvalActionHead] saved report: {output_json}")


def evaluate_ego_only(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    head = load_ego_head(checkpoint, device)
    sums = Counter()
    with torch.no_grad():
        for sample in samples:
            ego_history = sample["ego_history"].to(device)
            pred_train = head(ego_history)
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
    return sums


def evaluate_qwen_hidden(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    action_head = load_qwen_hidden_head(checkpoint, device)
    qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    sums = Counter()
    with torch.no_grad():
        for sample in samples:
            inputs = build_qwen_inputs(
                prompt=sample["planning_prompt"],
                images=sample["image_path"],
                processor=processor,
                model=qwen_model,
                args=args,
                get_message_fn=qwen_eval_message_builder(sample["system_message"]),
            )
            last_hidden_state = extract_qwen_hidden_states(inputs, qwen_model)
            planning_hidden = select_planning_hidden(last_hidden_state, inputs["attention_mask"])
            planning_hidden = planning_hidden.to(device=device, dtype=next(action_head.parameters()).dtype)
            pred_train = action_head(planning_hidden)
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
    return sums


def main():
    args = parse_args()
    records, skipped = load_jsonl_records(args.jsonl)
    samples, sample_skipped = collect_samples(records, args)
    skipped.update(sample_skipped)
    print(
        f"[EvalActionHead] loaded_records={len(records)} usable_samples={len(samples)} "
        f"skipped={sum(skipped.values())}"
    )
    if skipped:
        print(f"[EvalActionHead] skipped_reasons={dict(skipped)}")
    if not samples:
        metrics = finalize_metrics(args, len(records), skipped, Counter())
        print_summary(metrics)
        save_report(metrics, args.output_json)
        return 1

    device = resolve_device(args.device)
    if args.model_type == "ego_only":
        sums = evaluate_ego_only(args, samples, device)
    else:
        sums = evaluate_qwen_hidden(args, samples, device)

    metrics = finalize_metrics(args, len(records), skipped, sums)
    print_summary(metrics)
    save_report(metrics, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
