#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from action_head import PredictedGeometryFusionHead, action_chunk_l1_loss
from geometry_token import ORACLE_GEOMETRY_DESCRIPTOR_DIM, build_oracle_geometry_descriptor_from_waypoints
from qwen_planner import (
    build_qwen_inputs,
    extract_qwen_hidden_states,
    select_planning_hidden,
)
from train_action_head import load_qwen_model_and_processor


SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]
WAYPOINT_SCHEMA = ["x_m_local", "y_m_local"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train predicted-geometry fusion action head.")
    parser.add_argument("--jsonl", type=str, default="/root/autodl-tmp/action_chunks_gt_waypoint.jsonl")
    parser.add_argument("--model-path", type=str, default="qwen", help="Qwen local path or qwen alias.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/openemma_predicted_geometry_fusion_100")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=100, help="0 means no limit.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--speed_weight", type=float, default=1.0)
    parser.add_argument("--curvature_weight", type=float, default=1.0)
    parser.add_argument("--geometry_weight", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--geometry_dim", type=int, default=ORACLE_GEOMETRY_DESCRIPTOR_DIM)
    parser.add_argument("--dropout", type=float, default=0.1)
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
        print("[PredictedGeometryFusionTrain] cuda requested but unavailable; falling back to cpu")
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


def make_train_target(record: Dict[str, Any]) -> Optional[torch.Tensor]:
    target = tensor_or_none(record.get("target", {}).get("future_action_gt"), (10, 2))
    if target is None:
        return None
    target = target.clone()
    target[:, 1] *= 100.0
    return target.unsqueeze(0)


def make_waypoint_target(record: Dict[str, Any]) -> Optional[torch.Tensor]:
    target = record.get("target", {})
    if target.get("future_waypoints_schema") != WAYPOINT_SCHEMA:
        return None
    waypoints = tensor_or_none(target.get("future_waypoints_local"), (10, 2))
    if waypoints is None:
        return None
    return waypoints.unsqueeze(0)


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
    for record in records:
        if args.max_samples > 0 and len(samples) >= args.max_samples:
            break

        target = make_train_target(record)
        if target is None:
            skipped["invalid_target_future_action_gt"] += 1
            continue
        target_waypoints = make_waypoint_target(record)
        if target_waypoints is None:
            skipped["invalid_target_future_waypoints_local"] += 1
            continue
        ego_history = tensor_or_none(record.get("input", {}).get("ego_history_array"), (10, 3))
        if ego_history is None:
            skipped["invalid_input_ego_history_array"] += 1
            continue
        prompt_fields = get_prompt_fields(record)
        if prompt_fields is None:
            skipped["missing_input_system_or_planning_prompt"] += 1
            continue
        image_path = resolve_image_path(record, args.dataroot)
        if image_path is None:
            skipped["missing_image_path"] += 1
            continue

        samples.append(
            {
                "line_no": record.get("_line_no"),
                "image_path": image_path,
                "system_message": prompt_fields["system_message"],
                "planning_prompt": prompt_fields["planning_prompt"],
                "ego_history": ego_history.unsqueeze(0),
                "target": target,
                "target_waypoints": target_waypoints,
            }
        )
    return samples, skipped


def qwen_train_message_builder(system_message: str):
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


def train(args):
    if args.batch_size != 1:
        raise ValueError("train_predicted_geometry_fusion_head.py currently supports batch_size=1 only.")
    if args.geometry_dim != ORACLE_GEOMETRY_DESCRIPTOR_DIM:
        raise ValueError(
            f"first-version geometry descriptor dim is {ORACLE_GEOMETRY_DESCRIPTOR_DIM}, "
            f"got --geometry_dim {args.geometry_dim}."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    records, skipped = load_jsonl_records(args.jsonl)
    samples, sample_skipped = collect_samples(records, args)
    skipped.update(sample_skipped)
    print(
        f"[PredictedGeometryFusionTrain] loaded_records={len(records)} usable_samples={len(samples)} "
        f"skipped={sum(skipped.values())}"
    )
    if skipped:
        print(f"[PredictedGeometryFusionTrain] skipped_reasons={dict(skipped)}")
    print(f"[PredictedGeometryFusionTrain] source_action_schema={SOURCE_ACTION_SCHEMA}")
    print(f"[PredictedGeometryFusionTrain] train_action_schema={TRAIN_ACTION_SCHEMA}")
    print(f"[PredictedGeometryFusionTrain] waypoint_schema={WAYPOINT_SCHEMA}")
    print("[PredictedGeometryFusionTrain] geometry target = oracle descriptor from GT future_waypoints_local")
    print(
        f"[PredictedGeometryFusionTrain] loss_weights speed_weight={args.speed_weight} "
        f"curvature_weight={args.curvature_weight} geometry_weight={args.geometry_weight} "
        f"dt={args.dt} geometry_dim={args.geometry_dim}"
    )
    if not samples:
        print("[PredictedGeometryFusionTrain] no usable samples")
        return 1

    device = resolve_device(args.device)
    qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    head = PredictedGeometryFusionHead(
        qwen_hidden_dim=3584,
        history_steps=10,
        ego_dim=3,
        geometry_descriptor_dim=args.geometry_dim,
        qwen_embed_dim=512,
        ego_embed_dim=256,
        geometry_embed_dim=128,
        fusion_hidden_size=1024,
        chunk_size=10,
        action_dim=2,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    global_step = 0
    train_history = []
    for epoch in range(args.epochs):
        epoch_total_loss = 0.0
        epoch_action_loss = 0.0
        epoch_geometry_loss = 0.0
        epoch_speed_l1 = 0.0
        epoch_curvature_l1 = 0.0
        epoch_steps = 0

        for sample in samples:
            inputs = build_qwen_inputs(
                prompt=sample["planning_prompt"],
                images=sample["image_path"],
                processor=processor,
                model=qwen_model,
                args=args,
                get_message_fn=qwen_train_message_builder(sample["system_message"]),
            )
            with torch.no_grad():
                last_hidden_state = extract_qwen_hidden_states(inputs, qwen_model)
                planning_hidden = select_planning_hidden(last_hidden_state, inputs["attention_mask"])
            planning_hidden = planning_hidden.to(device=device, dtype=next(head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            target = sample["target"].to(device=device, dtype=planning_hidden.dtype)
            target_waypoints = sample["target_waypoints"].to(device=device, dtype=planning_hidden.dtype)
            target_geometry = build_oracle_geometry_descriptor_from_waypoints(
                target_waypoints,
                dt=args.dt,
            ).to(device=device, dtype=planning_hidden.dtype)

            outputs = head(planning_hidden, ego_history)
            pred = outputs["action_chunk"]
            pred_geometry = outputs["geometry_descriptor"]
            action_loss = action_chunk_l1_loss(
                pred,
                target,
                speed_weight=args.speed_weight,
                curvature_weight=args.curvature_weight,
            )
            geometry_loss = F.l1_loss(pred_geometry, target_geometry)
            total_loss = action_loss + args.geometry_weight * geometry_loss
            speed_l1 = torch.mean(torch.abs(pred[..., 0] - target[..., 0])).detach()
            curvature_l1 = torch.mean(torch.abs(pred[..., 1] - target[..., 1])).detach()

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            global_step += 1
            epoch_steps += 1
            epoch_total_loss += float(total_loss.detach())
            epoch_action_loss += float(action_loss.detach())
            epoch_geometry_loss += float(geometry_loss.detach())
            epoch_speed_l1 += float(speed_l1)
            epoch_curvature_l1 += float(curvature_l1)

            if global_step % 10 == 0:
                print(
                    f"step={global_step} total_loss={float(total_loss.detach()):.6f} "
                    f"action_loss={float(action_loss.detach()):.6f} "
                    f"geometry_descriptor_l1={float(geometry_loss.detach()):.6f} "
                    f"speed_l1={float(speed_l1):.6f} curvature_l1={float(curvature_l1):.6f}"
                )

        epoch_summary = {
            "epoch": epoch + 1,
            "avg_total_loss": epoch_total_loss / max(epoch_steps, 1),
            "avg_action_loss": epoch_action_loss / max(epoch_steps, 1),
            "avg_geometry_descriptor_l1": epoch_geometry_loss / max(epoch_steps, 1),
            "avg_speed_l1": epoch_speed_l1 / max(epoch_steps, 1),
            "avg_curvature_l1": epoch_curvature_l1 / max(epoch_steps, 1),
            "steps": epoch_steps,
        }
        train_history.append(epoch_summary)
        print(
            f"epoch={epoch_summary['epoch']} avg_total_loss={epoch_summary['avg_total_loss']:.6f} "
            f"avg_action_loss={epoch_summary['avg_action_loss']:.6f} "
            f"avg_geometry_descriptor_l1={epoch_summary['avg_geometry_descriptor_l1']:.6f} "
            f"avg_speed_l1={epoch_summary['avg_speed_l1']:.6f} "
            f"avg_curvature_l1={epoch_summary['avg_curvature_l1']:.6f} "
            f"steps={epoch_summary['steps']}"
        )

    checkpoint = {
        "predicted_geometry_fusion_head_state_dict": head.state_dict(),
        "config": {
            "qwen_hidden_dim": 3584,
            "history_steps": 10,
            "ego_dim": 3,
            "geometry_descriptor_dim": args.geometry_dim,
            "qwen_embed_dim": 512,
            "ego_embed_dim": 256,
            "geometry_embed_dim": 128,
            "fusion_hidden_size": 1024,
            "chunk_size": 10,
            "action_dim": 2,
            "dropout": args.dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "speed_weight": args.speed_weight,
            "curvature_weight": args.curvature_weight,
            "geometry_weight": args.geometry_weight,
            "dt": args.dt,
            "target_curvature_scale": 100.0,
        },
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "train_action_schema": TRAIN_ACTION_SCHEMA,
        "waypoint_schema": WAYPOINT_SCHEMA,
        "geometry_target_source": "oracle_descriptor_from_gt_future_waypoints_local",
        "jsonl_path": args.jsonl,
        "global_step": global_step,
        "train_history": train_history,
    }
    ckpt_path = os.path.join(args.output_dir, "predicted_geometry_fusion_head.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"[PredictedGeometryFusionTrain] saved checkpoint: {ckpt_path}")
    return 0


def main():
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
