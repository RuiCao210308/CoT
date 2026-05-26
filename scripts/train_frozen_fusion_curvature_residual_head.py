#!/usr/bin/env python3
import argparse
import json
import os
import random
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

from action_head import (
    ChannelWiseGatedResidualHead,
    CurvatureResidualHead,
    FrozenFusionCurvatureResidualHead,
    FusionActionHead,
    GatedGeometryResidualFusionHead,
    GeometryDescriptorHead,
    QueryGatedResidualHead,
    action_chunk_l1_loss,
)
from geometry_sequence_predictor import GEOMETRY_SEQUENCE_SCHEMA, GeometrySequencePredictor
from qwen_planner import (
    build_qwen_inputs,
    extract_qwen_hidden_states,
    select_planning_hidden,
)
from train_action_head import load_qwen_model_and_processor
from waypoint_metrics import waypoint_ade, waypoint_fde, waypoint_l1_loss


SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]
WAYPOINT_SCHEMA = ["x_m_local", "y_m_local"]
EXPECTED_QWEN_HIDDEN_DIM = 3584


def parse_args():
    parser = argparse.ArgumentParser(description="Train frozen Fusion base with curvature-only geometry residual.")
    parser.add_argument(
        "--model_type",
        choices=(
            "frozen_fusion_curvature_residual",
            "gated_geometry_residual_fusion",
            "query_gated_geometry_residual_fusion",
        ),
        default="frozen_fusion_curvature_residual",
    )
    parser.add_argument("--jsonl", type=str, default="/root/autodl-tmp/action_chunks_gt_waypoint_trainval_100scenes_train.jsonl")
    parser.add_argument("--model-path", type=str, default="qwen", help="Qwen local path or qwen alias.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--fusion_checkpoint", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/openemma_frozen_fusion_curvature_residual_trainval_100scenes_e3",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--speed_weight", type=float, default=1.0)
    parser.add_argument("--curvature_weight", type=float, default=1.0)
    parser.add_argument("--geometry_sequence_weight", type=float, default=0.5)
    parser.add_argument("--geometry_descriptor_weight", type=float, default=0.5)
    parser.add_argument("--gate_reg_weight", type=float, default=0.0)
    parser.add_argument("--descriptor_dim", type=int, default=6)
    parser.add_argument("--query_decoder_dim", type=int, default=256)
    parser.add_argument("--query_decoder_layers", type=int, default=1)
    parser.add_argument("--query_decoder_heads", type=int, default=4)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--hidden_cache", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
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
        print("[FrozenFusionCurvatureResidualTrain] cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_fusion_head_from_checkpoint(path: str, device: torch.device) -> FusionActionHead:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    head = FusionActionHead(
        qwen_hidden_dim=int(config.get("qwen_hidden_dim", 3584)),
        history_steps=int(config.get("history_steps", 10)),
        ego_dim=int(config.get("ego_dim", 3)),
        ego_embed_dim=int(config.get("ego_embed_dim", 256)),
        qwen_embed_dim=int(config.get("qwen_embed_dim", 512)),
        fusion_hidden_size=int(config.get("fusion_hidden_size", 1024)),
        chunk_size=int(config.get("chunk_size", 10)),
        action_dim=int(config.get("action_dim", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint.get("fusion_action_head_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"fusion checkpoint missing fusion_action_head_state_dict: {path}")
    head.load_state_dict(state_dict)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad = False
    return head


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hidden_cache(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    cache = torch.load(path, map_location="cpu")
    if cache.get("format") != "openemma_qwen_planning_hidden_cache_v1":
        raise ValueError(f"unsupported hidden cache format: {cache.get('format')}")
    if "hidden" not in cache or "record_to_hidden_index" not in cache:
        raise KeyError("hidden cache must contain hidden and record_to_hidden_index")
    hidden = cache["hidden"]
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("hidden cache field 'hidden' must be a torch.Tensor")
    if hidden.ndim != 2 or hidden.shape[1] != EXPECTED_QWEN_HIDDEN_DIM:
        raise ValueError(
            f"hidden cache tensor must have shape [N,{EXPECTED_QWEN_HIDDEN_DIM}], got {tuple(hidden.shape)}"
        )
    return cache


def cached_planning_hidden(cache: Dict[str, Any], record_index: int) -> torch.Tensor:
    mapping = cache.get("record_to_hidden_index", {})
    hidden_index = mapping.get(record_index)
    if hidden_index is None:
        hidden_index = mapping.get(str(record_index))
    if hidden_index is None:
        raise KeyError(f"hidden cache missing record_index={record_index}")
    hidden = cache["hidden"][int(hidden_index)]
    if hidden.shape != (EXPECTED_QWEN_HIDDEN_DIM,):
        raise ValueError(
            f"hidden cache row for record_index={record_index} must have shape "
            f"({EXPECTED_QWEN_HIDDEN_DIM},), got {tuple(hidden.shape)}"
        )
    return hidden.unsqueeze(0)


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


def geometry_descriptor_from_waypoints(waypoints: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build [endpoint_x, endpoint_y, path_length, mean_signed_curvature, max_abs_lateral_offset, heading_change]."""
    if waypoints.ndim != 3 or waypoints.shape[1:] != (10, 2):
        raise ValueError(f"waypoints must have shape [B,10,2], got {tuple(waypoints.shape)}")
    origin = torch.zeros(waypoints.shape[0], 1, 2, dtype=waypoints.dtype, device=waypoints.device)
    points = torch.cat([origin, waypoints], dim=1)
    deltas = points[:, 1:] - points[:, :-1]
    segment_lengths = torch.linalg.norm(deltas, dim=-1)
    path_length = segment_lengths.sum(dim=1)
    headings = torch.atan2(deltas[..., 1], deltas[..., 0])
    heading_deltas = torch.atan2(torch.sin(headings[:, 1:] - headings[:, :-1]), torch.cos(headings[:, 1:] - headings[:, :-1]))
    curvature_denominator = segment_lengths[:, 1:].clamp_min(eps)
    signed_curvature = heading_deltas / curvature_denominator
    mean_signed_curvature = signed_curvature.mean(dim=1)
    heading_change = torch.atan2(torch.sin(headings[:, -1] - headings[:, 0]), torch.cos(headings[:, -1] - headings[:, 0]))
    endpoint = waypoints[:, -1]
    max_abs_lateral_offset = torch.max(torch.abs(waypoints[..., 1]), dim=1).values
    return torch.stack(
        [
            endpoint[:, 0],
            endpoint[:, 1],
            path_length,
            mean_signed_curvature,
            max_abs_lateral_offset,
            heading_change,
        ],
        dim=-1,
    )


def geometry_descriptor_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


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
    use_hidden_cache = bool(args.hidden_cache)
    for record_index, record in enumerate(records):
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
        prompt_fields = None if use_hidden_cache else get_prompt_fields(record)
        image_path = None
        if not use_hidden_cache:
            if prompt_fields is None:
                skipped["missing_input_system_or_planning_prompt"] += 1
                continue
            image_path = resolve_image_path(record, args.dataroot)
            if image_path is None:
                skipped["missing_image_path"] += 1
                continue

        sample = {
            "record_index": record_index,
            "line_no": record.get("_line_no"),
            "ego_history": ego_history.unsqueeze(0),
            "target": target,
            "target_waypoints": target_waypoints,
        }
        if prompt_fields is not None:
            sample.update(prompt_fields)
            sample["image_path"] = image_path
        samples.append(sample)
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


def planning_hidden_for_sample(sample, args, hidden_cache, qwen_model, processor, device, dtype: torch.dtype) -> torch.Tensor:
    if hidden_cache is None:
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
    else:
        planning_hidden = cached_planning_hidden(hidden_cache, int(sample["record_index"]))
    return planning_hidden.to(device=device, dtype=dtype)


def train_gated_geometry_residual_fusion(args):
    is_query_decoder = args.model_type == "query_gated_geometry_residual_fusion"
    decoder_type = "query" if is_query_decoder else "mlp"
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError(
            "train_frozen_fusion_curvature_residual_head.py currently supports batch_size=1 only."
        )
    if args.descriptor_dim != 6:
        raise ValueError(f"{args.model_type} currently expects --descriptor_dim 6.")

    os.makedirs(args.output_dir, exist_ok=True)
    records, skipped = load_jsonl_records(args.jsonl)
    samples, sample_skipped = collect_samples(records, args)
    skipped.update(sample_skipped)
    print(
        f"[GatedGeometryResidualFusionTrain] loaded_records={len(records)} "
        f"usable_samples={len(samples)} skipped={sum(skipped.values())}"
    )
    if skipped:
        print(f"[GatedGeometryResidualFusionTrain] skipped_reasons={dict(skipped)}")
    print(f"[GatedGeometryResidualFusionTrain] source_action_schema={SOURCE_ACTION_SCHEMA}")
    print(f"[GatedGeometryResidualFusionTrain] train_action_schema={TRAIN_ACTION_SCHEMA}")
    print(
        f"[GatedGeometryResidualFusionTrain] fusion_checkpoint={args.fusion_checkpoint} "
        f"freeze_fusion_base=True descriptor_dim={args.descriptor_dim} "
        f"decoder_type={decoder_type} "
        f"geometry_descriptor_weight={args.geometry_descriptor_weight} "
        f"gate_reg_weight={args.gate_reg_weight} residual_scale={args.residual_scale}"
    )
    if not samples:
        print("[GatedGeometryResidualFusionTrain] no usable samples")
        return 1

    device = resolve_device(args.device)
    hidden_cache = load_hidden_cache(args.hidden_cache)
    if hidden_cache is None:
        qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    else:
        qwen_model, processor = None, None
        print(
            f"[GatedGeometryResidualFusionTrain] using_hidden_cache={args.hidden_cache} "
            f"num_cached={hidden_cache.get('num_cached')}"
        )

    frozen_fusion_head = load_fusion_head_from_checkpoint(args.fusion_checkpoint, device)
    geometry_descriptor_head = GeometryDescriptorHead(
        descriptor_dim=args.descriptor_dim,
        dropout=0.1,
    ).to(device)
    if is_query_decoder:
        gated_residual_head = QueryGatedResidualHead(
            descriptor_dim=args.descriptor_dim,
            decoder_dim=args.query_decoder_dim,
            num_queries=10,
            action_dim=2,
            num_layers=args.query_decoder_layers,
            num_heads=args.query_decoder_heads,
            dropout=0.1,
        ).to(device)
    else:
        gated_residual_head = ChannelWiseGatedResidualHead(
            descriptor_dim=args.descriptor_dim,
            dropout=0.1,
        ).to(device)
    head = GatedGeometryResidualFusionHead(
        frozen_fusion_head=frozen_fusion_head,
        geometry_descriptor_head=geometry_descriptor_head,
        gated_residual_head=gated_residual_head,
        residual_scale=args.residual_scale,
    ).to(device)
    parameters = list(geometry_descriptor_head.parameters()) + list(gated_residual_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    train_dtype = next(geometry_descriptor_head.parameters()).dtype

    global_step = 0
    train_history = []
    for epoch in range(args.epochs):
        epoch_total_loss = 0.0
        epoch_action_loss = 0.0
        epoch_geo_desc_loss = 0.0
        epoch_gate_loss = 0.0
        epoch_speed_l1 = 0.0
        epoch_curvature_l1 = 0.0
        epoch_base_speed_l1 = 0.0
        epoch_base_curvature_l1 = 0.0
        epoch_gate_mean = 0.0
        epoch_residual_abs = 0.0
        epoch_steps = 0

        for sample in samples:
            planning_hidden = planning_hidden_for_sample(
                sample, args, hidden_cache, qwen_model, processor, device, train_dtype
            )
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            target = sample["target"].to(device=device, dtype=planning_hidden.dtype)
            target_waypoints = sample["target_waypoints"].to(device=device, dtype=planning_hidden.dtype)
            target_descriptor = geometry_descriptor_from_waypoints(target_waypoints)

            outputs = head(planning_hidden, ego_history, detach_geometry=True)
            pred_action = outputs["action_chunk"]
            action_loss = action_chunk_l1_loss(
                pred_action,
                target,
                speed_weight=args.speed_weight,
                curvature_weight=args.curvature_weight,
            )
            geo_desc_loss = geometry_descriptor_l1_loss(outputs["geometry_descriptor"], target_descriptor)
            gate_loss = torch.mean(outputs["gate"])
            total_loss = (
                action_loss
                + args.geometry_descriptor_weight * geo_desc_loss
                + args.gate_reg_weight * gate_loss
            )
            speed_l1 = torch.mean(torch.abs(pred_action[..., 0] - target[..., 0])).detach()
            curvature_l1 = torch.mean(torch.abs(pred_action[..., 1] - target[..., 1])).detach()
            base_action = outputs["base_action"]
            base_speed_l1 = torch.mean(torch.abs(base_action[..., 0] - target[..., 0])).detach()
            base_curvature_l1 = torch.mean(torch.abs(base_action[..., 1] - target[..., 1])).detach()
            residual_abs = torch.mean(torch.abs(outputs["residual_action"])).detach()
            gate_mean = torch.mean(outputs["gate"]).detach()

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            global_step += 1
            epoch_steps += 1
            epoch_total_loss += float(total_loss.detach())
            epoch_action_loss += float(action_loss.detach())
            epoch_geo_desc_loss += float(geo_desc_loss.detach())
            epoch_gate_loss += float(gate_loss.detach())
            epoch_speed_l1 += float(speed_l1)
            epoch_curvature_l1 += float(curvature_l1)
            epoch_base_speed_l1 += float(base_speed_l1)
            epoch_base_curvature_l1 += float(base_curvature_l1)
            epoch_gate_mean += float(gate_mean)
            epoch_residual_abs += float(residual_abs)

            if global_step % 10 == 0:
                print(
                    f"step={global_step} total_loss={float(total_loss.detach()):.6f} "
                    f"action_loss={float(action_loss.detach()):.6f} "
                    f"geometry_descriptor_loss={float(geo_desc_loss.detach()):.6f} "
                    f"gate_loss={float(gate_loss.detach()):.6f} "
                    f"speed_l1={float(speed_l1):.6f} curvature_l1={float(curvature_l1):.6f} "
                    f"base_speed_l1={float(base_speed_l1):.6f} "
                    f"base_curvature_l1={float(base_curvature_l1):.6f} "
                    f"gate_mean={float(gate_mean):.6f} residual_abs_mean={float(residual_abs):.6f}"
                )

        epoch_summary = {
            "epoch": epoch + 1,
            "avg_total_loss": epoch_total_loss / max(epoch_steps, 1),
            "avg_action_loss": epoch_action_loss / max(epoch_steps, 1),
            "avg_geometry_descriptor_loss": epoch_geo_desc_loss / max(epoch_steps, 1),
            "avg_gate_loss": epoch_gate_loss / max(epoch_steps, 1),
            "avg_speed_l1": epoch_speed_l1 / max(epoch_steps, 1),
            "avg_curvature_l1": epoch_curvature_l1 / max(epoch_steps, 1),
            "avg_base_speed_l1": epoch_base_speed_l1 / max(epoch_steps, 1),
            "avg_base_curvature_l1": epoch_base_curvature_l1 / max(epoch_steps, 1),
            "avg_gate_mean": epoch_gate_mean / max(epoch_steps, 1),
            "avg_residual_abs_mean": epoch_residual_abs / max(epoch_steps, 1),
            "steps": epoch_steps,
        }
        train_history.append(epoch_summary)
        print(
            f"epoch={epoch_summary['epoch']} avg_total_loss={epoch_summary['avg_total_loss']:.6f} "
            f"avg_action_loss={epoch_summary['avg_action_loss']:.6f} "
            f"avg_geometry_descriptor_loss={epoch_summary['avg_geometry_descriptor_loss']:.6f} "
            f"avg_gate_loss={epoch_summary['avg_gate_loss']:.6f} "
            f"avg_speed_l1={epoch_summary['avg_speed_l1']:.6f} "
            f"avg_curvature_l1={epoch_summary['avg_curvature_l1']:.6f} "
            f"avg_gate_mean={epoch_summary['avg_gate_mean']:.6f} "
            f"avg_residual_abs_mean={epoch_summary['avg_residual_abs_mean']:.6f} "
            f"steps={epoch_summary['steps']}"
        )

    checkpoint = {
        "geometry_descriptor_head_state_dict": geometry_descriptor_head.state_dict(),
        "gated_residual_head_state_dict": gated_residual_head.state_dict(),
        "config": {
            "model_type": args.model_type,
            "fusion_checkpoint_path": args.fusion_checkpoint,
            "freeze_fusion_base": True,
            "qwen_hidden_dim": 3584,
            "history_steps": 10,
            "ego_dim": 3,
            "qwen_embed_dim": 512,
            "ego_embed_dim": 256,
            "descriptor_dim": args.descriptor_dim,
            "geometry_descriptor_schema": [
                "endpoint_x",
                "endpoint_y",
                "path_length",
                "mean_signed_curvature",
                "max_abs_lateral_offset",
                "heading_change",
            ],
            "geometry_descriptor_hidden_size": 512,
            "gated_residual_hidden_size": 512,
            "decoder_type": decoder_type,
            "query_decoder_dim": args.query_decoder_dim if is_query_decoder else None,
            "query_decoder_layers": args.query_decoder_layers if is_query_decoder else None,
            "query_decoder_heads": args.query_decoder_heads if is_query_decoder else None,
            "chunk_size": 10,
            "action_dim": 2,
            "geometry_descriptor_weight": args.geometry_descriptor_weight,
            "gate_reg_weight": args.gate_reg_weight,
            "residual_scale": args.residual_scale,
            "detach_geometry_for_action": True,
            "residual_target": "speed_and_curvature_channelwise_gated",
            "speed_source": "frozen_fusion_base_plus_gated_residual",
            "dt": args.dt,
            "dropout": 0.1,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "speed_weight": args.speed_weight,
            "curvature_weight": args.curvature_weight,
            "target_curvature_scale": 100.0,
            "seed": args.seed,
            "hidden_cache": args.hidden_cache,
        },
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "train_action_schema": TRAIN_ACTION_SCHEMA,
        "waypoint_schema": WAYPOINT_SCHEMA,
        "jsonl_path": args.jsonl,
        "global_step": global_step,
        "train_history": train_history,
    }
    if is_query_decoder:
        checkpoint["query_gated_residual_head_state_dict"] = checkpoint.pop("gated_residual_head_state_dict")
    ckpt_name = (
        "query_gated_geometry_residual_fusion_head.pt"
        if is_query_decoder
        else "gated_geometry_residual_fusion_head.pt"
    )
    ckpt_path = os.path.join(args.output_dir, ckpt_name)
    torch.save(checkpoint, ckpt_path)
    print(f"[GatedGeometryResidualFusionTrain] saved checkpoint: {ckpt_path}")
    return 0


def train(args):
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError(
            "train_frozen_fusion_curvature_residual_head.py currently supports batch_size=1 only."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    records, skipped = load_jsonl_records(args.jsonl)
    samples, sample_skipped = collect_samples(records, args)
    skipped.update(sample_skipped)
    print(
        f"[FrozenFusionCurvatureResidualTrain] loaded_records={len(records)} "
        f"usable_samples={len(samples)} skipped={sum(skipped.values())}"
    )
    if skipped:
        print(f"[FrozenFusionCurvatureResidualTrain] skipped_reasons={dict(skipped)}")
    print(f"[FrozenFusionCurvatureResidualTrain] source_action_schema={SOURCE_ACTION_SCHEMA}")
    print(f"[FrozenFusionCurvatureResidualTrain] train_action_schema={TRAIN_ACTION_SCHEMA}")
    print(f"[FrozenFusionCurvatureResidualTrain] geometry_sequence_schema={GEOMETRY_SEQUENCE_SCHEMA}")
    print(
        f"[FrozenFusionCurvatureResidualTrain] fusion_checkpoint={args.fusion_checkpoint} "
        f"freeze_fusion_base=True loss_weights speed_weight={args.speed_weight} "
        f"curvature_weight={args.curvature_weight} "
        f"geometry_sequence_weight={args.geometry_sequence_weight} residual_scale={args.residual_scale} "
        f"detach_geometry_for_action=True residual_target=curvature_only speed_source=frozen_fusion_base dt={args.dt}"
    )
    if not samples:
        print("[FrozenFusionCurvatureResidualTrain] no usable samples")
        return 1

    device = resolve_device(args.device)
    hidden_cache = load_hidden_cache(args.hidden_cache)
    if hidden_cache is None:
        qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    else:
        qwen_model, processor = None, None
        print(
            f"[FrozenFusionCurvatureResidualTrain] using_hidden_cache={args.hidden_cache} "
            f"num_cached={hidden_cache.get('num_cached')}"
        )
    frozen_fusion_head = load_fusion_head_from_checkpoint(args.fusion_checkpoint, device)
    geometry_predictor = GeometrySequencePredictor(dropout=0.1).to(device)
    residual_head = CurvatureResidualHead(
        dropout=0.1,
    ).to(device)
    head = FrozenFusionCurvatureResidualHead(
        frozen_fusion_head=frozen_fusion_head,
        residual_head=residual_head,
        residual_scale=args.residual_scale,
    ).to(device)
    parameters = list(geometry_predictor.parameters()) + list(residual_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)

    global_step = 0
    train_history = []
    for epoch in range(args.epochs):
        epoch_total_loss = 0.0
        epoch_action_loss = 0.0
        epoch_speed_l1 = 0.0
        epoch_curvature_l1 = 0.0
        epoch_geometry_sequence_loss = 0.0
        epoch_waypoint_ade = 0.0
        epoch_waypoint_fde = 0.0
        epoch_residual_curvature_abs = 0.0
        epoch_steps = 0

        for sample in samples:
            if hidden_cache is None:
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
            else:
                planning_hidden = cached_planning_hidden(hidden_cache, int(sample["record_index"]))
            planning_hidden = planning_hidden.to(device=device, dtype=next(residual_head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            target = sample["target"].to(device=device, dtype=planning_hidden.dtype)
            target_waypoints = sample["target_waypoints"].to(device=device, dtype=planning_hidden.dtype)

            pred_geometry_sequence = geometry_predictor(planning_hidden, ego_history)
            geometry_for_action = pred_geometry_sequence.detach()
            outputs = head(
                planning_hidden,
                ego_history,
                geometry_for_action,
                detach_geometry=True,
            )
            pred_action = outputs["action_chunk"]
            residual_curvature_abs = torch.mean(torch.abs(outputs["residual_curvature"])).detach()
            action_loss = action_chunk_l1_loss(
                pred_action,
                target,
                speed_weight=args.speed_weight,
                curvature_weight=args.curvature_weight,
            )
            geometry_sequence_loss = waypoint_l1_loss(pred_geometry_sequence, target_waypoints)
            total_loss = action_loss + args.geometry_sequence_weight * geometry_sequence_loss
            speed_l1 = torch.mean(torch.abs(pred_action[..., 0] - target[..., 0])).detach()
            curvature_l1 = torch.mean(torch.abs(pred_action[..., 1] - target[..., 1])).detach()
            wp_ade = waypoint_ade(pred_geometry_sequence, target_waypoints).detach()
            wp_fde = waypoint_fde(pred_geometry_sequence, target_waypoints).detach()

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            global_step += 1
            epoch_steps += 1
            epoch_total_loss += float(total_loss.detach())
            epoch_action_loss += float(action_loss.detach())
            epoch_speed_l1 += float(speed_l1)
            epoch_curvature_l1 += float(curvature_l1)
            epoch_geometry_sequence_loss += float(geometry_sequence_loss.detach())
            epoch_waypoint_ade += float(wp_ade)
            epoch_waypoint_fde += float(wp_fde)
            epoch_residual_curvature_abs += float(residual_curvature_abs)

            if global_step % 10 == 0:
                print(
                    f"step={global_step} total_loss={float(total_loss.detach()):.6f} "
                    f"action_loss={float(action_loss.detach()):.6f} "
                    f"speed_l1={float(speed_l1):.6f} curvature_l1={float(curvature_l1):.6f} "
                    f"geometry_sequence_loss={float(geometry_sequence_loss.detach()):.6f} "
                    f"waypoint_ade={float(wp_ade):.6f} waypoint_fde={float(wp_fde):.6f} "
                    f"residual_scale={args.residual_scale:.6f} "
                    f"residual_curvature_l1_abs_mean={float(residual_curvature_abs):.6f}"
                )

        epoch_summary = {
            "epoch": epoch + 1,
            "avg_total_loss": epoch_total_loss / max(epoch_steps, 1),
            "avg_action_loss": epoch_action_loss / max(epoch_steps, 1),
            "avg_speed_l1": epoch_speed_l1 / max(epoch_steps, 1),
            "avg_curvature_l1": epoch_curvature_l1 / max(epoch_steps, 1),
            "avg_geometry_sequence_loss": epoch_geometry_sequence_loss / max(epoch_steps, 1),
            "avg_waypoint_ade": epoch_waypoint_ade / max(epoch_steps, 1),
            "avg_waypoint_fde": epoch_waypoint_fde / max(epoch_steps, 1),
            "avg_residual_curvature_l1_abs_mean": epoch_residual_curvature_abs / max(epoch_steps, 1),
            "steps": epoch_steps,
        }
        train_history.append(epoch_summary)
        print(
            f"epoch={epoch_summary['epoch']} avg_total_loss={epoch_summary['avg_total_loss']:.6f} "
            f"avg_action_loss={epoch_summary['avg_action_loss']:.6f} "
            f"avg_speed_l1={epoch_summary['avg_speed_l1']:.6f} "
            f"avg_curvature_l1={epoch_summary['avg_curvature_l1']:.6f} "
            f"avg_geometry_sequence_loss={epoch_summary['avg_geometry_sequence_loss']:.6f} "
            f"avg_waypoint_ade={epoch_summary['avg_waypoint_ade']:.6f} "
            f"avg_waypoint_fde={epoch_summary['avg_waypoint_fde']:.6f} "
            f"avg_residual_curvature_l1_abs_mean={epoch_summary['avg_residual_curvature_l1_abs_mean']:.6f} "
            f"steps={epoch_summary['steps']}"
        )

    checkpoint = {
        "geometry_sequence_predictor_state_dict": geometry_predictor.state_dict(),
        "curvature_residual_head_state_dict": residual_head.state_dict(),
        "config": {
            "model_type": "frozen_fusion_curvature_residual",
            "fusion_checkpoint_path": args.fusion_checkpoint,
            "freeze_fusion_base": True,
            "qwen_hidden_dim": 3584,
            "history_steps": 10,
            "ego_dim": 3,
            "qwen_embed_dim": 512,
            "ego_embed_dim": 256,
            "global_context_dim": 512,
            "geometry_step_embed_dim": 128,
            "base_hidden_size": 1024,
            "residual_hidden_size": 256,
            "chunk_size": 10,
            "action_dim": 2,
            "geometry_sequence_dim": 2,
            "geometry_sequence_schema": GEOMETRY_SEQUENCE_SCHEMA,
            "geometry_sequence_weight": args.geometry_sequence_weight,
            "residual_scale": args.residual_scale,
            "detach_geometry_for_action": True,
            "residual_target": "curvature_only",
            "speed_source": "frozen_fusion_base",
            "dt": args.dt,
            "dropout": 0.1,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "speed_weight": args.speed_weight,
            "curvature_weight": args.curvature_weight,
            "target_curvature_scale": 100.0,
            "seed": args.seed,
            "hidden_cache": args.hidden_cache,
        },
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "train_action_schema": TRAIN_ACTION_SCHEMA,
        "waypoint_schema": WAYPOINT_SCHEMA,
        "jsonl_path": args.jsonl,
        "global_step": global_step,
        "train_history": train_history,
    }
    ckpt_path = os.path.join(args.output_dir, "frozen_fusion_curvature_residual_head.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"[FrozenFusionCurvatureResidualTrain] saved checkpoint: {ckpt_path}")
    return 0


def main():
    args = parse_args()
    if args.model_type in ("gated_geometry_residual_fusion", "query_gated_geometry_residual_fusion"):
        return train_gated_geometry_residual_fusion(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
