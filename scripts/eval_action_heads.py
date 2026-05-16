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

from action_head import (
    ContinuousActionHead,
    DecoupledEgoVLAActionHead,
    EgoOnlyActionHead,
    FusionActionHead,
    OracleGeometryFusionHead,
    PredictedGeometryFusionHead,
    PredictedGeometrySequenceFusionHead,
    WaypointAuxFusionHead,
)
from geometry_token import (
    ORACLE_GEOMETRY_PURPOSE,
    build_oracle_geometry_descriptor_from_waypoints,
)
from geometry_sequence_predictor import GEOMETRY_SEQUENCE_SCHEMA, GeometrySequencePredictor
from qwen_planner import (
    build_qwen_inputs,
    extract_qwen_hidden_states,
    select_planning_hidden,
)
from train_action_head import load_qwen_model_and_processor
from waypoint_metrics import (
    derive_action_from_waypoints,
    waypoint_ade,
    waypoint_fde,
    waypoint_l1_loss,
    waypoint_longitudinal_lateral_mae,
)


SOURCE_ACTION_SCHEMA = ["speed_mps", "curvature_1pm"]
TRAIN_ACTION_SCHEMA = ["speed_mps", "curvature_x100"]
MODEL_TYPES = (
    "ego_only",
    "qwen_hidden",
    "fusion",
    "decoupled_egovla",
    "waypoint_aux_fusion",
    "waypoint_consistency_fusion",
    "oracle_geometry_fusion",
    "predicted_geometry_fusion",
    "predicted_geometry_sequence_fusion",
)


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
    parser.add_argument("--dt", type=float, default=None, help="Waypoint-to-action dt; defaults to checkpoint config then 0.5.")
    parser.add_argument("--max_abs_curvature_1pm", type=float, default=0.2)
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


def make_waypoint_target(record: Dict[str, Any]) -> Optional[torch.Tensor]:
    target = record.get("target", {})
    if target.get("future_waypoints_schema") != ["x_m_local", "y_m_local"]:
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

        if args.model_type in (
            "ego_only",
            "fusion",
            "decoupled_egovla",
            "waypoint_aux_fusion",
            "waypoint_consistency_fusion",
            "oracle_geometry_fusion",
            "predicted_geometry_fusion",
            "predicted_geometry_sequence_fusion",
        ):
            ego_history = tensor_or_none(record.get("input", {}).get("ego_history_array"), (10, 3))
            if ego_history is None:
                skipped["invalid_input_ego_history_array"] += 1
                continue
            sample["ego_history"] = ego_history.unsqueeze(0)

        if args.model_type in (
            "qwen_hidden",
            "fusion",
            "decoupled_egovla",
            "waypoint_aux_fusion",
            "waypoint_consistency_fusion",
            "oracle_geometry_fusion",
            "predicted_geometry_fusion",
            "predicted_geometry_sequence_fusion",
        ):
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

        if args.model_type in ("waypoint_aux_fusion", "waypoint_consistency_fusion"):
            target_waypoints = make_waypoint_target(record)
            if target_waypoints is None:
                skipped["invalid_target_future_waypoints_local"] += 1
                continue
            sample["target_waypoints"] = target_waypoints
        elif args.model_type == "oracle_geometry_fusion":
            target_waypoints = make_waypoint_target(record)
            if target_waypoints is None:
                skipped["invalid_target_future_waypoints_local"] += 1
                continue
            sample["target_waypoints"] = target_waypoints
        elif args.model_type == "predicted_geometry_fusion":
            target_waypoints = make_waypoint_target(record)
            if target_waypoints is None:
                skipped["invalid_target_future_waypoints_local"] += 1
                continue
            sample["target_waypoints"] = target_waypoints
        elif args.model_type == "predicted_geometry_sequence_fusion":
            target_waypoints = make_waypoint_target(record)
            if target_waypoints is None:
                skipped["invalid_target_future_waypoints_local"] += 1
                continue
            sample["target_waypoints"] = target_waypoints

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


def load_fusion_head(checkpoint: Dict[str, Any], device: torch.device) -> FusionActionHead:
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
        raise KeyError("checkpoint missing fusion_action_head_state_dict")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def load_decoupled_egovla_head(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> DecoupledEgoVLAActionHead:
    config = checkpoint.get("config", {})
    head = DecoupledEgoVLAActionHead(
        qwen_hidden_dim=int(config.get("qwen_hidden_dim", 3584)),
        history_steps=int(config.get("history_steps", 10)),
        ego_dim=int(config.get("ego_dim", 3)),
        qwen_embed_dim=int(config.get("qwen_embed_dim", 512)),
        ego_embed_dim=int(config.get("ego_embed_dim", 256)),
        fusion_dim=int(config.get("fusion_dim", 1024)),
        speed_hidden=int(config.get("speed_hidden", 512)),
        curvature_hidden=int(config.get("curvature_hidden", 512)),
        chunk_size=int(config.get("chunk_size", 10)),
        action_dim=int(config.get("action_dim", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint.get("decoupled_egovla_head_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("checkpoint missing decoupled_egovla_head_state_dict")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def load_waypoint_aux_fusion_head(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> WaypointAuxFusionHead:
    config = checkpoint.get("config", {})
    head = WaypointAuxFusionHead(
        qwen_hidden_dim=int(config.get("qwen_hidden_dim", 3584)),
        history_steps=int(config.get("history_steps", 10)),
        ego_dim=int(config.get("ego_dim", 3)),
        qwen_embed_dim=int(config.get("qwen_embed_dim", 512)),
        ego_embed_dim=int(config.get("ego_embed_dim", 256)),
        fusion_hidden_size=int(config.get("fusion_hidden_size", 1024)),
        chunk_size=int(config.get("chunk_size", 10)),
        action_dim=int(config.get("action_dim", 2)),
        waypoint_dim=int(config.get("waypoint_dim", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint.get("waypoint_aux_fusion_head_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("checkpoint missing waypoint_aux_fusion_head_state_dict")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def load_oracle_geometry_fusion_head(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> OracleGeometryFusionHead:
    config = checkpoint.get("config", {})
    head = OracleGeometryFusionHead(
        qwen_hidden_dim=int(config.get("qwen_hidden_dim", 3584)),
        history_steps=int(config.get("history_steps", 10)),
        ego_dim=int(config.get("ego_dim", 3)),
        geometry_descriptor_dim=int(config.get("geometry_descriptor_dim", 16)),
        qwen_embed_dim=int(config.get("qwen_embed_dim", 512)),
        ego_embed_dim=int(config.get("ego_embed_dim", 256)),
        geometry_embed_dim=int(config.get("geometry_embed_dim", 128)),
        fusion_hidden_size=int(config.get("fusion_hidden_size", 1024)),
        chunk_size=int(config.get("chunk_size", 10)),
        action_dim=int(config.get("action_dim", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint.get("oracle_geometry_fusion_head_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("checkpoint missing oracle_geometry_fusion_head_state_dict")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def load_predicted_geometry_fusion_head(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> PredictedGeometryFusionHead:
    config = checkpoint.get("config", {})
    head = PredictedGeometryFusionHead(
        qwen_hidden_dim=int(config.get("qwen_hidden_dim", 3584)),
        history_steps=int(config.get("history_steps", 10)),
        ego_dim=int(config.get("ego_dim", 3)),
        geometry_descriptor_dim=int(config.get("geometry_descriptor_dim", 16)),
        qwen_embed_dim=int(config.get("qwen_embed_dim", 512)),
        ego_embed_dim=int(config.get("ego_embed_dim", 256)),
        geometry_embed_dim=int(config.get("geometry_embed_dim", 128)),
        fusion_hidden_size=int(config.get("fusion_hidden_size", 1024)),
        chunk_size=int(config.get("chunk_size", 10)),
        action_dim=int(config.get("action_dim", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    state_dict = checkpoint.get("predicted_geometry_fusion_head_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("checkpoint missing predicted_geometry_fusion_head_state_dict")
    head.load_state_dict(state_dict)
    head.eval()
    return head


def load_geometry_sequence_modules(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    config = checkpoint.get("config", {})
    predictor = GeometrySequencePredictor(
        qwen_hidden_dim=int(config.get("qwen_hidden_dim", 3584)),
        history_steps=int(config.get("history_steps", 10)),
        ego_dim=int(config.get("ego_dim", 3)),
        qwen_embed_dim=int(config.get("qwen_embed_dim", 512)),
        ego_embed_dim=int(config.get("ego_embed_dim", 256)),
        fusion_hidden_size=int(config.get("predictor_fusion_hidden_size", 1024)),
        future_steps=int(config.get("chunk_size", 10)),
        geometry_sequence_dim=int(config.get("geometry_sequence_dim", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    predictor_state = checkpoint.get("geometry_sequence_predictor_state_dict")
    if predictor_state is None:
        raise KeyError("checkpoint missing geometry_sequence_predictor_state_dict")
    predictor.load_state_dict(predictor_state)
    predictor.eval()

    head = PredictedGeometrySequenceFusionHead(
        qwen_hidden_dim=int(config.get("qwen_hidden_dim", 3584)),
        history_steps=int(config.get("history_steps", 10)),
        ego_dim=int(config.get("ego_dim", 3)),
        qwen_embed_dim=int(config.get("qwen_embed_dim", 512)),
        ego_embed_dim=int(config.get("ego_embed_dim", 256)),
        global_context_dim=int(config.get("global_context_dim", 512)),
        geometry_step_embed_dim=int(config.get("geometry_step_embed_dim", 128)),
        per_step_hidden_size=int(config.get("per_step_hidden_size", 512)),
        chunk_size=int(config.get("chunk_size", 10)),
        action_dim=int(config.get("action_dim", 2)),
        geometry_sequence_dim=int(config.get("geometry_sequence_dim", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    head_state = checkpoint.get("predicted_geometry_sequence_fusion_head_state_dict")
    if head_state is None:
        raise KeyError("checkpoint missing predicted_geometry_sequence_fusion_head_state_dict")
    head.load_state_dict(head_state)
    head.eval()
    return {"predictor": predictor, "head": head}


def resolve_consistency_dt(args, checkpoint: Dict[str, Any]) -> float:
    if args.dt is not None:
        return float(args.dt)
    config = checkpoint.get("config", {})
    if "dt" in config:
        return float(config["dt"])
    return 0.5


def resolve_checkpoint_dt(args, checkpoint: Dict[str, Any]) -> float:
    if args.dt is not None:
        return float(args.dt)
    config = checkpoint.get("config", {})
    if "dt" in config:
        return float(config["dt"])
    return 0.5


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


def update_waypoint_metric_sums(
    sums: Dict[str, float],
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
):
    pred_waypoints = pred_waypoints.detach().float().cpu()
    target_waypoints = target_waypoints.detach().float().cpu()
    x_mae, y_mae = waypoint_longitudinal_lateral_mae(pred_waypoints, target_waypoints)
    sums["waypoint_samples"] += int(pred_waypoints.shape[0])
    sums["waypoint_points"] += int(pred_waypoints.shape[0] * pred_waypoints.shape[1])
    sums["waypoint_l1_sum"] += float(waypoint_l1_loss(pred_waypoints, target_waypoints)) * int(pred_waypoints.numel())
    sums["waypoint_values"] += int(pred_waypoints.numel())
    sums["waypoint_ade_sum"] += float(waypoint_ade(pred_waypoints, target_waypoints)) * int(
        pred_waypoints.shape[0] * pred_waypoints.shape[1]
    )
    sums["waypoint_fde_sum"] += float(waypoint_fde(pred_waypoints, target_waypoints)) * int(pred_waypoints.shape[0])
    sums["waypoint_x_abs_sum"] += float(x_mae) * int(pred_waypoints.shape[0] * pred_waypoints.shape[1])
    sums["waypoint_y_abs_sum"] += float(y_mae) * int(pred_waypoints.shape[0] * pred_waypoints.shape[1])


def update_consistency_metric_sums(
    sums: Dict[str, float],
    pred_train: torch.Tensor,
    pred_waypoints: torch.Tensor,
    dt: float,
    max_abs_curvature_1pm: float,
):
    derived_action = derive_action_from_waypoints(
        pred_waypoints,
        dt=dt,
        max_abs_curvature_1pm=max_abs_curvature_1pm,
    )
    pred_train = pred_train.detach().float().cpu()
    derived_action = derived_action.detach().float().cpu()
    abs_error = torch.abs(pred_train - derived_action)
    sums["consistency_points"] += int(pred_train.shape[0] * pred_train.shape[1])
    sums["consistency_values"] += int(abs_error.numel())
    sums["consistency_abs_train_scale"] += float(abs_error.sum())
    sums["derived_speed_abs_vs_pred"] += float(abs_error[..., 0].sum())
    sums["derived_curvature_abs_x100_vs_pred"] += float(abs_error[..., 1].sum())
    sums["consistency_dt"] = float(dt)
    sums["max_abs_curvature_1pm"] = float(max_abs_curvature_1pm)


def update_geometry_sequence_metric_sums(
    sums: Dict[str, float],
    pred_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
):
    pred_sequence = pred_sequence.detach().float().cpu()
    target_sequence = target_sequence.detach().float().cpu()
    x_mae, y_mae = waypoint_longitudinal_lateral_mae(pred_sequence, target_sequence)
    sums["geometry_sequence_samples"] += int(pred_sequence.shape[0])
    sums["geometry_sequence_points"] += int(pred_sequence.shape[0] * pred_sequence.shape[1])
    sums["geometry_sequence_values"] += int(pred_sequence.numel())
    sums["geometry_sequence_l1_sum"] += float(waypoint_l1_loss(pred_sequence, target_sequence)) * int(
        pred_sequence.numel()
    )
    sums["geometry_sequence_ade_sum"] += float(waypoint_ade(pred_sequence, target_sequence)) * int(
        pred_sequence.shape[0] * pred_sequence.shape[1]
    )
    sums["geometry_sequence_fde_sum"] += float(waypoint_fde(pred_sequence, target_sequence)) * int(
        pred_sequence.shape[0]
    )
    sums["geometry_sequence_x_abs_sum"] += float(x_mae) * int(pred_sequence.shape[0] * pred_sequence.shape[1])
    sums["geometry_sequence_y_abs_sum"] += float(y_mae) * int(pred_sequence.shape[0] * pred_sequence.shape[1])


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
        "waypoint_l1": None,
        "waypoint_ade": None,
        "waypoint_fde": None,
        "waypoint_x_mae": None,
        "waypoint_y_mae": None,
        "dt": sums.get("consistency_dt", args.dt),
        "max_abs_curvature_1pm": sums.get("max_abs_curvature_1pm", args.max_abs_curvature_1pm),
        "consistency_l1_train_scale": None,
        "derived_speed_mae_vs_pred_mps": None,
        "derived_curvature_mae_x100_vs_pred": None,
        "oracle_geometry_from_gt_waypoints": sums.get("oracle_geometry_from_gt_waypoints"),
        "purpose": sums.get("oracle_geometry_purpose"),
        "geometry_descriptor_dim": sums.get("geometry_descriptor_dim"),
        "geometry_descriptor_l1": None,
        "geometry_sequence_dim": sums.get("geometry_sequence_dim"),
        "geometry_sequence_schema": sums.get("geometry_sequence_schema"),
        "geometry_sequence_l1": None,
        "geometry_sequence_ade": None,
        "geometry_sequence_fde": None,
        "geometry_sequence_x_mae": None,
        "geometry_sequence_y_mae": None,
    }
    if num_points == 0:
        return metrics

    metrics["speed_mae_mps"] = sums["speed_abs"] / num_points
    metrics["curvature_mae_1pm"] = sums["curvature_abs_1pm"] / num_points
    metrics["curvature_mae_x100"] = sums["curvature_abs_x100"] / num_points
    metrics["overall_l1_train_scale"] = sums["overall_abs_train_scale"] / num_action_values
    metrics["speed_rmse_mps"] = (sums["speed_sq"] / num_points) ** 0.5
    metrics["curvature_rmse_x100"] = (sums["curvature_sq_x100"] / num_points) ** 0.5
    waypoint_points = int(sums["waypoint_points"])
    waypoint_values = int(sums["waypoint_values"])
    waypoint_samples = int(sums["waypoint_samples"])
    if waypoint_points > 0 and waypoint_values > 0 and waypoint_samples > 0:
        metrics["waypoint_l1"] = sums["waypoint_l1_sum"] / waypoint_values
        metrics["waypoint_ade"] = sums["waypoint_ade_sum"] / waypoint_points
        metrics["waypoint_fde"] = sums["waypoint_fde_sum"] / waypoint_samples
        metrics["waypoint_x_mae"] = sums["waypoint_x_abs_sum"] / waypoint_points
        metrics["waypoint_y_mae"] = sums["waypoint_y_abs_sum"] / waypoint_points
    consistency_points = int(sums["consistency_points"])
    consistency_values = int(sums["consistency_values"])
    if consistency_points > 0 and consistency_values > 0:
        metrics["consistency_l1_train_scale"] = sums["consistency_abs_train_scale"] / consistency_values
        metrics["derived_speed_mae_vs_pred_mps"] = sums["derived_speed_abs_vs_pred"] / consistency_points
        metrics["derived_curvature_mae_x100_vs_pred"] = (
            sums["derived_curvature_abs_x100_vs_pred"] / consistency_points
        )
    geometry_descriptor_values = int(sums["geometry_descriptor_values"])
    if geometry_descriptor_values > 0:
        metrics["geometry_descriptor_l1"] = sums["geometry_descriptor_l1_sum"] / geometry_descriptor_values
    geometry_sequence_points = int(sums["geometry_sequence_points"])
    geometry_sequence_values = int(sums["geometry_sequence_values"])
    geometry_sequence_samples = int(sums["geometry_sequence_samples"])
    if geometry_sequence_points > 0 and geometry_sequence_values > 0 and geometry_sequence_samples > 0:
        metrics["geometry_sequence_l1"] = sums["geometry_sequence_l1_sum"] / geometry_sequence_values
        metrics["geometry_sequence_ade"] = sums["geometry_sequence_ade_sum"] / geometry_sequence_points
        metrics["geometry_sequence_fde"] = sums["geometry_sequence_fde_sum"] / geometry_sequence_samples
        metrics["geometry_sequence_x_mae"] = sums["geometry_sequence_x_abs_sum"] / geometry_sequence_points
        metrics["geometry_sequence_y_mae"] = sums["geometry_sequence_y_abs_sum"] / geometry_sequence_points
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
    if metrics.get("waypoint_ade") is not None:
        print(f"waypoint_l1={metrics['waypoint_l1']}")
        print(f"waypoint_ade={metrics['waypoint_ade']}")
        print(f"waypoint_fde={metrics['waypoint_fde']}")
        print(f"waypoint_x_mae={metrics['waypoint_x_mae']}")
        print(f"waypoint_y_mae={metrics['waypoint_y_mae']}")
    if metrics.get("consistency_l1_train_scale") is not None:
        print(f"dt={metrics['dt']}")
        print(f"max_abs_curvature_1pm={metrics['max_abs_curvature_1pm']}")
        print(f"consistency_l1_train_scale={metrics['consistency_l1_train_scale']}")
        print(f"derived_speed_mae_vs_pred_mps={metrics['derived_speed_mae_vs_pred_mps']}")
        print(f"derived_curvature_mae_x100_vs_pred={metrics['derived_curvature_mae_x100_vs_pred']}")
    if metrics.get("oracle_geometry_from_gt_waypoints") is not None:
        print(f"oracle_geometry_from_gt_waypoints={metrics['oracle_geometry_from_gt_waypoints']}")
        print(f"purpose={metrics['purpose']}")
        print(f"geometry_descriptor_dim={metrics['geometry_descriptor_dim']}")
        print(f"dt={metrics['dt']}")
    if metrics.get("geometry_descriptor_l1") is not None:
        print(f"geometry_descriptor_l1={metrics['geometry_descriptor_l1']}")
    if metrics.get("geometry_sequence_l1") is not None:
        print(f"geometry_sequence_dim={metrics['geometry_sequence_dim']}")
        print(f"geometry_sequence_schema={metrics['geometry_sequence_schema']}")
        print(f"geometry_sequence_l1={metrics['geometry_sequence_l1']}")
        print(f"geometry_sequence_ade={metrics['geometry_sequence_ade']}")
        print(f"geometry_sequence_fde={metrics['geometry_sequence_fde']}")
        print(f"geometry_sequence_x_mae={metrics['geometry_sequence_x_mae']}")
        print(f"geometry_sequence_y_mae={metrics['geometry_sequence_y_mae']}")


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


def evaluate_fusion(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    fusion_head = load_fusion_head(checkpoint, device)
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
            planning_hidden = planning_hidden.to(device=device, dtype=next(fusion_head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            pred_train = fusion_head(planning_hidden, ego_history)
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
    return sums


def evaluate_decoupled_egovla(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    head = load_decoupled_egovla_head(checkpoint, device)
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
            planning_hidden = planning_hidden.to(device=device, dtype=next(head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            pred_train = head(planning_hidden, ego_history)
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
    return sums


def evaluate_waypoint_aux_fusion(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    head = load_waypoint_aux_fusion_head(checkpoint, device)
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
            planning_hidden = planning_hidden.to(device=device, dtype=next(head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            outputs = head(planning_hidden, ego_history)
            pred_train = outputs["action_chunk"]
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
            target_waypoints = sample["target_waypoints"].to(device=device, dtype=outputs["waypoints"].dtype)
            update_waypoint_metric_sums(sums, outputs["waypoints"], target_waypoints)
    return sums


def evaluate_waypoint_consistency_fusion(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    head = load_waypoint_aux_fusion_head(checkpoint, device)
    dt = resolve_consistency_dt(args, checkpoint)
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
            planning_hidden = planning_hidden.to(device=device, dtype=next(head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            outputs = head(planning_hidden, ego_history)
            pred_train = outputs["action_chunk"]
            pred_waypoints = outputs["waypoints"]
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
            target_waypoints = sample["target_waypoints"].to(device=device, dtype=pred_waypoints.dtype)
            update_waypoint_metric_sums(sums, pred_waypoints, target_waypoints)
            update_consistency_metric_sums(
                sums,
                pred_train,
                pred_waypoints,
                dt=dt,
                max_abs_curvature_1pm=args.max_abs_curvature_1pm,
            )
    return sums


def evaluate_oracle_geometry_fusion(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    head = load_oracle_geometry_fusion_head(checkpoint, device)
    config = checkpoint.get("config", {})
    dt = resolve_checkpoint_dt(args, checkpoint)
    qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    sums = Counter()
    sums["oracle_geometry_from_gt_waypoints"] = True
    sums["oracle_geometry_purpose"] = str(config.get("purpose", ORACLE_GEOMETRY_PURPOSE))
    sums["geometry_descriptor_dim"] = int(config.get("geometry_descriptor_dim", 16))
    sums["consistency_dt"] = float(dt)
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
            planning_hidden = planning_hidden.to(device=device, dtype=next(head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            target_waypoints = sample["target_waypoints"].to(device=device, dtype=planning_hidden.dtype)
            geometry_descriptor = build_oracle_geometry_descriptor_from_waypoints(
                target_waypoints,
                dt=dt,
            ).to(device=device, dtype=planning_hidden.dtype)
            pred_train = head(planning_hidden, ego_history, geometry_descriptor)
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
    return sums


def evaluate_predicted_geometry_fusion(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    head = load_predicted_geometry_fusion_head(checkpoint, device)
    config = checkpoint.get("config", {})
    dt = resolve_checkpoint_dt(args, checkpoint)
    qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    sums = Counter()
    sums["geometry_descriptor_dim"] = int(config.get("geometry_descriptor_dim", 16))
    sums["consistency_dt"] = float(dt)
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
            planning_hidden = planning_hidden.to(device=device, dtype=next(head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            target_waypoints = sample["target_waypoints"].to(device=device, dtype=planning_hidden.dtype)
            target_geometry = build_oracle_geometry_descriptor_from_waypoints(
                target_waypoints,
                dt=dt,
            ).to(device=device, dtype=planning_hidden.dtype)
            outputs = head(planning_hidden, ego_history)
            pred_train = outputs["action_chunk"]
            pred_geometry = outputs["geometry_descriptor"]
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
            geometry_l1 = F.l1_loss(pred_geometry, target_geometry, reduction="sum")
            sums["geometry_descriptor_l1_sum"] += float(geometry_l1.detach().float().cpu())
            sums["geometry_descriptor_values"] += int(pred_geometry.numel())
    return sums


def evaluate_predicted_geometry_sequence_fusion(args, samples: List[Dict[str, Any]], device: torch.device):
    checkpoint = load_checkpoint(args.checkpoint, device)
    modules = load_geometry_sequence_modules(checkpoint, device)
    predictor = modules["predictor"]
    head = modules["head"]
    config = checkpoint.get("config", {})
    qwen_model, processor = load_qwen_model_and_processor(args.model_path, str(device))
    sums = Counter()
    sums["geometry_sequence_dim"] = int(config.get("geometry_sequence_dim", 2))
    sums["geometry_sequence_schema"] = config.get("geometry_sequence_schema", GEOMETRY_SEQUENCE_SCHEMA)
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
            planning_hidden = planning_hidden.to(device=device, dtype=next(head.parameters()).dtype)
            ego_history = sample["ego_history"].to(device=device, dtype=planning_hidden.dtype)
            pred_geometry_sequence = predictor(planning_hidden, ego_history)
            pred_train = head(planning_hidden, ego_history, pred_geometry_sequence)
            target_train = sample["target_train"].to(device=device, dtype=pred_train.dtype)
            update_metric_sums(sums, pred_train, target_train)
            target_waypoints = sample["target_waypoints"].to(device=device, dtype=pred_geometry_sequence.dtype)
            update_geometry_sequence_metric_sums(sums, pred_geometry_sequence, target_waypoints)
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
    elif args.model_type == "qwen_hidden":
        sums = evaluate_qwen_hidden(args, samples, device)
    elif args.model_type == "fusion":
        sums = evaluate_fusion(args, samples, device)
    elif args.model_type == "decoupled_egovla":
        sums = evaluate_decoupled_egovla(args, samples, device)
    elif args.model_type == "waypoint_aux_fusion":
        sums = evaluate_waypoint_aux_fusion(args, samples, device)
    elif args.model_type == "waypoint_consistency_fusion":
        sums = evaluate_waypoint_consistency_fusion(args, samples, device)
    elif args.model_type == "oracle_geometry_fusion":
        sums = evaluate_oracle_geometry_fusion(args, samples, device)
    elif args.model_type == "predicted_geometry_fusion":
        sums = evaluate_predicted_geometry_fusion(args, samples, device)
    else:
        sums = evaluate_predicted_geometry_sequence_fusion(args, samples, device)

    metrics = finalize_metrics(args, len(records), skipped, sums)
    print_summary(metrics)
    save_report(metrics, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
