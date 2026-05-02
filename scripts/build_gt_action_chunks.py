#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from nuscenes import NuScenes


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from base_planner import (
    PlannerInput,
    build_speed_curvature_prompt,
    build_speed_curvature_sys_message,
    encode_ego_state_array,
)
from utils import EstimateCurvatureFromTrajectory


def parse_args():
    parser = argparse.ArgumentParser(description="Build GT action chunk JSONL directly from nuScenes.")
    parser.add_argument("--dataroot", type=str, default="/root/autodl-tmp/data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--output_jsonl", type=str, default="/root/autodl-tmp/action_chunks_gt.jsonl")
    parser.add_argument("--history_steps", type=int, default=10)
    parser.add_argument("--future_steps", type=int, default=10)
    parser.add_argument("--camera", type=str, default="CAM_FRONT")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--method", type=str, default="gt")
    parser.add_argument("--model_path", type=str, default="qwen")
    return parser.parse_args()


def numpy_to_list_safe(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float16, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int8, np.int16, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {key: numpy_to_list_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_to_list_safe(item) for item in value]
    return value


def collect_scene_samples(nusc: NuScenes, scene: Dict[str, Any], camera: str):
    sample_tokens = []
    sample_timestamps = []
    image_paths = []
    ego_poses = []

    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        if camera not in sample["data"]:
            raise KeyError(f"camera {camera} not found in sample {token}")
        sample_data = nusc.get("sample_data", sample["data"][camera])
        image_path = os.path.join(nusc.dataroot, sample_data["filename"])
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

        sample_tokens.append(token)
        sample_timestamps.append(sample.get("timestamp", sample_data.get("timestamp")))
        image_paths.append(image_path)
        ego_poses.append(ego_pose)

        if token == scene["last_sample_token"]:
            break
        token = sample["next"]

    return sample_tokens, sample_timestamps, image_paths, ego_poses


def estimate_scene_motion(ego_poses: List[Dict[str, Any]]):
    ego_poses_world = np.asarray([pose["translation"][:3] for pose in ego_poses], dtype=np.float32)
    ego_velocities = np.zeros_like(ego_poses_world)
    ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
    if len(ego_velocities) > 1:
        ego_velocities[0] = ego_velocities[1]
    ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world).astype(np.float32)
    return ego_poses_world, ego_velocities, ego_curvatures


def build_gt_record(
    args,
    scene: Dict[str, Any],
    scene_index: int,
    frame_idx: int,
    sample_token: str,
    timestamp: Optional[int],
    image_path: str,
    obs_velocities: np.ndarray,
    obs_curvatures: np.ndarray,
    future_velocities: np.ndarray,
    future_curvatures: np.ndarray,
):
    system_message = build_speed_curvature_sys_message(article="an")
    planner_input = PlannerInput(
        images=image_path,
        obs_velocities=obs_velocities,
        obs_curvatures=obs_curvatures,
        scene_description=scene.get("description"),
        object_description=None,
        intent_description=None,
        method=args.method,
        reasoning_mode="cot",
        horizon=args.future_steps,
    )
    planning_prompt, _ = build_speed_curvature_prompt(planner_input)
    future_action_gt = np.stack(
        [np.linalg.norm(future_velocities, axis=1), future_curvatures],
        axis=1,
    ).astype(np.float32)

    return {
        "metadata": {
            "scene_name": scene.get("name"),
            "scene_index": scene_index,
            "frame_idx": frame_idx,
            "sample_token": sample_token,
            "timestamp": timestamp,
            "method": args.method,
            "model_path": args.model_path,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "image_path": image_path,
            "camera": args.camera,
        },
        "input": {
            "ego_history_array": encode_ego_state_array(obs_velocities, obs_curvatures),
            "ego_history_schema": ["relative_time", "speed_mps", "curvature_x100"],
            "system_message": system_message,
            "planning_prompt": planning_prompt,
            "prompt_type": "speed_curvature_planning",
        },
        "target": {
            "future_action_gt": future_action_gt,
            "future_action_schema": ["speed_mps", "curvature_1pm"],
        },
        "prediction": {
            "qwen_predicted_action": None,
            "qwen_predicted_action_schema": ["speed_mps", "curvature_1pm"],
            "raw_qwen_text": None,
            "raw_qwen_text_schema": ["speed_mps", "curvature_x100"],
            "retry_used": False,
            "retry_reason": None,
            "planner_guard": None,
        },
        "metrics": {},
    }


def validate_record(record: Dict[str, Any], history_steps: int, future_steps: int):
    image_path = record.get("metadata", {}).get("image_path")
    if not image_path or not os.path.exists(image_path):
        return False, "missing_image_path"

    ego_history = np.asarray(record.get("input", {}).get("ego_history_array"), dtype=np.float32)
    if ego_history.shape != (history_steps, 3) or not np.isfinite(ego_history).all():
        return False, "invalid_ego_history_array"

    future_gt = np.asarray(record.get("target", {}).get("future_action_gt"), dtype=np.float32)
    if future_gt.shape != (future_steps, 2) or not np.isfinite(future_gt).all():
        return False, "invalid_future_action_gt"

    if future_steps != 10:
        return False, "future_steps_must_be_10_for_train_action_head"

    return True, None


def write_record(record: Dict[str, Any], output_jsonl: str):
    with open(output_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(numpy_to_list_safe(record), ensure_ascii=False))
        f.write("\n")


def build_gt_action_chunks(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    if os.path.exists(args.output_jsonl):
        os.remove(args.output_jsonl)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    skipped = Counter()
    total_candidates = 0
    written = 0

    for scene_index, scene in enumerate(nusc.scene):
        try:
            sample_tokens, sample_timestamps, image_paths, ego_poses = collect_scene_samples(
                nusc,
                scene,
                args.camera,
            )
        except Exception as exc:
            skipped[f"scene_load_error:{type(exc).__name__}"] += 1
            continue

        scene_len = len(sample_tokens)
        min_len = args.history_steps + args.future_steps
        if scene_len < min_len:
            skipped["scene_too_short"] += 1
            continue

        _, ego_velocities, ego_curvatures = estimate_scene_motion(ego_poses)
        first_current_idx = args.history_steps - 1
        last_current_idx = scene_len - args.future_steps - 1

        for current_idx in range(first_current_idx, last_current_idx + 1):
            if args.max_samples > 0 and written >= args.max_samples:
                break
            total_candidates += 1

            hist_slice = slice(current_idx - args.history_steps + 1, current_idx + 1)
            fut_slice = slice(current_idx + 1, current_idx + 1 + args.future_steps)
            record = build_gt_record(
                args=args,
                scene=scene,
                scene_index=scene_index,
                frame_idx=current_idx,
                sample_token=sample_tokens[current_idx],
                timestamp=sample_timestamps[current_idx],
                image_path=image_paths[current_idx],
                obs_velocities=ego_velocities[hist_slice],
                obs_curvatures=ego_curvatures[hist_slice],
                future_velocities=ego_velocities[fut_slice],
                future_curvatures=ego_curvatures[fut_slice],
            )
            is_valid, reason = validate_record(record, args.history_steps, args.future_steps)
            if not is_valid:
                skipped[reason] += 1
                continue
            write_record(record, args.output_jsonl)
            written += 1

        if args.max_samples > 0 and written >= args.max_samples:
            break

    print(f"total scenes: {len(nusc.scene)}")
    print(f"total candidate samples: {total_candidates}")
    print(f"written records: {written}")
    print(f"skipped count by reason: {dict(skipped)}")
    print(f"output path: {args.output_jsonl}")


def main():
    args = parse_args()
    build_gt_action_chunks(args)


if __name__ == "__main__":
    main()
