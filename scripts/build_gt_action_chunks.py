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
from pyquaternion import Quaternion


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
    parser.add_argument("--max_abs_curvature_1pm", type=float, default=0.2)
    parser.add_argument("--min_segment_distance", type=float, default=0.2)
    parser.add_argument("--skip_if_extreme_curvature", type=lambda x: str(x).lower() == "true", default=False)
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


def timestamps_to_seconds(timestamps: List[Optional[int]]) -> np.ndarray:
    values = np.asarray([0 if ts is None else ts for ts in timestamps], dtype=np.float64)
    if len(values) == 0:
        return values
    if np.nanmax(np.abs(values)) > 1e6:
        values = values * 1e-6
    return values


def stable_estimate_speed_curvature(
    points: np.ndarray,
    timestamps: List[Optional[int]],
    min_segment_distance: float = 0.2,
    max_abs_curvature_1pm: float = 0.2,
):
    """Estimate stable per-sample speed and curvature from ordered ego positions."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    velocities = np.zeros((n, 3), dtype=np.float32)
    curvatures = np.zeros(n, dtype=np.float32)
    clipped_mask = np.zeros(n, dtype=bool)
    stats = Counter()
    if n < 2:
        return velocities, curvatures, clipped_mask, stats

    xy = points[:, :2]
    deltas = xy[1:] - xy[:-1]
    ds = np.linalg.norm(deltas, axis=1)
    seconds = timestamps_to_seconds(timestamps)
    dt = np.diff(seconds)
    if len(dt) != len(ds) or not np.isfinite(dt).all():
        dt = np.full_like(ds, 0.5, dtype=np.float64)

    small_segment = ds < min_segment_distance
    bad_dt = dt <= 0
    stats["small_segment_count"] += int(np.count_nonzero(small_segment))
    stats["bad_dt_count"] += int(np.count_nonzero(bad_dt))

    valid_speed = (~bad_dt) & np.isfinite(ds) & np.isfinite(dt)
    speed_seg = np.zeros_like(ds, dtype=np.float64)
    speed_seg[valid_speed] = ds[valid_speed] / dt[valid_speed]
    segment_vel = np.zeros((len(ds), 3), dtype=np.float64)
    segment_vel[valid_speed, :2] = deltas[valid_speed] / dt[valid_speed, None]
    velocities[1:] = segment_vel.astype(np.float32)
    velocities[0] = velocities[1]

    valid_heading = (~small_segment) & np.isfinite(deltas).all(axis=1)
    headings = np.zeros(len(ds), dtype=np.float64)
    headings[valid_heading] = np.arctan2(deltas[valid_heading, 1], deltas[valid_heading, 0])
    headings = np.unwrap(headings)

    raw_curvatures = np.zeros(n, dtype=np.float64)
    for idx in range(1, n - 1):
        prev_seg = idx - 1
        next_seg = idx
        if small_segment[prev_seg] or small_segment[next_seg] or bad_dt[prev_seg] or bad_dt[next_seg]:
            raw_curvatures[idx] = 0.0
            continue
        arc_length = 0.5 * (ds[prev_seg] + ds[next_seg])
        if arc_length <= 1e-6 or not np.isfinite(arc_length):
            raw_curvatures[idx] = 0.0
            continue
        raw_curvatures[idx] = (headings[next_seg] - headings[prev_seg]) / arc_length

    if n > 2:
        raw_curvatures[0] = raw_curvatures[1]
        raw_curvatures[-1] = raw_curvatures[-2]

    nonfinite = ~np.isfinite(raw_curvatures)
    stats["nonfinite_curvature_count"] += int(np.count_nonzero(nonfinite))
    raw_curvatures[nonfinite] = 0.0

    clipped_mask = np.abs(raw_curvatures) > max_abs_curvature_1pm
    stats["curvature_clipped_count"] += int(np.count_nonzero(clipped_mask))
    curvatures = np.clip(raw_curvatures, -max_abs_curvature_1pm, max_abs_curvature_1pm).astype(np.float32)
    return velocities, curvatures, clipped_mask, stats


def compute_future_waypoints_local(
    current_ego_pose: Dict[str, Any],
    future_ego_poses: List[Dict[str, Any]],
) -> np.ndarray:
    """Transform future global ego positions into the current ego frame.

    nuScenes ego_pose rotation maps ego-local coordinates to global coordinates.
    Therefore R_current.T maps a global displacement into current ego-local
    coordinates, where +x is current ego forward and +y is current ego left.
    """
    current_translation = np.asarray(current_ego_pose["translation"][:3], dtype=np.float64)
    current_rotation = Quaternion(current_ego_pose["rotation"]).rotation_matrix
    waypoints = []
    for future_pose in future_ego_poses:
        future_translation = np.asarray(future_pose["translation"][:3], dtype=np.float64)
        rel_global = future_translation - current_translation
        rel_local = current_rotation.T @ rel_global
        waypoints.append(rel_local[:2])
    return np.asarray(waypoints, dtype=np.float32)


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
    future_waypoints_local: np.ndarray,
    curvature_clipped: bool = False,
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
            "curvature_clipped": bool(curvature_clipped),
            "max_abs_curvature_1pm": args.max_abs_curvature_1pm,
            "min_segment_distance": args.min_segment_distance,
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
            "future_waypoints_local": future_waypoints_local.astype(np.float32),
            "future_waypoints_schema": ["x_m_local", "y_m_local"],
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

    future_waypoints = np.asarray(record.get("target", {}).get("future_waypoints_local"), dtype=np.float32)
    if future_waypoints.shape != (future_steps, 2):
        return False, "invalid_future_waypoints_shape"
    if not np.isfinite(future_waypoints).all():
        return False, "invalid_future_waypoints_nonfinite"
    if record.get("target", {}).get("future_waypoints_schema") != ["x_m_local", "y_m_local"]:
        return False, "invalid_future_waypoints_schema"

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
    motion_stats = Counter()
    waypoint_stats = {
        "waypoint_nonfinite_count": 0,
        "waypoint_bad_shape_count": 0,
        "waypoint_max_abs_x": 0.0,
        "waypoint_max_abs_y": 0.0,
        "waypoint_max_distance": 0.0,
    }
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

        ego_points = np.asarray([pose["translation"][:3] for pose in ego_poses], dtype=np.float32)
        ego_velocities, ego_curvatures, clipped_mask, scene_motion_stats = stable_estimate_speed_curvature(
            ego_points,
            sample_timestamps,
            min_segment_distance=args.min_segment_distance,
            max_abs_curvature_1pm=args.max_abs_curvature_1pm,
        )
        motion_stats.update(scene_motion_stats)
        first_current_idx = args.history_steps - 1
        last_current_idx = scene_len - args.future_steps - 1

        for current_idx in range(first_current_idx, last_current_idx + 1):
            if args.max_samples > 0 and written >= args.max_samples:
                break
            total_candidates += 1

            hist_slice = slice(current_idx - args.history_steps + 1, current_idx + 1)
            fut_slice = slice(current_idx + 1, current_idx + 1 + args.future_steps)
            future_clipped = bool(np.any(clipped_mask[fut_slice]))
            if args.skip_if_extreme_curvature and future_clipped:
                skipped["skipped_by_extreme_curvature"] += 1
                continue
            future_poses = ego_poses[current_idx + 1 : current_idx + 1 + args.future_steps]
            if len(future_poses) != args.future_steps:
                skipped["insufficient_future_waypoints"] += 1
                continue
            future_waypoints_local = compute_future_waypoints_local(
                ego_poses[current_idx],
                future_poses,
            )
            if future_waypoints_local.shape != (args.future_steps, 2):
                waypoint_stats["waypoint_bad_shape_count"] += 1
                skipped["invalid_future_waypoints_shape"] += 1
                continue
            if not np.isfinite(future_waypoints_local).all():
                waypoint_stats["waypoint_nonfinite_count"] += int(
                    future_waypoints_local.size - np.count_nonzero(np.isfinite(future_waypoints_local))
                )
                skipped["invalid_future_waypoints_nonfinite"] += 1
                continue
            waypoint_distances = np.linalg.norm(future_waypoints_local, axis=1)
            waypoint_stats["waypoint_max_abs_x"] = max(
                waypoint_stats["waypoint_max_abs_x"],
                float(np.max(np.abs(future_waypoints_local[:, 0]))),
            )
            waypoint_stats["waypoint_max_abs_y"] = max(
                waypoint_stats["waypoint_max_abs_y"],
                float(np.max(np.abs(future_waypoints_local[:, 1]))),
            )
            waypoint_stats["waypoint_max_distance"] = max(
                waypoint_stats["waypoint_max_distance"],
                float(np.max(waypoint_distances)),
            )
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
                future_waypoints_local=future_waypoints_local,
                curvature_clipped=future_clipped,
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
    print(f"motion stats: {dict(motion_stats)}")
    print(f"waypoint stats: {waypoint_stats}")
    print(f"skipped count by reason: {dict(skipped)}")
    print(f"output path: {args.output_jsonl}")


def main():
    args = parse_args()
    build_gt_action_chunks(args)


if __name__ == "__main__":
    main()
