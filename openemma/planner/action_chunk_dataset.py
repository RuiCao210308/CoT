import json
import os
from datetime import datetime

import numpy as np

from openemma.planner.base_planner import encode_ego_state_array


def numpy_to_list_safe(x):
    """Convert numpy arrays/scalars to JSON-serializable Python objects."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64, np.float16)):
        return float(x)
    if isinstance(x, (np.int32, np.int64, np.int16, np.int8)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {k: numpy_to_list_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [numpy_to_list_safe(v) for v in x]
    return x


def validate_action_chunk_record(record):
    """Validate required action-chunk fields and basic [T, C] shapes."""
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


def append_action_chunk_jsonl(record, jsonl_path):
    """Append one validated action chunk record to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
    safe_record = numpy_to_list_safe(record)
    validate_action_chunk_record(safe_record)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe_record, ensure_ascii=False))
        f.write("\n")


def save_action_chunk_record(record, output_dir, scene_name=None):
    """Save one action chunk record as a standalone JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    safe_record = numpy_to_list_safe(record)
    validate_action_chunk_record(safe_record)
    metadata = safe_record.get("metadata", {})
    scene = scene_name or metadata.get("scene_name") or "scene"
    frame_idx = metadata.get("frame_idx", "unknown")
    filename = f"{scene}_{frame_idx}_action_chunk.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_record, f, ensure_ascii=False, indent=2)
    return path


def make_action_chunk_record(
    scene_name,
    frame_idx,
    obs_velocities,
    obs_curvatures,
    future_speed_curvature_gt,
    predicted_speed_curvature,
    raw_qwen_text=None,
    planner_guard=None,
    metrics=None,
    method=None,
    model_path=None,
    scene_index=None,
    sample_token=None,
    timestamp=None,
    retry_used=False,
    retry_reason=None,
    extra_metadata=None,
):
    """Build one JSON-ready record with ego history [T,3] and action chunks [10,2]."""
    guard = planner_guard or {}
    record = {
        "metadata": {
            "scene_name": scene_name,
            "scene_index": scene_index,
            "frame_idx": frame_idx,
            "sample_token": sample_token,
            "timestamp": timestamp,
            "method": method,
            "model_path": model_path,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "input": {
            "ego_history_array": encode_ego_state_array(obs_velocities, obs_curvatures),
            "ego_history_schema": ["relative_time", "speed_mps", "curvature_x100"],
        },
        "target": {
            "future_action_gt": future_speed_curvature_gt,
            "future_action_schema": ["speed_mps", "curvature_1pm"],
        },
        "prediction": {
            "qwen_predicted_action": predicted_speed_curvature,
            "raw_qwen_text": raw_qwen_text,
            "retry_used": bool(retry_used),
            "retry_reason": retry_reason,
            "planner_guard": {
                "is_valid": guard.get("is_valid"),
                "reasons": guard.get("reasons", []),
                "repeated_pair_ratio": guard.get("repeated_pair_ratio"),
                "history_overlap_ratio": guard.get("history_overlap_ratio"),
                "flat_repeat_detected": guard.get("flat_repeat_detected"),
            },
        },
        "metrics": metrics or {},
    }
    if extra_metadata:
        record["metadata"].update(extra_metadata)
    return numpy_to_list_safe(record)
