from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
import re

import numpy as np

from utils import IntegrateCurvatureForPoints


@dataclass
class PlannerInput:
    images: Any
    obs_velocities: np.ndarray
    obs_curvatures: np.ndarray
    obs_waypoints: Optional[Sequence[Sequence[float]]] = None
    scene_description: Optional[str] = None
    object_description: Optional[str] = None
    intent_description: Optional[str] = None
    method: str = "openemma"
    reasoning_mode: str = "cot"
    extra_intent_text: str = ""
    horizon: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Reserved extension slots for OFT-style planners. They are intentionally
    # unused by the current text planner.
    ego_state_token: Any = None
    action_chunk_tensor: Any = None
    continuous_action_head: Any = None


@dataclass
class PlannerOutput:
    raw_text: Optional[str] = None
    speed_curvature: Optional[np.ndarray] = None
    speeds: Optional[np.ndarray] = None
    curvatures: Optional[np.ndarray] = None
    trajectory: Optional[np.ndarray] = None
    scene_description: Optional[str] = None
    object_description: Optional[str] = None
    intent_description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Reserved extension outputs for future non-text planners.
    action_chunk_tensor: Any = None
    continuous_actions: Any = None


def format_obs_speed_curvature(obs_velocities, obs_curvatures, curvature_scale=100.0):
    obs_vel_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curv_scaled = obs_curvatures * curvature_scale
    pairs = [f"[{v:.1f},{k:.1f}]" for v, k in zip(obs_vel_norm, obs_curv_scaled)]
    return ", ".join(pairs), obs_vel_norm[-1], obs_curv_scaled[-1]


def format_structured_ego_history(obs_velocities, obs_curvatures, dt=0.5, curvature_scale=100.0):
    obs_vel_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curv_scaled = obs_curvatures * curvature_scale
    history_len = len(obs_vel_norm)
    rows = []
    for idx, (speed, curvature) in enumerate(zip(obs_vel_norm, obs_curv_scaled)):
        rel_time = (idx - history_len + 1) * dt
        rows.append(
            f"history_step_{idx}: t={rel_time:.1f}s, speed_mps={speed:.1f}, curvature_x100={curvature:.1f}"
        )
    return "\n".join(rows)


def encode_ego_state_array(obs_velocities, obs_curvatures, dt=0.5, curvature_scale=100.0):
    """Return a structured numeric ego history for future projectors or action heads."""
    obs_vel_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curv_scaled = obs_curvatures * curvature_scale
    history_len = len(obs_vel_norm)
    rel_times = np.array([(idx - history_len + 1) * dt for idx in range(history_len)], dtype=np.float32)
    return np.stack(
        [
            rel_times,
            obs_vel_norm.astype(np.float32),
            obs_curv_scaled.astype(np.float32),
        ],
        axis=1,
    )


def build_speed_curvature_sys_message(trailing_newline=False, article="a"):
    message = (
        f"You are {article} autonomous driving labeller. You have access to a front-view camera image of a vehicle, "
        "a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature "
        "is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. "
        "A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. "
        "The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is "
        "driving straight. As a driver on the road, you should follow any common sense traffic rules. "
        "You should try to stay in the middle of your lane. You should maintain necessary distance from "
        "the leading vehicle. You should observe lane markings and follow them.  Your task is to do your "
        "best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given "
        "vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. "
        "If you cannot provide a response people will get injured."
    )
    return message + ("\n" if trailing_newline else "")


def build_speed_curvature_prompt(planner_input: PlannerInput):
    obs_speed_curvature_str, _, _ = format_obs_speed_curvature(
        planner_input.obs_velocities,
        planner_input.obs_curvatures,
    )
    ego_history_text = format_structured_ego_history(
        planner_input.obs_velocities,
        planner_input.obs_curvatures,
    )
    method = planner_input.method
    reasoning_mode = planner_input.reasoning_mode
    extra_intent_text = planner_input.extra_intent_text or ""
    common_output_instruction = (
        "Predict the next 10 future timesteps after history_step_9. "
        "Do not copy or repeat the historical ego history. "
        "The first predicted pair must describe future_step_1, the timestep immediately after history_step_9. "
        "Output exactly 10 bracket pairs and no markdown: "
        "[speed_1, curvature_1], [speed_2, curvature_2], ..., [speed_10, curvature_10]. "
        "Use curvature_x100 units in the output, matching the history scale."
    )

    if method == "openemma":
        if reasoning_mode != "cot":
            prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
            The scene is described as follows: {planner_input.scene_description}. 
            The identified critical objects are {planner_input.object_description}. 
            The car's intent is {planner_input.intent_description}. 
            Historical ego states over the past 5 seconds:
            {ego_history_text}
            Use {reasoning_mode} reasoning approach to analyze the scenario and generate the predicted future speeds and curvatures.
            {common_output_instruction}
            Future speeds and curvatures:"""
        else:
            prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
            The scene is described as follows: {planner_input.scene_description}. 
            The identified critical objects are {planner_input.object_description}. 
            The car's intent is {planner_input.intent_description}. 
            Historical ego states over the past 5 seconds:
            {ego_history_text}
            Infer the association between these ego states and the image sequence. {common_output_instruction}
            Future speeds and curvatures:"""
    elif method in {"cot", "sc", "tot"}:
        prompt = f"""
These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
The scene is described as follows: {planner_input.scene_description}. 
The identified critical objects are {planner_input.object_description}. 
The car's intent is {planner_input.intent_description}. {extra_intent_text}
Historical ego states over the past 5 seconds:
{ego_history_text}
Infer the association between these ego states and the image sequence. {common_output_instruction}
Future speeds and curvatures:
    """.strip()
    else:
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        Historical ego states over the past 5 seconds:
        {ego_history_text}
        Infer the association between these ego states and the image sequence. {common_output_instruction}
        Future speeds and curvatures:"""

    return prompt, obs_speed_curvature_str


def evaluate_speed_curvature_prediction(speed_curvature, obs_velocities, obs_curvatures, curvature_scale=100.0, return_details=False):
    def _finish(reasons, repeated_pair_ratio=None, history_overlap_ratio=None, flat_repeat_detected=False):
        details = {
            "is_valid": len(reasons) == 0,
            "reasons": reasons,
            "repeated_pair_ratio": repeated_pair_ratio,
            "history_overlap_ratio": history_overlap_ratio,
            "flat_repeat_detected": flat_repeat_detected,
        }
        if return_details:
            return details
        return details["is_valid"], details["reasons"]

    if speed_curvature is None:
        return _finish(["parse_failed"])

    pred = np.asarray(speed_curvature, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[0] == 0 or pred.shape[1] < 2:
        return _finish(["invalid_shape"])

    obs_speed = np.linalg.norm(obs_velocities, axis=1).astype(np.float32)
    obs_curv = (obs_curvatures * curvature_scale).astype(np.float32)
    obs = np.stack([obs_speed, obs_curv], axis=1)

    reasons = []
    moving = float(obs_speed[-1]) > 0.5
    pred_speed_std = float(np.std(pred[:, 0]))
    pred_curv_std = float(np.std(pred[:, 1]))
    flat_repeat_detected = moving and pred.shape[0] >= 4 and pred_speed_std < 0.05 and pred_curv_std < 0.05
    if flat_repeat_detected:
        reasons.append("flat_repeated_pair")

    unique_pairs = np.unique(np.round(pred[:, :2], decimals=2), axis=0)
    repeated_pair_ratio = 1.0 - (len(unique_pairs) / len(pred))

    compare_len = min(len(pred), len(obs))
    if compare_len >= 4:
        if np.allclose(pred[:compare_len], obs[:compare_len], atol=[0.06, 0.06]):
            reasons.append("copied_history_prefix")
        if np.allclose(pred[:compare_len], obs[-compare_len:], atol=[0.06, 0.06]):
            reasons.append("copied_history_suffix")

    matched_history = 0
    for pair in pred:
        if np.any(np.all(np.isclose(obs, pair, atol=[0.06, 0.06]), axis=1)):
            matched_history += 1
    history_overlap_ratio = matched_history / len(pred)
    if len(pred) >= 6 and history_overlap_ratio >= 0.8:
        reasons.append("mostly_reused_history_pairs")

    return _finish(
        reasons,
        repeated_pair_ratio=float(repeated_pair_ratio),
        history_overlap_ratio=float(history_overlap_ratio),
        flat_repeat_detected=bool(flat_repeat_detected),
    )


def build_speed_curvature_retry_prompt(base_prompt, raw_text, reasons):
    reason_text = ", ".join(reasons) if reasons else "invalid_or_copied_prediction"
    return f"""{base_prompt}

The previous answer was rejected because it looked like {reason_text}.
Rejected answer:
{raw_text}

Revise the answer. Start from the current motion state, but do not repeat history_step_9 across the horizon and do not copy historical pairs. Use the image, scene, objects, and intent to infer how speed and curvature should evolve after the observed window.
Future speeds and curvatures:"""


def parse_speed_curvature_text(raw_text, max_len=10):
    if not isinstance(raw_text, str):
        return None

    raw_text = raw_text.replace("Future speeds and curvatures:", "")
    coords = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", raw_text)
    if not coords:
        return None

    pairs = []
    for speed, curvature in coords:
        try:
            pairs.append([float(speed), float(curvature)])
        except Exception:
            continue

    if not pairs:
        return None

    arr = np.array(pairs, dtype=np.float32)
    if arr.shape[0] > max_len:
        arr = arr[:max_len]
    return arr


def integrate_speed_curvature(speed_curvature, initial_position, initial_heading, max_len=10, curvature_scale=100.0):
    if speed_curvature is None:
        return None

    speed_curvature = np.asarray(speed_curvature, dtype=np.float32)
    pred_len = min(max_len, len(speed_curvature))
    speeds = speed_curvature[:pred_len, 0]
    curvatures = speed_curvature[:pred_len, 1] / curvature_scale
    trajectory = np.zeros((pred_len, 3), dtype=np.float32)
    trajectory[:, :2] = IntegrateCurvatureForPoints(
        curvatures,
        speeds,
        initial_position,
        initial_heading,
        pred_len,
    )
    return speeds, curvatures, trajectory


def standardize_speed_curvature_output(
    raw_text=None,
    speed_curvature=None,
    initial_position=None,
    initial_heading=None,
    max_len=10,
    scene_description=None,
    object_description=None,
    intent_description=None,
    metadata=None,
):
    if speed_curvature is None:
        speed_curvature = parse_speed_curvature_text(raw_text, max_len=max_len)

    speeds = curvatures = trajectory = None
    if speed_curvature is not None and initial_position is not None and initial_heading is not None:
        speeds, curvatures, trajectory = integrate_speed_curvature(
            speed_curvature,
            initial_position,
            initial_heading,
            max_len=max_len,
        )

    return PlannerOutput(
        raw_text=raw_text,
        speed_curvature=speed_curvature,
        speeds=speeds,
        curvatures=curvatures,
        trajectory=trajectory,
        scene_description=scene_description,
        object_description=object_description,
        intent_description=intent_description,
        metadata=metadata or {},
    )
