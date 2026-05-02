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
    method = planner_input.method
    reasoning_mode = planner_input.reasoning_mode
    extra_intent_text = planner_input.extra_intent_text or ""

    if method == "openemma":
        if reasoning_mode != "cot":
            prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
            The scene is described as follows: {planner_input.scene_description}. 
            The identified critical objects are {planner_input.object_description}. 
            The car's intent is {planner_input.intent_description}. 
            The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
            Use {reasoning_mode} reasoning approach to analyze the scenario and generate the predicted future speeds and curvatures.
            Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
        else:
            prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
            The scene is described as follows: {planner_input.scene_description}. 
            The identified critical objects are {planner_input.object_description}. 
            The car's intent is {planner_input.intent_description}. 
            The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
            Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    elif method in {"cot", "sc", "tot"}:
        prompt = f"""
These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
The scene is described as follows: {planner_input.scene_description}. 
The identified critical objects are {planner_input.object_description}. 
The car's intent is {planner_input.intent_description}. {extra_intent_text}
The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format
[speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. 
Write the raw text not markdown or latex. Future speeds and curvatures:
    """.strip()
    else:
        prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
        The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
        Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""

    return prompt, obs_speed_curvature_str


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
