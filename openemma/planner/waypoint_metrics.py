import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _validate_waypoints(pred_waypoints: torch.Tensor, target_waypoints: torch.Tensor):
    if pred_waypoints.shape != target_waypoints.shape:
        raise ValueError(
            f"pred_waypoints and target_waypoints must have the same shape, got "
            f"{tuple(pred_waypoints.shape)} and {tuple(target_waypoints.shape)}."
        )
    if pred_waypoints.ndim != 3 or pred_waypoints.shape[-1] != 2:
        raise ValueError("waypoints must have shape [B, future_steps, 2].")


def waypoint_l1_loss(pred_waypoints: torch.Tensor, target_waypoints: torch.Tensor) -> torch.Tensor:
    _validate_waypoints(pred_waypoints, target_waypoints)
    return F.l1_loss(pred_waypoints, target_waypoints)


def waypoint_ade(pred_waypoints: torch.Tensor, target_waypoints: torch.Tensor) -> torch.Tensor:
    _validate_waypoints(pred_waypoints, target_waypoints)
    return torch.linalg.norm(pred_waypoints - target_waypoints, dim=-1).mean()


def waypoint_fde(pred_waypoints: torch.Tensor, target_waypoints: torch.Tensor) -> torch.Tensor:
    _validate_waypoints(pred_waypoints, target_waypoints)
    return torch.linalg.norm(pred_waypoints[:, -1] - target_waypoints[:, -1], dim=-1).mean()


def waypoint_longitudinal_lateral_mae(
    pred_waypoints: torch.Tensor,
    target_waypoints: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_waypoints(pred_waypoints, target_waypoints)
    abs_error = torch.abs(pred_waypoints - target_waypoints)
    return abs_error[..., 0].mean(), abs_error[..., 1].mean()


def _validate_single_waypoints(waypoints: torch.Tensor):
    if waypoints.ndim != 3 or waypoints.shape[-1] != 2:
        raise ValueError("waypoints must have shape [B, future_steps, 2].")


def derive_action_from_waypoints(
    waypoints: torch.Tensor,
    dt: float = 0.5,
    curvature_scale: float = 100.0,
    eps: float = 1e-6,
    max_abs_curvature_1pm: float = 0.2,
) -> torch.Tensor:
    """Derive stable [speed_mps, curvature_x100] chunks from ego-local waypoints."""
    _validate_single_waypoints(waypoints)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    batch_size = waypoints.shape[0]
    p0 = torch.zeros(batch_size, 1, 2, device=waypoints.device, dtype=waypoints.dtype)
    points = torch.cat([p0, waypoints], dim=1)
    delta = points[:, 1:] - points[:, :-1]
    dx = delta[..., 0]
    dy = delta[..., 1]

    segment_length = torch.sqrt(dx * dx + dy * dy).clamp_min(eps)
    speed = segment_length / dt
    heading = torch.atan2(dy, dx)

    raw_delta_heading = heading[:, 1:] - heading[:, :-1]
    two_pi = 2.0 * math.pi
    wrapped_delta_heading = torch.remainder(raw_delta_heading + math.pi, two_pi) - math.pi
    delta_heading = torch.zeros_like(heading)
    delta_heading[:, 1:] = wrapped_delta_heading

    curvature_1pm = delta_heading / segment_length
    curvature_1pm = torch.clamp(
        curvature_1pm,
        min=-max_abs_curvature_1pm,
        max=max_abs_curvature_1pm,
    )
    curvature_x100 = curvature_1pm * curvature_scale
    return torch.stack([speed, curvature_x100], dim=-1)


def waypoint_action_consistency_loss(
    pred_action: torch.Tensor,
    pred_waypoints: torch.Tensor,
    dt: float = 0.5,
    curvature_scale: float = 100.0,
    max_abs_curvature_1pm: float = 0.2,
    speed_weight: float = 1.0,
    curvature_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if pred_action.ndim != 3 or pred_action.shape[-1] != 2:
        raise ValueError("pred_action must have shape [B, future_steps, 2].")
    _validate_single_waypoints(pred_waypoints)
    if pred_action.shape[:2] != pred_waypoints.shape[:2]:
        raise ValueError(
            f"pred_action and pred_waypoints must share [B, T], got "
            f"{tuple(pred_action.shape[:2])} and {tuple(pred_waypoints.shape[:2])}."
        )

    derived_action = derive_action_from_waypoints(
        pred_waypoints,
        dt=dt,
        curvature_scale=curvature_scale,
        max_abs_curvature_1pm=max_abs_curvature_1pm,
    )
    abs_error = torch.abs(pred_action - derived_action)
    speed_l1 = abs_error[..., 0].mean()
    curvature_l1 = abs_error[..., 1].mean()
    weights = pred_action.new_tensor([speed_weight, curvature_weight]).view(1, 1, 2)
    loss = (abs_error * weights).mean()
    return {
        "loss": loss,
        "derived_action": derived_action,
        "derived_speed_mae_vs_pred_mps": speed_l1,
        "derived_curvature_mae_x100_vs_pred": curvature_l1,
    }
