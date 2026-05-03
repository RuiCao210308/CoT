from typing import Tuple

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
