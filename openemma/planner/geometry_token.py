import math

import torch


ORACLE_GEOMETRY_PURPOSE = "upper_bound_diagnosis_not_deployment"
ORACLE_GEOMETRY_DESCRIPTOR_DIM = 16


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def build_oracle_geometry_descriptor_from_waypoints(
    waypoints: torch.Tensor,
    dt: float = 0.5,
    eps: float = 1e-6,
    max_abs_curvature_1pm: float = 0.5,
) -> torch.Tensor:
    """Build an oracle geometry token from GT future waypoints for upper-bound diagnosis only."""
    if waypoints.ndim != 3 or waypoints.shape[-1] != 2:
        raise ValueError("waypoints must have shape [B, future_steps, 2].")
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
    heading = torch.atan2(dy, dx)
    raw_delta_heading = heading[:, 1:] - heading[:, :-1]
    delta_heading = _wrap_to_pi(raw_delta_heading)
    curvature_abs = torch.abs(delta_heading / segment_length[:, 1:]).clamp_max(max_abs_curvature_1pm)

    final_x = waypoints[:, -1, 0]
    final_y = waypoints[:, -1, 1]
    mean_x = waypoints[..., 0].mean(dim=1)
    mean_y = waypoints[..., 1].mean(dim=1)
    max_abs_y = torch.abs(waypoints[..., 1]).amax(dim=1)
    path_length = segment_length.sum(dim=1)
    displacement = torch.linalg.norm(waypoints[:, -1], dim=-1)
    final_heading = heading[:, -1]
    mean_heading = heading.mean(dim=1)
    heading_change_total = _wrap_to_pi(final_heading - heading[:, 0])
    mean_abs_heading_change = torch.abs(delta_heading).mean(dim=1)
    max_abs_heading_change = torch.abs(delta_heading).amax(dim=1)
    mean_lateral_step = dy.mean(dim=1)
    max_abs_lateral_step = torch.abs(dy).amax(dim=1)
    approx_mean_curvature = curvature_abs.mean(dim=1)
    approx_max_curvature = curvature_abs.amax(dim=1)

    descriptor = torch.stack(
        [
            final_x,
            final_y,
            mean_x,
            mean_y,
            max_abs_y,
            path_length,
            displacement,
            final_heading,
            mean_heading,
            heading_change_total,
            mean_abs_heading_change,
            max_abs_heading_change,
            mean_lateral_step,
            max_abs_lateral_step,
            approx_mean_curvature,
            approx_max_curvature,
        ],
        dim=-1,
    )
    return torch.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0).float()
