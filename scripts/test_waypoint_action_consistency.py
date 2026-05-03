#!/usr/bin/env python3
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from waypoint_metrics import derive_action_from_waypoints, waypoint_action_consistency_loss


def assert_finite(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf.")


def test_straight_constant_speed():
    x = torch.arange(1, 11, dtype=torch.float32)
    y = torch.zeros_like(x)
    waypoints = torch.stack([x, y], dim=-1).unsqueeze(0)
    derived = derive_action_from_waypoints(waypoints, dt=1.0)
    if tuple(derived.shape) != (1, 10, 2):
        raise AssertionError(f"unexpected derived shape: {tuple(derived.shape)}")
    if not torch.allclose(derived[..., 0], torch.ones(1, 10), atol=1e-5):
        raise AssertionError(f"straight-line speed mismatch: {derived[..., 0]}")
    if not torch.allclose(derived[..., 1], torch.zeros(1, 10), atol=1e-5):
        raise AssertionError(f"straight-line curvature mismatch: {derived[..., 1]}")
    assert_finite("straight derived action", derived)


def test_simple_curve():
    x = torch.arange(1, 11, dtype=torch.float32)
    y = 0.1 * x * x
    waypoints = torch.stack([x, y], dim=-1).unsqueeze(0)
    derived = derive_action_from_waypoints(waypoints, dt=0.5)
    if tuple(derived.shape) != (1, 10, 2):
        raise AssertionError(f"unexpected derived shape: {tuple(derived.shape)}")
    assert_finite("curve derived action", derived)
    if not torch.any(torch.abs(derived[..., 1]) > 0.0):
        raise AssertionError("curve curvature should contain non-zero values.")


def test_consistency_loss():
    waypoints = torch.tensor(
        [
            [
                [1.0, 0.0],
                [2.0, 0.1],
                [3.0, 0.3],
                [4.0, 0.6],
                [5.0, 1.0],
                [6.0, 1.5],
                [7.0, 2.1],
                [8.0, 2.8],
                [9.0, 3.6],
                [10.0, 4.5],
            ]
        ],
        dtype=torch.float32,
    )
    pred_action = derive_action_from_waypoints(waypoints, dt=0.5)
    result = waypoint_action_consistency_loss(pred_action, waypoints, dt=0.5)
    if result["loss"].ndim != 0:
        raise AssertionError(f"loss should be scalar, got shape {tuple(result['loss'].shape)}")
    if tuple(result["derived_action"].shape) != (1, 10, 2):
        raise AssertionError(f"unexpected derived_action shape: {tuple(result['derived_action'].shape)}")
    assert_finite("consistency loss", result["loss"])
    assert_finite("consistency derived_action", result["derived_action"])
    assert_finite("derived speed mae", result["derived_speed_mae_vs_pred_mps"])
    assert_finite("derived curvature mae", result["derived_curvature_mae_x100_vs_pred"])


def main():
    test_straight_constant_speed()
    test_simple_curve()
    test_consistency_loss()
    print("[WaypointActionConsistencyTest] all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
