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

from geometry_token import ORACLE_GEOMETRY_DESCRIPTOR_DIM, build_oracle_geometry_descriptor_from_waypoints


def assert_finite(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf.")


def test_straight_descriptor():
    x = torch.arange(1, 11, dtype=torch.float32)
    waypoints = torch.stack([x, torch.zeros_like(x)], dim=-1).unsqueeze(0)
    descriptor = build_oracle_geometry_descriptor_from_waypoints(waypoints, dt=1.0)
    if tuple(descriptor.shape) != (1, ORACLE_GEOMETRY_DESCRIPTOR_DIM):
        raise AssertionError(f"unexpected descriptor shape: {tuple(descriptor.shape)}")
    assert_finite("straight descriptor", descriptor)
    final_y = descriptor[0, 1]
    heading_change_total = descriptor[0, 9]
    if abs(float(final_y)) > 1e-6:
        raise AssertionError(f"straight final_y should be 0, got {float(final_y)}")
    if abs(float(heading_change_total)) > 1e-6:
        raise AssertionError(f"straight heading_change_total should be 0, got {float(heading_change_total)}")


def test_curve_descriptor():
    x = torch.arange(1, 11, dtype=torch.float32)
    y = 0.1 * x * x
    waypoints = torch.stack([x, y], dim=-1).unsqueeze(0)
    descriptor = build_oracle_geometry_descriptor_from_waypoints(waypoints, dt=0.5)
    if tuple(descriptor.shape) != (1, ORACLE_GEOMETRY_DESCRIPTOR_DIM):
        raise AssertionError(f"unexpected descriptor shape: {tuple(descriptor.shape)}")
    assert_finite("curve descriptor", descriptor)
    max_abs_y = descriptor[0, 4]
    heading_change_total = descriptor[0, 9]
    if float(max_abs_y) <= 0.0:
        raise AssertionError("curve max_abs_y should be positive.")
    if abs(float(heading_change_total)) <= 0.0:
        raise AssertionError("curve heading_change_total should be non-zero.")


def main():
    test_straight_descriptor()
    test_curve_descriptor()
    print("[GeometryTokenTest] all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
