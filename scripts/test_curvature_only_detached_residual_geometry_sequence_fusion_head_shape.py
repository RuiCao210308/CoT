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

from action_head import CurvatureOnlyDetachedResidualGeometrySequenceFusionHead


def main():
    batch_size = 2
    head = CurvatureOnlyDetachedResidualGeometrySequenceFusionHead()
    planning_hidden = torch.randn(batch_size, 3584)
    ego_history = torch.randn(batch_size, 10, 3)
    geometry_sequence = torch.randn(batch_size, 10, 2)
    outputs = head(planning_hidden, ego_history, geometry_sequence)

    expected_shapes = {
        "action_chunk": (batch_size, 10, 2),
        "base_action": (batch_size, 10, 2),
        "residual_curvature": (batch_size, 10, 1),
    }
    for name, shape in expected_shapes.items():
        value = outputs.get(name)
        if value is None:
            raise AssertionError(f"missing output: {name}")
        if tuple(value.shape) != shape:
            raise AssertionError(f"unexpected {name} shape: {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise AssertionError(f"{name} contains NaN or Inf.")

    if not torch.allclose(outputs["action_chunk"][..., 0], outputs["base_action"][..., 0]):
        raise AssertionError("speed channel must match base_action speed exactly.")
    print("[CurvatureOnlyDetachedResidualGeometrySequenceFusionHeadShapeTest] all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
