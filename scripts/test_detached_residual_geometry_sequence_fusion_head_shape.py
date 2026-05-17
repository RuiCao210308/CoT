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

from action_head import DetachedResidualGeometrySequenceFusionHead


def main():
    batch_size = 2
    head = DetachedResidualGeometrySequenceFusionHead()
    planning_hidden = torch.randn(batch_size, 3584)
    ego_history = torch.randn(batch_size, 10, 3)
    geometry_sequence = torch.randn(batch_size, 10, 2)
    output = head(planning_hidden, ego_history, geometry_sequence)
    if tuple(output.shape) != (batch_size, 10, 2):
        raise AssertionError(f"unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("output contains NaN or Inf.")
    print("[DetachedResidualGeometrySequenceFusionHeadShapeTest] all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
