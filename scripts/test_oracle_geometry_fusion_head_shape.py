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

from action_head import OracleGeometryFusionHead
from geometry_token import ORACLE_GEOMETRY_DESCRIPTOR_DIM


def main():
    batch_size = 2
    head = OracleGeometryFusionHead(geometry_descriptor_dim=ORACLE_GEOMETRY_DESCRIPTOR_DIM)
    planning_hidden = torch.randn(batch_size, 3584)
    ego_history = torch.randn(batch_size, 10, 3)
    geometry_descriptor = torch.randn(batch_size, ORACLE_GEOMETRY_DESCRIPTOR_DIM)
    output = head(planning_hidden, ego_history, geometry_descriptor)
    if tuple(output.shape) != (batch_size, 10, 2):
        raise AssertionError(f"unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("output contains NaN or Inf.")
    print("[OracleGeometryFusionHeadShapeTest] all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
