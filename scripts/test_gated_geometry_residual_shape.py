#!/usr/bin/env python3
import os
import sys

try:
    import torch
except ModuleNotFoundError:
    torch = None


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

def main() -> int:
    if torch is None:
        print("[GatedGeometryResidualShapeTest] skipped: torch is not installed")
        return 0

    from action_head import (
        ChannelWiseGatedResidualHead,
        FusionActionHead,
        GatedGeometryResidualFusionHead,
        GeometryDescriptorHead,
    )

    torch.manual_seed(0)
    batch_size = 2
    qwen_hidden_dim = 32
    history_steps = 10
    ego_dim = 3
    chunk_size = 10
    action_dim = 2
    descriptor_dim = 6

    fusion_head = FusionActionHead(
        qwen_hidden_dim=qwen_hidden_dim,
        history_steps=history_steps,
        ego_dim=ego_dim,
        qwen_embed_dim=16,
        ego_embed_dim=8,
        fusion_hidden_size=32,
        chunk_size=chunk_size,
        action_dim=action_dim,
        dropout=0.0,
    )
    geometry_descriptor_head = GeometryDescriptorHead(
        qwen_hidden_dim=qwen_hidden_dim,
        descriptor_dim=descriptor_dim,
        hidden_size=32,
        dropout=0.0,
    )
    gated_residual_head = ChannelWiseGatedResidualHead(
        qwen_hidden_dim=qwen_hidden_dim,
        descriptor_dim=descriptor_dim,
        hidden_size=32,
        chunk_size=chunk_size,
        action_dim=action_dim,
        dropout=0.0,
    )
    head = GatedGeometryResidualFusionHead(
        frozen_fusion_head=fusion_head,
        geometry_descriptor_head=geometry_descriptor_head,
        gated_residual_head=gated_residual_head,
        residual_scale=0.1,
    )

    planning_hidden = torch.randn(batch_size, qwen_hidden_dim)
    ego_history = torch.randn(batch_size, history_steps, ego_dim)
    outputs = head(planning_hidden, ego_history)

    assert outputs["base_action"].shape == (2, 10, 2), outputs["base_action"].shape
    assert outputs["action_chunk"].shape == (2, 10, 2), outputs["action_chunk"].shape
    assert outputs["geometry_descriptor"].shape == (2, 6), outputs["geometry_descriptor"].shape
    assert outputs["gate"].shape == (2, 10, 2), outputs["gate"].shape
    assert torch.all(outputs["gate"] >= 0.0)
    assert torch.all(outputs["gate"] <= 1.0)
    print("[GatedGeometryResidualShapeTest] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
