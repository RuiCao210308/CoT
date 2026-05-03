import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from openemma.planner.action_head import WaypointAuxFusionHead, action_chunk_l1_loss
from openemma.planner.waypoint_metrics import waypoint_ade, waypoint_fde


def main():
    head = WaypointAuxFusionHead(
        qwen_hidden_dim=3584,
        history_steps=10,
        ego_dim=3,
        chunk_size=10,
        action_dim=2,
        waypoint_dim=2,
    )
    planning_hidden = torch.randn(2, 3584)
    ego_history_array = torch.randn(2, 10, 3)
    outputs = head(planning_hidden, ego_history_array)
    assert outputs["action_chunk"].shape == (2, 10, 2)
    assert outputs["waypoints"].shape == (2, 10, 2)

    target_action = torch.randn(2, 10, 2)
    target_waypoints = torch.randn(2, 10, 2)
    action_loss = action_chunk_l1_loss(outputs["action_chunk"], target_action)
    ade = waypoint_ade(outputs["waypoints"], target_waypoints)
    fde = waypoint_fde(outputs["waypoints"], target_waypoints)
    assert action_loss.ndim == 0
    assert ade.ndim == 0
    assert fde.ndim == 0
    print(
        "waypoint aux fusion head shape test passed:",
        outputs["action_chunk"].shape,
        outputs["waypoints"].shape,
        "action_loss:",
        float(action_loss),
        "ade:",
        float(ade),
        "fde:",
        float(fde),
    )


if __name__ == "__main__":
    main()
