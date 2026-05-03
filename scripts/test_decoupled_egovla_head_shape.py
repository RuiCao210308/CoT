import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from openemma.planner.action_head import DecoupledEgoVLAActionHead, action_chunk_l1_loss


def main():
    head = DecoupledEgoVLAActionHead(
        qwen_hidden_dim=3584,
        history_steps=10,
        ego_dim=3,
        chunk_size=10,
        action_dim=2,
    )
    planning_hidden = torch.randn(2, 3584)
    ego_history_array = torch.randn(2, 10, 3)
    pred = head(planning_hidden, ego_history_array)
    assert pred.shape == (2, 10, 2)

    target = torch.randn(2, 10, 2)
    loss = action_chunk_l1_loss(pred, target)
    assert loss.ndim == 0
    print("decoupled egovla head shape test passed:", pred.shape, "loss:", float(loss))


if __name__ == "__main__":
    main()
