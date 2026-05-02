import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANNER_DIR = os.path.join(REPO_ROOT, "openemma", "planner")
if PLANNER_DIR not in sys.path:
    sys.path.insert(0, PLANNER_DIR)

from action_head import EgoOnlyActionHead


def main():
    head = EgoOnlyActionHead(history_steps=10, ego_dim=3, chunk_size=10, action_dim=2)
    x = torch.randn(2, 10, 3)
    y = head(x)
    assert y.shape == (2, 10, 2)
    print("ego action head shape test passed:", y.shape)


if __name__ == "__main__":
    main()
