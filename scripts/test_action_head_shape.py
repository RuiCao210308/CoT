import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from openemma.planner.action_head import ContinuousActionHead, action_chunk_l1_loss


def main():
    head = ContinuousActionHead(hidden_dim=3584, chunk_size=10, action_dim=2)
    x = torch.randn(2, 3584)
    y = head(x)
    assert y.shape == (2, 10, 2)

    target = torch.randn(2, 10, 2)
    loss = action_chunk_l1_loss(y, target)
    assert loss.ndim == 0
    print("action head shape test passed:", y.shape, "loss:", float(loss))


if __name__ == "__main__":
    main()
