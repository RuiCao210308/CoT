import torch
from torch import nn
import torch.nn.functional as F


class ContinuousActionHead(nn.Module):
    """Lightweight MLP head for OFT-style parallel action chunk regression."""

    def __init__(
        self,
        hidden_dim: int = 3584,
        chunk_size: int = 10,
        action_dim: int = 2,
        hidden_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, chunk_size * action_dim),
        )

    def forward(self, planning_hidden: torch.Tensor) -> torch.Tensor:
        if planning_hidden.ndim != 2:
            raise ValueError("planning_hidden must have shape [B, hidden_dim].")
        if planning_hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"planning_hidden last dim must be {self.hidden_dim}, got {planning_hidden.shape[-1]}."
            )
        actions = self.net(planning_hidden)
        return actions.view(-1, self.chunk_size, self.action_dim)


def build_continuous_action_head(
    hidden_dim: int = 3584,
    chunk_size: int = 10,
    action_dim: int = 2,
    hidden_size: int = 1024,
    dropout: float = 0.1,
) -> ContinuousActionHead:
    return ContinuousActionHead(
        hidden_dim=hidden_dim,
        chunk_size=chunk_size,
        action_dim=action_dim,
        hidden_size=hidden_size,
        dropout=dropout,
    )


def action_chunk_l1_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    speed_weight: float = 1.0,
    curvature_weight: float = 1.0,
) -> torch.Tensor:
    if pred_actions.shape != target_actions.shape:
        raise ValueError(
            f"pred_actions and target_actions must have the same shape, got "
            f"{tuple(pred_actions.shape)} and {tuple(target_actions.shape)}."
        )
    if pred_actions.ndim != 3 or pred_actions.shape[-1] != 2:
        raise ValueError("actions must have shape [B, chunk_size, 2].")

    weights = pred_actions.new_tensor([speed_weight, curvature_weight]).view(1, 1, 2)
    return (F.l1_loss(pred_actions, target_actions, reduction="none") * weights).mean()


def normalize_action_chunk(
    action_chunk: torch.Tensor,
    speed_scale: float = 1.0,
    curvature_scale: float = 100.0,
) -> torch.Tensor:
    """Normalize [speed_mps, curvature_1pm] action chunks without changing schema."""
    return _scale_action_chunk(action_chunk, speed_scale, curvature_scale, inverse=False)


def denormalize_action_chunk(
    action_chunk: torch.Tensor,
    speed_scale: float = 1.0,
    curvature_scale: float = 100.0,
) -> torch.Tensor:
    """Restore normalized chunks back to [speed_mps, curvature_1pm] units."""
    return _scale_action_chunk(action_chunk, speed_scale, curvature_scale, inverse=True)


def _scale_action_chunk(
    action_chunk: torch.Tensor,
    speed_scale: float,
    curvature_scale: float,
    inverse: bool,
) -> torch.Tensor:
    if action_chunk.ndim < 1 or action_chunk.shape[-1] != 2:
        raise ValueError("action_chunk must have last dimension [speed_mps, curvature_1pm].")
    scales = _action_scales(action_chunk, speed_scale, curvature_scale)
    if inverse:
        return action_chunk * scales
    return action_chunk / scales


def _action_scales(
    action_chunk: torch.Tensor,
    speed_scale: float,
    curvature_scale: float,
) -> torch.Tensor:
    if speed_scale == 0 or curvature_scale == 0:
        raise ValueError("speed_scale and curvature_scale must be non-zero.")
    return action_chunk.new_tensor([speed_scale, curvature_scale])
