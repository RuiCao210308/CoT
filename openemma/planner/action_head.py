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


class EgoOnlyActionHead(nn.Module):
    """MLP baseline that predicts future actions from observed ego history only."""

    def __init__(
        self,
        history_steps: int = 10,
        ego_dim: int = 3,
        chunk_size: int = 10,
        action_dim: int = 2,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_steps = history_steps
        self.ego_dim = ego_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        input_dim = history_steps * ego_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, chunk_size * action_dim),
        )

    def forward(self, ego_history_array: torch.Tensor) -> torch.Tensor:
        if ego_history_array.ndim != 3:
            raise ValueError("ego_history_array must have shape [B, history_steps, ego_dim].")
        expected = (self.history_steps, self.ego_dim)
        if tuple(ego_history_array.shape[1:]) != expected:
            raise ValueError(
                f"ego_history_array trailing shape must be {expected}, got {tuple(ego_history_array.shape[1:])}."
            )
        flat = ego_history_array.reshape(ego_history_array.shape[0], -1)
        actions = self.net(flat)
        return actions.view(-1, self.chunk_size, self.action_dim)


class FusionActionHead(nn.Module):
    """Fuse Qwen planning hidden state with structured ego history for action chunks."""

    def __init__(
        self,
        qwen_hidden_dim: int = 3584,
        history_steps: int = 10,
        ego_dim: int = 3,
        ego_embed_dim: int = 256,
        qwen_embed_dim: int = 512,
        fusion_hidden_size: int = 1024,
        chunk_size: int = 10,
        action_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.qwen_hidden_dim = qwen_hidden_dim
        self.history_steps = history_steps
        self.ego_dim = ego_dim
        self.ego_embed_dim = ego_embed_dim
        self.qwen_embed_dim = qwen_embed_dim
        self.fusion_hidden_size = fusion_hidden_size
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        ego_input_dim = history_steps * ego_dim
        self.ego_encoder = nn.Sequential(
            nn.LayerNorm(ego_input_dim),
            nn.Linear(ego_input_dim, ego_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.qwen_projection = nn.Sequential(
            nn.LayerNorm(qwen_hidden_dim),
            nn.Linear(qwen_hidden_dim, qwen_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(qwen_embed_dim + ego_embed_dim, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, chunk_size * action_dim),
        )

    def forward(
        self,
        planning_hidden: torch.Tensor,
        ego_history_array: torch.Tensor,
    ) -> torch.Tensor:
        if planning_hidden.ndim != 2:
            raise ValueError("planning_hidden must have shape [B, qwen_hidden_dim].")
        if planning_hidden.shape[-1] != self.qwen_hidden_dim:
            raise ValueError(
                f"planning_hidden last dim must be {self.qwen_hidden_dim}, "
                f"got {planning_hidden.shape[-1]}."
            )
        if ego_history_array.ndim != 3:
            raise ValueError("ego_history_array must have shape [B, history_steps, ego_dim].")
        expected = (self.history_steps, self.ego_dim)
        if tuple(ego_history_array.shape[1:]) != expected:
            raise ValueError(
                f"ego_history_array trailing shape must be {expected}, "
                f"got {tuple(ego_history_array.shape[1:])}."
            )
        if planning_hidden.shape[0] != ego_history_array.shape[0]:
            raise ValueError("planning_hidden and ego_history_array must have the same batch size.")

        ego_flat = ego_history_array.reshape(ego_history_array.shape[0], -1)
        ego_embed = self.ego_encoder(ego_flat)
        qwen_embed = self.qwen_projection(planning_hidden)
        fused = torch.cat([qwen_embed, ego_embed], dim=-1)
        actions = self.fusion(fused)
        return actions.view(-1, self.chunk_size, self.action_dim)


class DecoupledEgoVLAActionHead(nn.Module):
    """Ego-grounded VLA action head with decoupled speed and curvature predictors."""

    def __init__(
        self,
        qwen_hidden_dim: int = 3584,
        history_steps: int = 10,
        ego_dim: int = 3,
        qwen_embed_dim: int = 512,
        ego_embed_dim: int = 256,
        fusion_dim: int = 1024,
        speed_hidden: int = 512,
        curvature_hidden: int = 512,
        chunk_size: int = 10,
        action_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.qwen_hidden_dim = qwen_hidden_dim
        self.history_steps = history_steps
        self.ego_dim = ego_dim
        self.qwen_embed_dim = qwen_embed_dim
        self.ego_embed_dim = ego_embed_dim
        self.fusion_dim = fusion_dim
        self.speed_hidden = speed_hidden
        self.curvature_hidden = curvature_hidden
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        ego_input_dim = history_steps * ego_dim
        self.qwen_projection = nn.Sequential(
            nn.LayerNorm(qwen_hidden_dim),
            nn.Linear(qwen_hidden_dim, qwen_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ego_encoder = nn.Sequential(
            nn.LayerNorm(ego_input_dim),
            nn.Linear(ego_input_dim, ego_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_trunk = nn.Sequential(
            nn.Linear(qwen_embed_dim + ego_embed_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(fusion_dim, speed_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(speed_hidden, chunk_size),
        )
        self.curvature_head = nn.Sequential(
            nn.Linear(fusion_dim, curvature_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(curvature_hidden, chunk_size),
        )

    def forward(
        self,
        planning_hidden: torch.Tensor,
        ego_history_array: torch.Tensor,
    ) -> torch.Tensor:
        if planning_hidden.ndim != 2:
            raise ValueError("planning_hidden must have shape [B, qwen_hidden_dim].")
        if planning_hidden.shape[-1] != self.qwen_hidden_dim:
            raise ValueError(
                f"planning_hidden last dim must be {self.qwen_hidden_dim}, "
                f"got {planning_hidden.shape[-1]}."
            )
        if ego_history_array.ndim != 3:
            raise ValueError("ego_history_array must have shape [B, history_steps, ego_dim].")
        expected = (self.history_steps, self.ego_dim)
        if tuple(ego_history_array.shape[1:]) != expected:
            raise ValueError(
                f"ego_history_array trailing shape must be {expected}, "
                f"got {tuple(ego_history_array.shape[1:])}."
            )
        if planning_hidden.shape[0] != ego_history_array.shape[0]:
            raise ValueError("planning_hidden and ego_history_array must have the same batch size.")
        if self.action_dim != 2:
            raise ValueError("DecoupledEgoVLAActionHead currently expects action_dim=2.")

        qwen_embed = self.qwen_projection(planning_hidden)
        ego_flat = ego_history_array.reshape(ego_history_array.shape[0], -1)
        ego_embed = self.ego_encoder(ego_flat)
        fused = torch.cat([qwen_embed, ego_embed], dim=-1)
        trunk = self.fusion_trunk(fused)
        speed = self.speed_head(trunk).unsqueeze(-1)
        curvature = self.curvature_head(trunk).unsqueeze(-1)
        return torch.cat([speed, curvature], dim=-1)


class WaypointAuxFusionHead(nn.Module):
    """Fusion action head with an auxiliary ego-local future waypoint decoder."""

    def __init__(
        self,
        qwen_hidden_dim: int = 3584,
        history_steps: int = 10,
        ego_dim: int = 3,
        qwen_embed_dim: int = 512,
        ego_embed_dim: int = 256,
        fusion_hidden_size: int = 1024,
        chunk_size: int = 10,
        action_dim: int = 2,
        waypoint_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.qwen_hidden_dim = qwen_hidden_dim
        self.history_steps = history_steps
        self.ego_dim = ego_dim
        self.qwen_embed_dim = qwen_embed_dim
        self.ego_embed_dim = ego_embed_dim
        self.fusion_hidden_size = fusion_hidden_size
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.waypoint_dim = waypoint_dim

        ego_input_dim = history_steps * ego_dim
        self.qwen_projection = nn.Sequential(
            nn.LayerNorm(qwen_hidden_dim),
            nn.Linear(qwen_hidden_dim, qwen_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ego_encoder = nn.Sequential(
            nn.LayerNorm(ego_input_dim),
            nn.Linear(ego_input_dim, ego_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_trunk = nn.Sequential(
            nn.Linear(qwen_embed_dim + ego_embed_dim, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.GELU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, chunk_size * action_dim),
        )
        self.waypoint_head = nn.Sequential(
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, chunk_size * waypoint_dim),
        )

    def forward(
        self,
        planning_hidden: torch.Tensor,
        ego_history_array: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if planning_hidden.ndim != 2:
            raise ValueError("planning_hidden must have shape [B, qwen_hidden_dim].")
        if planning_hidden.shape[-1] != self.qwen_hidden_dim:
            raise ValueError(
                f"planning_hidden last dim must be {self.qwen_hidden_dim}, "
                f"got {planning_hidden.shape[-1]}."
            )
        if ego_history_array.ndim != 3:
            raise ValueError("ego_history_array must have shape [B, history_steps, ego_dim].")
        expected = (self.history_steps, self.ego_dim)
        if tuple(ego_history_array.shape[1:]) != expected:
            raise ValueError(
                f"ego_history_array trailing shape must be {expected}, "
                f"got {tuple(ego_history_array.shape[1:])}."
            )
        if planning_hidden.shape[0] != ego_history_array.shape[0]:
            raise ValueError("planning_hidden and ego_history_array must have the same batch size.")

        qwen_embed = self.qwen_projection(planning_hidden)
        ego_flat = ego_history_array.reshape(ego_history_array.shape[0], -1)
        ego_embed = self.ego_encoder(ego_flat)
        fused = torch.cat([qwen_embed, ego_embed], dim=-1)
        trunk = self.fusion_trunk(fused)
        action_chunk = self.action_head(trunk).view(-1, self.chunk_size, self.action_dim)
        waypoints = self.waypoint_head(trunk).view(-1, self.chunk_size, self.waypoint_dim)
        return {"action_chunk": action_chunk, "waypoints": waypoints}


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


def build_fusion_action_head(
    qwen_hidden_dim: int = 3584,
    history_steps: int = 10,
    ego_dim: int = 3,
    ego_embed_dim: int = 256,
    qwen_embed_dim: int = 512,
    fusion_hidden_size: int = 1024,
    chunk_size: int = 10,
    action_dim: int = 2,
    dropout: float = 0.1,
) -> FusionActionHead:
    return FusionActionHead(
        qwen_hidden_dim=qwen_hidden_dim,
        history_steps=history_steps,
        ego_dim=ego_dim,
        ego_embed_dim=ego_embed_dim,
        qwen_embed_dim=qwen_embed_dim,
        fusion_hidden_size=fusion_hidden_size,
        chunk_size=chunk_size,
        action_dim=action_dim,
        dropout=dropout,
    )


def build_decoupled_egovla_action_head(
    qwen_hidden_dim: int = 3584,
    history_steps: int = 10,
    ego_dim: int = 3,
    qwen_embed_dim: int = 512,
    ego_embed_dim: int = 256,
    fusion_dim: int = 1024,
    speed_hidden: int = 512,
    curvature_hidden: int = 512,
    chunk_size: int = 10,
    action_dim: int = 2,
    dropout: float = 0.1,
) -> DecoupledEgoVLAActionHead:
    return DecoupledEgoVLAActionHead(
        qwen_hidden_dim=qwen_hidden_dim,
        history_steps=history_steps,
        ego_dim=ego_dim,
        qwen_embed_dim=qwen_embed_dim,
        ego_embed_dim=ego_embed_dim,
        fusion_dim=fusion_dim,
        speed_hidden=speed_hidden,
        curvature_hidden=curvature_hidden,
        chunk_size=chunk_size,
        action_dim=action_dim,
        dropout=dropout,
    )


def build_waypoint_aux_fusion_head(
    qwen_hidden_dim: int = 3584,
    history_steps: int = 10,
    ego_dim: int = 3,
    qwen_embed_dim: int = 512,
    ego_embed_dim: int = 256,
    fusion_hidden_size: int = 1024,
    chunk_size: int = 10,
    action_dim: int = 2,
    waypoint_dim: int = 2,
    dropout: float = 0.1,
) -> WaypointAuxFusionHead:
    return WaypointAuxFusionHead(
        qwen_hidden_dim=qwen_hidden_dim,
        history_steps=history_steps,
        ego_dim=ego_dim,
        qwen_embed_dim=qwen_embed_dim,
        ego_embed_dim=ego_embed_dim,
        fusion_hidden_size=fusion_hidden_size,
        chunk_size=chunk_size,
        action_dim=action_dim,
        waypoint_dim=waypoint_dim,
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
