import torch
from torch import nn


class GeometryPredictor(nn.Module):
    """Predict compact future-trajectory geometry from Qwen hidden state and ego history."""

    def __init__(
        self,
        qwen_hidden_dim: int = 3584,
        history_steps: int = 10,
        ego_dim: int = 3,
        qwen_embed_dim: int = 512,
        ego_embed_dim: int = 128,
        fusion_hidden_size: int = 512,
        geometry_descriptor_dim: int = 16,
    ):
        super().__init__()
        self.qwen_hidden_dim = qwen_hidden_dim
        self.history_steps = history_steps
        self.ego_dim = ego_dim
        self.qwen_embed_dim = qwen_embed_dim
        self.ego_embed_dim = ego_embed_dim
        self.fusion_hidden_size = fusion_hidden_size
        self.geometry_descriptor_dim = geometry_descriptor_dim

        ego_input_dim = history_steps * ego_dim
        self.qwen_projection = nn.Sequential(
            nn.LayerNorm(qwen_hidden_dim),
            nn.Linear(qwen_hidden_dim, qwen_embed_dim),
            nn.GELU(),
        )
        self.ego_encoder = nn.Sequential(
            nn.LayerNorm(ego_input_dim),
            nn.Linear(ego_input_dim, ego_embed_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(qwen_embed_dim + ego_embed_dim, fusion_hidden_size),
            nn.GELU(),
            nn.Linear(fusion_hidden_size, geometry_descriptor_dim),
        )

    def forward(
        self,
        qwen_hidden: torch.Tensor,
        ego_history: torch.Tensor,
    ) -> torch.Tensor:
        if qwen_hidden.ndim != 2:
            raise ValueError("qwen_hidden must have shape [B, qwen_hidden_dim].")
        if qwen_hidden.shape[-1] != self.qwen_hidden_dim:
            raise ValueError(
                f"qwen_hidden last dim must be {self.qwen_hidden_dim}, got {qwen_hidden.shape[-1]}."
            )
        if ego_history.ndim != 3:
            raise ValueError("ego_history must have shape [B, history_steps, ego_dim].")
        expected = (self.history_steps, self.ego_dim)
        if tuple(ego_history.shape[1:]) != expected:
            raise ValueError(f"ego_history trailing shape must be {expected}, got {tuple(ego_history.shape[1:])}.")
        if qwen_hidden.shape[0] != ego_history.shape[0]:
            raise ValueError("qwen_hidden and ego_history must have the same batch size.")

        qwen_embed = self.qwen_projection(qwen_hidden)
        ego_flat = ego_history.reshape(ego_history.shape[0], -1)
        ego_embed = self.ego_encoder(ego_flat)
        return self.fusion(torch.cat([qwen_embed, ego_embed], dim=-1))
