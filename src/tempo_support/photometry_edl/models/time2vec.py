from __future__ import annotations

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Time2Vec positional encoding (Kazemi et al., 2019).

    Maps scalar time t to a d-dimensional embedding:
      v0 = w0 * t + b0
      vi = sin(wi * t + bi), i>=1
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be >= 2")
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, L) returns (B, L, d_model)."""
        v0 = self.w0 * t + self.b0  # (B, L)
        vp = torch.sin(t.unsqueeze(-1) * self.w + self.b)  # (B, L, d_model-1)
        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)
