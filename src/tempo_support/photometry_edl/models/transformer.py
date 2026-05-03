from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .time2vec import Time2Vec


class EventTransformerEncoder(nn.Module):
    """Transformer encoder for variable-length event sequences."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        band_mode: Literal["onehot", "embed"] = "onehot",
        band_embed_dim: int = 8,
        time_encoding: Literal["dt_first", "dt_both"] = "dt_both",
        ff_mult: int = 4,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.band_mode = band_mode
        self.time_encoding = time_encoding

        self.cont_proj = nn.Linear(4, self.d_model)
        if band_mode == "onehot":
            self.band_proj = nn.Linear(3, self.d_model)
            self.band_emb = None
            self.token_dim = 7
        elif band_mode == "embed":
            # Learned band representation (g/r/i) added to token state.
            self.band_emb = nn.Embedding(3, self.d_model)
            self.band_proj = None
            self.token_dim = 5
        else:
            raise ValueError(f"Unknown band_mode={band_mode}")

        if time_encoding == "dt_first":
            self.time2vec = Time2Vec(self.d_model)
            self.time_mlp = None
        elif time_encoding == "dt_both":
            self.time2vec = None
            self.time_mlp = nn.Sequential(
                nn.Linear(2, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
        else:
            raise ValueError(f"Unknown time_encoding={time_encoding}")

        self.cls_tok = nn.Parameter(torch.zeros(1, 1, self.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)

        nn.init.normal_(self.cls_tok, mean=0.0, std=0.02)

    def _band_term(self, x: torch.Tensor) -> torch.Tensor:
        if self.band_mode == "onehot":
            return self.band_proj(x[..., 4:7])
        band_id = x[..., 4].long().clamp(0, 2)
        return self.band_emb(band_id)

    def _time_term(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_encoding == "dt_first":
            return self.time2vec(x[..., 0])
        return self.time_mlp(x[..., :2])

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch.

        Args:
            x: (B, L, 7|5) normalized event tensor
            pad_mask: (B, L) boolean, True = pad
        """
        if x.ndim != 3 or pad_mask.ndim != 2:
            raise ValueError(f"Invalid shapes x={tuple(x.shape)} pad_mask={tuple(pad_mask.shape)}")
        if x.shape[:2] != pad_mask.shape:
            raise ValueError("x and pad_mask leading dims must match")

        B, _, D = x.shape
        if D != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {D}")

        h = self.cont_proj(x[..., :4]) + self._band_term(x)
        h = h + self.dropout(self._time_term(x))

        tok = self.cls_tok.expand(B, 1, -1)
        h = torch.cat([tok, h], dim=1)
        pad_full = torch.cat([torch.zeros(B, 1, device=pad_mask.device, dtype=torch.bool), pad_mask], dim=1)

        z = self.encoder(h, src_key_padding_mask=pad_full)
        z = self.norm(z)
        return z, pad_full


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.q, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        q = self.q.expand(tokens.size(0), 1, -1)
        scale = tokens.size(-1) ** 0.5
        attn = (q * tokens).sum(dim=-1) / scale
        attn = attn.masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(attn, dim=-1).unsqueeze(-1)
        return (w * tokens).sum(dim=1)


class EvidentialClassifier(nn.Module):
    """Evidential (Dirichlet) classifier head."""

    def __init__(
        self,
        encoder: EventTransformerEncoder,
        *,
        num_classes: int,
        dropout: float,
        pool: Literal["cls", "attn", "mean"] = "attn",
        use_global_features: bool = False,
        global_dim: int = 8,
        global_hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.use_global_features = bool(use_global_features)
        self.attn_pool = AttentionPool(encoder.d_model) if pool == "attn" else None

        if self.use_global_features:
            self.global_mlp = nn.Sequential(
                nn.LayerNorm(global_dim),
                nn.Linear(global_dim, global_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(global_hidden_dim, encoder.d_model),
            )
            head_in = encoder.d_model * 2
        else:
            self.global_mlp = None
            head_in = encoder.d_model

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )

    def _pool_tokens(self, z: torch.Tensor, pad_full: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return z[:, 0]
        tok = z[:, 1:]
        pm = pad_full[:, 1:]
        if self.pool == "mean":
            w = (~pm).float().unsqueeze(-1)
            return (tok * w).sum(dim=1) / torch.clamp(w.sum(dim=1), min=1.0)
        if self.pool == "attn":
            if self.attn_pool is None:
                raise RuntimeError("Attention pool module not initialized")
            return self.attn_pool(tok, pm)
        raise ValueError(f"Unknown pool={self.pool}")

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor, global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        z, pad_full = self.encoder(x, pad_mask)
        pooled = self._pool_tokens(z, pad_full)

        if self.use_global_features:
            if global_features is None:
                raise ValueError("global_features required when use_global_features=True")
            g = self.global_mlp(global_features.float())
            pooled = torch.cat([pooled, g], dim=-1)

        logits = self.head(pooled)
        evidence = F.softplus(logits)
        if torch.any(evidence < 0):
            raise RuntimeError("Evidence must be non-negative")
        return evidence

    @staticmethod
    def alpha_from_evidence(evidence: torch.Tensor) -> torch.Tensor:
        alpha = (evidence + 1.0).clamp_min(1.0 + 1e-6)
        if torch.any(alpha <= 1.0):
            raise RuntimeError("Alpha must be > 1")
        return alpha

    @staticmethod
    def mean_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
        return alpha / alpha.sum(dim=-1, keepdim=True)


class MaskedEventPretrainer(nn.Module):
    def __init__(self, encoder: EventTransformerEncoder):
        super().__init__()
        d = encoder.d_model
        self.encoder = encoder
        self.head_flux = nn.Linear(d, 1)
        self.head_band = nn.Linear(d, 3)
        self.head_dt = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        z, _ = self.encoder(x, pad_mask)
        tokens = z[:, 1:]
        return self.head_flux(tokens), self.head_band(tokens), self.head_dt(tokens)
