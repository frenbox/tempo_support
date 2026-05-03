from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class EvidenceTemperatureScaler:
    """Single-parameter temperature scaling for Dirichlet evidence.

    We scale evidence as: e_T = e / T, with T > 0.
    alpha_T = e_T + 1, and p_T = alpha_T / sum(alpha_T).

    This is a lightweight, stable calibrator that preserves the evidential semantics
    (T>1 => less evidence => more uncertainty).
    """

    temperature: float = 1.0

    def fit(self, evidence: np.ndarray, y_true: np.ndarray, *, max_steps: int = 500, lr: float = 0.05) -> "EvidenceTemperatureScaler":
        e = torch.from_numpy(np.asarray(evidence, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y_true, dtype=np.int64))
        logT = torch.zeros((), dtype=torch.float32, requires_grad=True)

        opt = torch.optim.Adam([logT], lr=lr)

        for _ in range(max_steps):
            T = torch.exp(logT).clamp(1e-3, 1e3)
            alpha = e / T + 1.0
            p = alpha / alpha.sum(dim=-1, keepdim=True)
            loss = F.nll_loss(torch.log(p.clamp_min(1e-12)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.temperature = float(torch.exp(logT).detach().cpu().item())
        return self

    def transform(self, evidence: np.ndarray) -> np.ndarray:
        T = float(self.temperature)
        alpha = evidence / T + 1.0
        p = alpha / np.sum(alpha, axis=1, keepdims=True)
        return p

    def transform_alpha(self, alpha: np.ndarray) -> np.ndarray:
        T = float(self.temperature)
        evidence = np.maximum(alpha - 1.0, 0.0)
        alpha_t = evidence / T + 1.0
        return alpha_t


@dataclass
class EvidenceVectorScaler:
    """Per-class evidence scaling e'_c = e_c / T_c with T_c > 0."""

    temperature: Optional[np.ndarray] = None

    def fit(self, evidence: np.ndarray, y_true: np.ndarray, *, max_steps: int = 800, lr: float = 0.03) -> "EvidenceVectorScaler":
        e = torch.from_numpy(np.asarray(evidence, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y_true, dtype=np.int64))
        C = e.shape[1]
        logT = torch.zeros((C,), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([logT], lr=lr)

        for _ in range(max_steps):
            T = torch.exp(logT).clamp(1e-3, 1e3)
            alpha = e / T[None, :] + 1.0
            p = alpha / alpha.sum(dim=-1, keepdim=True)
            loss = F.nll_loss(torch.log(p.clamp_min(1e-12)), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        self.temperature = np.exp(logT.detach().cpu().numpy())
        return self

    def transform(self, evidence: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("Vector scaler must be fit before transform")
        T = np.asarray(self.temperature, dtype=np.float64)
        alpha = np.asarray(evidence, dtype=np.float64) / T[None, :] + 1.0
        p = alpha / np.sum(alpha, axis=1, keepdims=True)
        return p
