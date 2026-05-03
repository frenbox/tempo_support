from __future__ import annotations

import numpy as np
import torch


def alpha_from_evidence(evidence: torch.Tensor) -> torch.Tensor:
    return (evidence + 1.0).clamp_min(1.0 + 1e-6)


def dirichlet_mean(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / alpha.sum(dim=-1, keepdim=True)


def vacuity(alpha: torch.Tensor) -> torch.Tensor:
    C = alpha.size(-1)
    return C / alpha.sum(dim=-1)


def predictive_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = p.clamp_min(eps)
    return -(q * q.log()).sum(dim=-1)


def expected_categorical_entropy(alpha: torch.Tensor) -> torch.Tensor:
    # E_p~Dir(alpha)[ H(Categorical(p)) ]
    S = alpha.sum(dim=-1, keepdim=True)
    return -((alpha / S) * (torch.digamma(alpha + 1.0) - torch.digamma(S + 1.0))).sum(dim=-1)


def mutual_information(alpha: torch.Tensor) -> torch.Tensor:
    p = dirichlet_mean(alpha)
    return predictive_entropy(p) - expected_categorical_entropy(alpha)

def dirichlet_variance(alpha: torch.Tensor) -> torch.Tensor:
    """Dirichlet marginal variance for each class probability.

    Var[p_k] = α_k(α0-α_k) / (α0^2(α0+1))

    Returns: (B, C)
    """
    S = alpha.sum(dim=-1, keepdim=True)
    return alpha * (S - alpha) / (S * S * (S + 1.0))


def dirichlet_std(alpha: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dirichlet_variance(alpha), min=0.0))


def total_uncertainty_trace(alpha: torch.Tensor) -> torch.Tensor:
    """Scalar uncertainty = sqrt(trace(Cov[p]))."""
    v = dirichlet_variance(alpha)
    return torch.sqrt(torch.clamp(v.sum(dim=-1), min=0.0))



def numpy_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.clip(probs, eps, 1.0)
    return -np.sum(q * np.log(q), axis=1)
