from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def dirichlet_kl(alpha: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(prior) ) for batches."""
    sum_alpha = alpha.sum(dim=-1)
    sum_prior = prior.sum(dim=-1)

    lgamma_sum_alpha = torch.lgamma(sum_alpha)
    lgamma_sum_prior = torch.lgamma(sum_prior)
    lgamma_alpha = torch.lgamma(alpha).sum(dim=-1)
    lgamma_prior = torch.lgamma(prior).sum(dim=-1)

    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(sum_alpha).unsqueeze(-1)

    t1 = lgamma_sum_alpha - lgamma_alpha
    t2 = -(lgamma_sum_prior - lgamma_prior)
    t3 = ((alpha - prior) * (digamma_alpha - digamma_sum_alpha)).sum(dim=-1)
    return t1 + t2 + t3


@dataclass
class LossBreakdown:
    total: torch.Tensor
    nll: torch.Tensor
    kl_unscaled: torch.Tensor
    kl_annealed: torch.Tensor
    suppressor: torch.Tensor
    margin: torch.Tensor
    anneal_coef: torch.Tensor

    def mean_dict(self) -> Dict[str, float]:
        return {
            "loss_total": float(self.total.mean().item()),
            "loss_nll": float(self.nll.mean().item()),
            "loss_kl_unscaled": float(self.kl_unscaled.mean().item()),
            "loss_kl_annealed": float(self.kl_annealed.mean().item()),
            "loss_suppressor": float(self.suppressor.mean().item()),
            "loss_margin": float(self.margin.mean().item()),
            "anneal_coef": float(self.anneal_coef.mean().item()),
        }


class EvidentialDirichletLoss(nn.Module):
    """EDL loss with optional class-balanced and focal-style NLL weighting."""

    def __init__(
        self,
        *,
        num_classes: int,
        anneal_epochs: int = 60,
        kl_strength: float = 5e-4,
        sup_strength: float = 2e-2,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        margin_strength: float = 0.0,
        margin_delta: float = 0.0,
        margin_pairs: Optional[list[list[int]]] = None,
    ):
        super().__init__()
        self.C = int(num_classes)
        self.anneal_epochs = int(anneal_epochs)
        self.kl_strength = float(kl_strength)
        self.sup_strength = float(sup_strength)
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.margin_strength = float(margin_strength)
        self.margin_delta = float(margin_delta)

        # if class_weights is None:
        #     self.register_buffer("class_weights", torch.ones(self.C, dtype=torch.float32), persistent=False)
        # else:
        #     cw = class_weights.detach().float().view(-1)
        #     if cw.numel() != self.C:
        #         raise ValueError(f"class_weights must have {self.C} entries")
        #     self.register_buffer("class_weights", cw, persistent=False)

        # inside EvidentialDirichletLoss.__init__(...)
        if class_weights is None:
            cw = torch.ones(num_classes, dtype=torch.float32)
        else:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            assert cw.numel() == num_classes, "class_weights must have length = num_classes"

        self.register_buffer("class_weights", cw)  # <- important
        rival = torch.full((self.C,), fill_value=-1, dtype=torch.long)
        if margin_pairs is not None:
            for pair in margin_pairs:
                if len(pair) != 2:
                    raise ValueError(f"Each margin pair must have 2 entries, got {pair}")
                true_c, rival_c = int(pair[0]), int(pair[1])
                if true_c < 0 or true_c >= self.C or rival_c < 0 or rival_c >= self.C:
                    raise ValueError(f"Margin pair out of bounds: {pair} for C={self.C}")
                rival[true_c] = rival_c
        self.register_buffer("margin_rival_map", rival, persistent=False)


    def _components(self, evidence: torch.Tensor, target: torch.Tensor, *, epoch: Optional[int]) -> LossBreakdown:
        alpha = (evidence + 1.0).clamp_min(1.0 + 1e-6)
        if torch.any(evidence < -1e-8):
            raise RuntimeError("Evidence must be >= 0")
        if torch.any(alpha <= 1.0):
            raise RuntimeError("Alpha must be > 1")

        S = alpha.sum(dim=-1, keepdim=True)

        y = F.one_hot(target, self.C).float()
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            y = (1.0 - eps) * y + eps / self.C

        nll = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)

        p = alpha / S
        if self.focal_gamma > 0.0:
            p_t = p.gather(1, target.view(-1, 1)).squeeze(1)
            nll = nll * (1.0 - p_t).pow(self.focal_gamma)

        cw = self.class_weights[target]
        nll = nll * cw

        mse = torch.sum((y - p) ** 2, dim=-1)
        sup = self.sup_strength * mse * torch.clamp(S.squeeze(-1) - self.C, min=0.0)

        margin = torch.zeros_like(nll)
        if self.margin_strength > 0.0 and self.margin_delta > 0.0:
            rival = self.margin_rival_map[target]
            valid = rival >= 0
            if torch.any(valid):
                ridx = rival.clamp_min(0)
                p_true = p[torch.arange(p.size(0), device=p.device), target]
                p_rival = p[torch.arange(p.size(0), device=p.device), ridx]
                m = F.relu(self.margin_delta - (p_true - p_rival))
                margin = self.margin_strength * m * valid.float() * cw

        prior = torch.ones_like(alpha)
        kl = dirichlet_kl(alpha, prior)

        a = 1.0 if epoch is None else min(1.0, float(epoch) / float(max(1, self.anneal_epochs)))
        a_t = torch.full_like(nll, fill_value=float(a))
        kl_annealed = a_t * self.kl_strength * kl
        total = nll + kl_annealed + sup + margin

        return LossBreakdown(total=total, nll=nll, kl_unscaled=kl, kl_annealed=kl_annealed, suppressor=sup, margin=margin, anneal_coef=a_t)

    def forward(
        self,
        evidence: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: Optional[int] = None,
        return_components: bool = False,
    ):
        b = self._components(evidence, target, epoch=epoch)
        loss = b.total.mean()
        if return_components:
            return loss, b
        return loss
