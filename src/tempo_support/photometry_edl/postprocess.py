from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import fbeta_score, precision_recall_curve


def adjust_for_priors(probs: np.ndarray, pi_train: np.ndarray, pi_deploy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Prior correction (label shift) using Bayes rule.

    p_adj(c|x) ∝ p(c|x) * pi_deploy(c) / pi_train(c)
    """
    probs = np.asarray(probs, dtype=float)
    w = (np.asarray(pi_deploy, dtype=float) + eps) / (np.asarray(pi_train, dtype=float) + eps)
    out = probs * w[None, :]
    out = out / np.sum(out, axis=1, keepdims=True)
    return out


def fit_prob_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    beta_by_class: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Find per-class probability thresholds that maximize F_beta (one-vs-rest)."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    C = probs.shape[1]
    beta_by_class = np.ones(C) if beta_by_class is None else np.asarray(beta_by_class, dtype=float)
    thr = np.zeros(C, dtype=float)

    for c in range(C):
        y_bin = (y_true == c).astype(int)
        prec, rec, t = precision_recall_curve(y_bin, probs[:, c])
        beta = float(beta_by_class[c])
        # PR curve outputs a precision/recall point *before* first threshold, so align:
        F = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-12, None)
        if len(t) == 0:
            thr[c] = 1.0
            continue
        # thresholds correspond to prec[1:], rec[1:]
        i = int(np.nanargmax(F[1:]))
        thr[c] = float(t[i])
    return thr


def predict_with_thresholds(probs: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Predict labels with per-class thresholds.

    If at least one class exceeds its threshold, pick the exceeding class with
    the highest probability. Otherwise fall back to argmax.
    """
    probs = np.asarray(probs, dtype=float)
    thr = np.asarray(thr, dtype=float)
    keep = probs >= thr[None, :]
    pred = probs.argmax(axis=1)
    any_keep = keep.any(axis=1)
    if np.any(any_keep):
        masked = np.where(keep, probs, -np.inf)
        pred[any_keep] = masked[any_keep].argmax(axis=1)
    return pred


def pick_uncertainty_cut(
    uncertainty: np.ndarray,
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    beta: float = 1.0,
    average: str = "macro",
    labels: Optional[np.ndarray] = None,
    min_keep: int = 100,
) -> Tuple[float, float, float]:
    """Pick an uncertainty threshold to maximize a metric on the kept set.

    Returns (u_thresh, score, coverage)
    """
    u = np.asarray(uncertainty, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)

    best_score = -1.0
    best_ut = float(np.max(u))
    best_cov = 1.0

    for q in range(0, 101):
        ut = float(np.percentile(u, q))
        keep = u <= ut
        if keep.sum() < min_keep:
            continue
        score = fbeta_score(y_true[keep], pred[keep], beta=beta, average=average, labels=labels, zero_division=0)
        cov = float(keep.mean())
        if score > best_score:
            best_score, best_ut, best_cov = float(score), ut, cov

    return best_ut, best_score, best_cov


def risk_coverage_curve(uncertainty: np.ndarray, correct: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve (select most certain first).

    uncertainty: lower = more certain
    correct: boolean array
    Returns:
      coverage (increasing), risk (error rate among kept)
    """
    u = np.asarray(uncertainty, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(u)
    corr_sorted = correct[order]

    N = len(correct)
    cov = np.arange(1, N + 1) / N
    err_cum = np.cumsum(~corr_sorted)
    risk = err_cum / np.arange(1, N + 1)
    return cov, risk


def aurc(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    cov, risk = risk_coverage_curve(uncertainty, correct)
    return float(np.trapz(risk, cov))
