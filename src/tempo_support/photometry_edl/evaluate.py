from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

from .plotting import NORD, nord_cmap, savefig_pdf, setup_mpl_paper, style_axes_inward


def ece(probs: np.ndarray, y_true: np.ndarray, *, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    e = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi if i < n_bins - 1 else conf <= hi)
        if not np.any(m):
            continue
        acc_b = correct[m].mean()
        conf_b = conf[m].mean()
        e += np.abs(acc_b - conf_b) * (m.sum() / N)
    return float(e)


def brier_score_multiclass(probs: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    y_oh = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def compute_basic_metrics(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    pred = probs.argmax(axis=1)

    y_oh = label_binarize(y_true, classes=np.arange(num_classes))

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    metrics["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    metrics["macro_precision"] = float(precision_score(y_true, pred, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, pred, average="macro", zero_division=0))
    try:
        metrics["top2_acc"] = float(top_k_accuracy_score(y_true, probs, k=2, labels=list(range(num_classes))))
    except Exception:
        metrics["top2_acc"] = float("nan")
    try:
        metrics["top3_acc"] = float(top_k_accuracy_score(y_true, probs, k=min(3, num_classes), labels=list(range(num_classes))))
    except Exception:
        metrics["top3_acc"] = float("nan")
    metrics["ece"] = float(ece(probs, y_true))
    metrics["brier"] = float(brier_score_multiclass(probs, y_true, num_classes))
    # NLL (log-loss)
    metrics["nll"] = float(log_loss(y_true, probs, labels=list(range(num_classes))))

    # AUROC/AUPRC macro
    try:
        metrics["auroc_ovr_macro"] = float(roc_auc_score(y_oh, probs, average="macro", multi_class="ovr"))
    except Exception:
        metrics["auroc_ovr_macro"] = float("nan")
    try:
        metrics["auprc_macro"] = float(average_precision_score(y_oh, probs, average="macro"))
    except Exception:
        metrics["auprc_macro"] = float("nan")

    return metrics


def plot_confusion(
    cm: np.ndarray,
    class_names: List[str],
    *,
    normalize: bool,
    title: str,
) -> plt.Figure:
    setup_mpl_paper(usetex=True)
    cm_plot = cm.astype(float)
    if normalize:
        cm_plot = cm_plot / np.maximum(cm_plot.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=300)
    im = ax.imshow(cm_plot, cmap=nord_cmap("nord_blues"), vmin=0.0, vmax=1.0 if normalize else None)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center", color=NORD["nord0"] if val < 0.7 else NORD["nord6"], fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _normalize_confusion(cm: np.ndarray, mode: str) -> np.ndarray:
    cmf = cm.astype(float)
    if mode == "recall":
        return cmf / np.maximum(cmf.sum(axis=1, keepdims=True), 1.0)
    if mode == "precision":
        return cmf / np.maximum(cmf.sum(axis=0, keepdims=True), 1.0)
    raise ValueError(f"Unknown normalization mode: {mode}")


def plot_confusion_panel(
    cm: np.ndarray,
    class_names: List[str],
    *,
    title: str,
) -> plt.Figure:
    setup_mpl_paper(usetex=True)

    mats = [
        ("Raw Counts", cm.astype(float), False, float(np.max(cm)) if np.size(cm) else 1.0),
        ("Recall Normalized", _normalize_confusion(cm, "recall"), True, 1.0),
        ("Precision Normalized", _normalize_confusion(cm, "precision"), True, 1.0),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=300)
    for ax, (subtitle, mat, is_norm, vmax) in zip(axs, mats):
        im = ax.imshow(mat, cmap=nord_cmap("nord_blues"), vmin=0.0, vmax=vmax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_title(subtitle)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = float(mat[i, j])
                txt = f"{val:.2f}" if is_norm else str(int(cm[i, j]))
                color = NORD["nord0"] if val < 0.7 * max(vmax, 1e-9) else NORD["nord6"]
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_abstention_overview(
    y_true: np.ndarray,
    pred: np.ndarray,
    keep_mask: np.ndarray,
    class_names: List[str],
    *,
    title: str,
) -> plt.Figure:
    setup_mpl_paper(usetex=True)
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    keep_mask = np.asarray(keep_mask, dtype=bool)

    C = len(class_names)
    kept = np.zeros(C, dtype=float)
    abst = np.zeros(C, dtype=float)
    wrong_kept = np.zeros(C, dtype=float)
    for c in range(C):
        m = y_true == c
        kept[c] = float(np.sum(m & keep_mask))
        abst[c] = float(np.sum(m & ~keep_mask))
        wrong_kept[c] = float(np.sum(m & keep_mask & (pred != y_true)))

    kept_total = float(np.sum(keep_mask))
    abst_total = float(np.sum(~keep_mask))
    corr_kept_total = float(np.sum(keep_mask & (pred == y_true)))
    wrong_kept_total = float(np.sum(keep_mask & (pred != y_true)))

    x = np.arange(C)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.5), dpi=300)
    ax0, ax1 = axs

    ax0.bar(x, kept, color=NORD["nord14"], label="kept")
    ax0.bar(x, abst, bottom=kept, color=NORD["nord11"], label="abstained")
    ax0.plot(x, wrong_kept, "o", color=NORD["nord3"], label="wrong-kept")
    ax0.set_xticks(x)
    ax0.set_xticklabels(class_names, rotation=45, ha="right")
    ax0.set_ylabel("count")
    ax0.set_title("By True Class: Kept vs Abstained")
    style_axes_inward(ax0, grid_y=True)
    ax0.legend(fontsize=8)

    bars = np.array([corr_kept_total, wrong_kept_total, abst_total], dtype=float)
    labels = ["kept-correct", "kept-wrong", "abstained"]
    colors = [NORD["nord14"], NORD["nord13"], NORD["nord11"]]
    ax1.bar(np.arange(3), bars, color=colors)
    tot = max(float(np.sum(bars)), 1.0)
    for i, v in enumerate(bars):
        ax1.text(i, v + 0.01 * max(np.max(bars), 1.0), f"{v:.0f}\n({100.0*v/tot:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("count")
    ax1.set_title("Global Abstention Breakdown")
    style_axes_inward(ax1, grid_y=True)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_reliability(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 15,
    title: str = "Reliability diagram",
) -> plt.Figure:
    setup_mpl_paper(usetex=True)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_acc = []
    bin_conf = []
    bin_frac = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi if i < n_bins - 1 else conf <= hi)
        if not np.any(m):
            continue
        bin_acc.append(correct[m].mean())
        bin_conf.append(conf[m].mean())
        bin_frac.append(m.mean())

    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=300)
    ax.plot([0, 1], [0, 1], ls=(0, (2, 4)), color=NORD["nord3"], lw=1.2)
    ax.scatter(bin_conf, bin_acc, s=35, zorder=3, edgecolors=NORD["nord0"], linewidths=1.5)
    ax.plot(bin_conf, bin_acc, zorder=2, color=NORD["nord9"], linewidth = 1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title(title)
    style_axes_inward(ax, grid_y=True)
    fig.tight_layout()
    return fig


def write_report_tables(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    prefix: str,
    *,
    pred_override: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)
    pred = probs.argmax(axis=1) if pred_override is None else np.asarray(pred_override, dtype=int)
    if pred.shape[0] != np.asarray(y_true).shape[0]:
        raise ValueError(f"pred_override length {pred.shape[0]} does not match y_true length {len(y_true)}")

    cm = confusion_matrix(y_true, pred, labels=list(range(num_classes)))
    metrics = compute_basic_metrics(y_true, probs, num_classes)
    if pred_override is not None:
        labels = list(range(num_classes))
        metrics["accuracy"] = float(accuracy_score(y_true, pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
        metrics["macro_f1"] = float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0))
        metrics["macro_precision"] = float(precision_score(y_true, pred, labels=labels, average="macro", zero_division=0))
        metrics["macro_recall"] = float(recall_score(y_true, pred, labels=labels, average="macro", zero_division=0))

    rep = classification_report(
        y_true,
        pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / f"{prefix}_classification_report.json").write_text(json.dumps(rep, indent=2))
    np.savetxt(out_dir / f"{prefix}_confusion_counts.csv", cm, delimiter=",")

    # figures
    fig_panel = plot_confusion_panel(cm, class_names, title=f"{prefix}: confusion (raw / recall-norm / precision-norm)")
    savefig_pdf(fig_panel, out_dir / f"{prefix}_confusion_panel.pdf")
    plt.close(fig_panel)

    fig_cm = plot_confusion(cm, class_names, normalize=True, title=f"{prefix}: normalized confusion")
    savefig_pdf(fig_cm, out_dir / f"{prefix}_confusion_norm.pdf")
    plt.close(fig_cm)

    fig_rel = plot_reliability(probs, y_true, title=f"{prefix}: reliability (ECE={metrics['ece']:.3f})")
    savefig_pdf(fig_rel, out_dir / f"{prefix}_reliability.pdf")
    plt.close(fig_rel)

    return metrics
