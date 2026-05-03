from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augment import AugmentConfig, OversampleAugmentDataset
from .config import Config
from .data import (
    EarlyHorizonCurriculumDataset,
    FeatureStats,
    PhotoEventDataset,
    filter_manifest_quality,
    global_feature_dim,
    make_collate,
    read_manifest_csv,
)
from .evaluate import compute_basic_metrics
from .losses import EvidentialDirichletLoss
from .models import EventTransformerEncoder, EvidentialClassifier
from .plotting import NORD, setup_mpl_paper, style_axes_inward
from .taxonomy import DEFAULT_TAXONOMY, Taxonomy
from .uncertainty import mutual_information, predictive_entropy, vacuity
from .utils import set_seed


@dataclass
class TrainHistory:
    rows: list

    def to_dict(self) -> Dict:
        return {"rows": self.rows}


class EMAState:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, sv in self.shadow.items():
            sv.mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        self.backup = {}
        msd = model.state_dict()
        for k, sv in self.shadow.items():
            self.backup[k] = msd[k].detach().clone()
            msd[k].copy_(sv)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        msd = model.state_dict()
        for k, bv in self.backup.items():
            msd[k].copy_(bv)
        self.backup = None


def _compute_effective_num_weights(labels: np.ndarray, C: int, beta: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=C).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-12)
    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32)


def _make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)


def _oversample_target(counts: np.ndarray, cfg: Config) -> int:
    nz = counts[counts > 0]
    if len(nz) == 0:
        return int(counts.max())
    if cfg.oversample_target == "max":
        return int(nz.max())
    if cfg.oversample_target == "fixed":
        return int(max(1, cfg.oversample_target_count))
    return int(np.median(nz))


def make_loaders(cfg: Config, taxonomy: Taxonomy = DEFAULT_TAXONOMY):
    data_dir = Path(cfg.data_dir)
    manifest_dir = Path(cfg.manifest_dir) if cfg.manifest_dir is not None else data_dir

    train_df_raw = read_manifest_csv(manifest_dir / "manifest_train.csv", data_dir=data_dir, path_prefix=cfg.path_prefix)
    val_df_raw = read_manifest_csv(manifest_dir / "manifest_val.csv", data_dir=data_dir, path_prefix=cfg.path_prefix)
    test_df_raw = read_manifest_csv(manifest_dir / "manifest_test.csv", data_dir=data_dir, path_prefix=cfg.path_prefix)

    qkwargs = {
        "horizon_days": cfg.horizon_days,
        "drop_i_band": cfg.drop_i_band,
        "min_obs_total": cfg.min_obs_total,
        "min_obs_g": cfg.min_obs_g,
        "min_obs_r": cfg.min_obs_r,
        "min_obs_i": cfg.min_obs_i,
        "min_bands_observed": cfg.min_bands_observed,
    }
    needs_quality_filter = bool(
        cfg.drop_i_band
        or cfg.min_obs_total > 0
        or cfg.min_obs_g > 0
        or cfg.min_obs_r > 0
        or cfg.min_obs_i > 0
        or cfg.min_bands_observed > 0
    )
    if needs_quality_filter:
        train_df, train_q = filter_manifest_quality(train_df_raw, **qkwargs)
        val_df, val_q = filter_manifest_quality(val_df_raw, **qkwargs)
        test_df, test_q = filter_manifest_quality(test_df_raw, **qkwargs)
    else:
        train_df, val_df, test_df = train_df_raw, val_df_raw, test_df_raw
        train_q = {"rows_before": len(train_df_raw), "rows_after": len(train_df_raw), "rows_dropped": 0}
        val_q = {"rows_before": len(val_df_raw), "rows_after": len(val_df_raw), "rows_dropped": 0}
        test_q = {"rows_before": len(test_df_raw), "rows_after": len(test_df_raw), "rows_dropped": 0}
    if len(train_df) == 0:
        raise RuntimeError(f"All train samples were filtered out by quality cuts: {train_q}")
    if len(val_df) == 0:
        raise RuntimeError(f"All val samples were filtered out by quality cuts: {val_q}")
    if len(test_df) == 0:
        raise RuntimeError(f"All test samples were filtered out by quality cuts: {test_q}")

    use_train_horizon_strategy = bool(cfg.train_random_horizon or cfg.train_curriculum_epochs > 0)
    if use_train_horizon_strategy:
        train_base = EarlyHorizonCurriculumDataset(
            train_df,
            taxonomy=taxonomy,
            horizon_days=cfg.horizon_days,
            band_mode=cfg.band_mode,
            drop_i_band=cfg.drop_i_band,
            random_horizon=cfg.train_random_horizon,
            random_horizon_min_days=cfg.train_random_horizon_min_days,
            random_horizon_power=cfg.train_random_horizon_power,
            curriculum_epochs=cfg.train_curriculum_epochs,
            curriculum_start_days=cfg.train_curriculum_start_days,
            seed=cfg.seed,
        )
    else:
        train_base = PhotoEventDataset(
            train_df,
            taxonomy=taxonomy,
            horizon_days=cfg.horizon_days,
            band_mode=cfg.band_mode,
            drop_i_band=cfg.drop_i_band,
        )
    val_ds = PhotoEventDataset(
        val_df,
        taxonomy=taxonomy,
        horizon_days=cfg.horizon_days,
        band_mode=cfg.band_mode,
        drop_i_band=cfg.drop_i_band,
    )
    test_ds = PhotoEventDataset(
        test_df,
        taxonomy=taxonomy,
        horizon_days=cfg.horizon_days,
        band_mode=cfg.band_mode,
        drop_i_band=cfg.drop_i_band,
    )

    stats = FeatureStats.load(data_dir / cfg.stats_file)
    collate = make_collate(
        stats,
        band_mode=cfg.band_mode,
        return_global=cfg.use_global_features,
        global_feature_set=cfg.global_feature_set,
    )

    y_train = np.array([train_base[i][1] for i in range(len(train_base))], dtype=int)
    counts = np.bincount(y_train, minlength=taxonomy.num_classes)

    aug_cfg = AugmentConfig(
        p_token_dropout=cfg.p_token_dropout,
        jitter_scale=cfg.jitter_scale,
        flux_nu=cfg.flux_nu,
        flux_jitter_frac=cfg.flux_jitter_frac,
    )

    if cfg.oversample:
        target = _oversample_target(counts, cfg)
        train_ds = OversampleAugmentDataset(
            train_base,
            y_train,
            target_per_class=target,
            augment_cfg=aug_cfg,
            oversample_classes=cfg.oversample_classes,
            seed=cfg.seed,
        )
        y_eff = np.array([train_ds[i][1] for i in range(len(train_ds))], dtype=int)
    else:
        train_ds = train_base
        y_eff = y_train

    sampler = _make_weighted_sampler(y_eff) if cfg.use_weighted_sampler else None
    train_ld = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )

    meta = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_df_raw": train_df_raw,
        "val_df_raw": val_df_raw,
        "test_df_raw": test_df_raw,
        "train_labels_raw": y_train,
        "train_labels_effective": y_eff,
        "counts_raw": counts,
        "quality_filter": {
            "train": train_q,
            "val": val_q,
            "test": test_q,
        },
        "train_horizon_strategy": {
            "enabled": use_train_horizon_strategy,
            "random_horizon": bool(cfg.train_random_horizon),
            "random_horizon_min_days": float(cfg.train_random_horizon_min_days),
            "random_horizon_power": float(cfg.train_random_horizon_power),
            "curriculum_epochs": int(cfg.train_curriculum_epochs),
            "curriculum_start_days": float(cfg.train_curriculum_start_days),
        },
    }
    return train_ld, val_ld, test_ld, stats, meta


def _cosine_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _batch_to_device(batch, device):
    if len(batch) == 3:
        xb, yb, mb = batch
        gb = None
    else:
        xb, yb, mb, gb = batch
    xb = xb.to(device)
    yb = yb.to(device)
    mb = mb.to(device)
    gb = gb.to(device) if gb is not None else None
    return xb, yb, mb, gb


def _binary_focus_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_id: int) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    c = int(class_id)

    tp = int(np.sum((y_true == c) & (y_pred == c)))
    fp = int(np.sum((y_true != c) & (y_pred == c)))
    fn = int(np.sum((y_true == c) & (y_pred != c)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def _resolve_focus_class_id(taxonomy: Taxonomy, class_name: str) -> Optional[int]:
    if not class_name:
        return None
    target = str(class_name).strip().lower()
    for i, name in enumerate(taxonomy.broad_classes):
        if str(name).strip().lower() == target:
            return int(i)
    return None


def _selection_score(metrics: Dict[str, float], metric_name: str) -> float:
    def _safe(v: float) -> float:
        x = float(v)
        return x if np.isfinite(x) else float("-inf")

    name = str(metric_name).strip().lower()
    if name == "macro_auprc":
        return _safe(metrics.get("macro_auprc", float("-inf")))
    if name == "macro_f1":
        return _safe(metrics.get("macro_f1", float("-inf")))
    if name == "balanced_accuracy":
        return _safe(metrics.get("balanced_accuracy", float("-inf")))
    if name == "composite_sota":
        macro_f1 = _safe(metrics.get("macro_f1", 0.0))
        bal_acc = _safe(metrics.get("balanced_accuracy", 0.0))
        focus_f1 = _safe(metrics.get("focus_f1", 0.0))
        focus_recall = _safe(metrics.get("focus_recall", 0.0))
        return float(0.45 * macro_f1 + 0.25 * bal_acc + 0.20 * focus_f1 + 0.10 * focus_recall)
    if name == "composite_prod":
        # Production-oriented checkpoint selection:
        # reward broad quality + focus-class quality while penalizing
        # calibration degradation (ECE/NLL) on validation.
        macro_f1 = _safe(metrics.get("macro_f1", 0.0))
        bal_acc = _safe(metrics.get("balanced_accuracy", 0.0))
        macro_auprc = _safe(metrics.get("macro_auprc", 0.0))
        focus_f1 = _safe(metrics.get("focus_f1", 0.0))
        focus_recall = _safe(metrics.get("focus_recall", 0.0))
        focus_precision = _safe(metrics.get("focus_precision", 0.0))
        ece = _safe(metrics.get("ece", 0.0))
        nll = _safe(metrics.get("nll", 0.0))
        return float(
            0.32 * macro_f1
            + 0.20 * bal_acc
            + 0.12 * macro_auprc
            + 0.14 * focus_f1
            + 0.08 * focus_recall
            + 0.06 * focus_precision
            - 0.12 * ece
            - 0.06 * nll
        )
    raise ValueError(f"Unknown checkpoint selection metric: {metric_name}")


def _run_epoch(
    model,
    loader,
    crit,
    *,
    device,
    epoch,
    optim=None,
    cfg: Optional[Config] = None,
    total_steps: int = 1,
    warmup_steps: int = 0,
    step: int = 0,
    ema: Optional[EMAState] = None,
    focus_class_id: Optional[int] = None,
    focus_class_name: Optional[str] = None,
):
    train_mode = optim is not None
    model.train(train_mode)

    sums = {
        "loss_total": 0.0,
        "loss_nll": 0.0,
        "loss_kl_unscaled": 0.0,
        "loss_kl_annealed": 0.0,
        "loss_suppressor": 0.0,
        "loss_margin": 0.0,
        "anneal_coef": 0.0,
    }
    n_tot = 0
    ys, ps, alphas = [], [], []

    for batch in loader:
        xb, yb, mb, gb = _batch_to_device(batch, device)

        if train_mode:
            lr_scale = _cosine_warmup(step, total_steps, warmup_steps)
            for pg in optim.param_groups:
                pg["lr"] = cfg.lr * lr_scale

        evidence = model(xb, mb, gb)
        loss, comp = crit(evidence, yb, epoch=epoch, return_components=True)

        if train_mode:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()
            if ema is not None:
                ema.update(model)

        with torch.no_grad():
            alpha = model.alpha_from_evidence(evidence)
            p = model.mean_from_alpha(alpha)
            if not torch.isfinite(alpha).all():
                raise RuntimeError("Alpha contains non-finite values")
            if not torch.isfinite(p).all():
                raise RuntimeError("Dirichlet mean probabilities contain non-finite values")
            if torch.any(alpha <= 1.0):
                raise RuntimeError("Alpha must be > 1 for evidential parameterization")
            p_sum = p.sum(dim=1)
            if not torch.allclose(p_sum, torch.ones_like(p_sum), atol=1e-4, rtol=1e-4):
                raise RuntimeError("Dirichlet mean probabilities must sum to 1")

            bs = yb.size(0)
            n_tot += bs
            sums["loss_total"] += float(comp.total.sum().item())
            sums["loss_nll"] += float(comp.nll.sum().item())
            sums["loss_kl_unscaled"] += float(comp.kl_unscaled.sum().item())
            sums["loss_kl_annealed"] += float(comp.kl_annealed.sum().item())
            sums["loss_suppressor"] += float(comp.suppressor.sum().item())
            sums["loss_margin"] += float(comp.margin.sum().item())
            sums["anneal_coef"] += float(comp.anneal_coef.sum().item())

            ys.append(yb.detach().cpu().numpy())
            ps.append(p.detach().cpu().numpy())
            alphas.append(alpha.detach().cpu().numpy())

        if train_mode:
            step += 1

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    a = np.concatenate(alphas)
    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite probabilities encountered after epoch aggregation")
    if not np.isfinite(a).all():
        raise RuntimeError("Non-finite alpha encountered after epoch aggregation")

    # Calibration metrics below are computed from Dirichlet mean probabilities p.
    metrics = compute_basic_metrics(y, p, p.shape[1])
    pred = np.argmax(p, axis=1).astype(int)
    alpha0 = a.sum(axis=1)
    C = a.shape[1]
    evidence = np.clip(alpha0 - C, a_min=0.0, a_max=None)

    u = C / np.maximum(alpha0, 1e-12)
    ent = predictive_entropy(torch.from_numpy(p)).numpy()
    mi = mutual_information(torch.from_numpy(a)).numpy()

    out = {
        # EDL objective decomposition (per-sample averages over epoch).
        "loss_total": sums["loss_total"] / max(1, n_tot),
        "loss_nll": sums["loss_nll"] / max(1, n_tot),
        "loss_expected_dirichlet_nll": sums["loss_nll"] / max(1, n_tot),
        "loss_kl_unscaled": sums["loss_kl_unscaled"] / max(1, n_tot),
        "loss_kl_unannealed": sums["loss_kl_unscaled"] / max(1, n_tot),
        "loss_kl_annealed": sums["loss_kl_annealed"] / max(1, n_tot),
        "loss_kl_annealed_contrib": sums["loss_kl_annealed"] / max(1, n_tot),
        "loss_suppressor": sums["loss_suppressor"] / max(1, n_tot),
        "loss_evidence_suppressor": sums["loss_suppressor"] / max(1, n_tot),
        "loss_margin": sums["loss_margin"] / max(1, n_tot),
        "anneal_coef": sums["anneal_coef"] / max(1, n_tot),
        "annealing_coef": sums["anneal_coef"] / max(1, n_tot),
        # Predictive performance.
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_auprc": float(metrics["auprc_macro"]),
        # Predictive calibration on mean probabilities.
        "ece": float(metrics["ece"]),
        "calib_ece": float(metrics["ece"]),
        "nll": float(metrics["nll"]),
        "calib_nll": float(metrics["nll"]),
        "brier": float(metrics["brier"]),
        "calib_brier": float(metrics["brier"]),
        # Uncertainty/evidence summaries.
        "mean_total_evidence": float(np.mean(evidence)),
        "mean_vacuity": float(np.mean(u)),
        "mean_predictive_entropy": float(np.mean(ent)),
        "mean_mutual_information": float(np.mean(mi)),
    }
    if focus_class_id is not None:
        focus = _binary_focus_metrics(y, pred, int(focus_class_id))
        out["focus_precision"] = float(focus["precision"])
        out["focus_recall"] = float(focus["recall"])
        out["focus_f1"] = float(focus["f1"])
        if focus_class_name:
            key = str(focus_class_name).strip().lower().replace(" ", "_")
            out[f"{key}_precision"] = float(focus["precision"])
            out[f"{key}_recall"] = float(focus["recall"])
            out[f"{key}_f1"] = float(focus["f1"])
    return out, step


def _plot_learning_curves(history_rows: list, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    setup_mpl_paper(usetex=True)

    def _v(split: str, key: str, fallback: Optional[str] = None):
        out = []
        for r in history_rows:
            d = r[split]
            if key in d:
                out.append(d[key])
            elif fallback is not None and fallback in d:
                out.append(d[fallback])
            else:
                out.append(float("nan"))
        return out

    ep = np.array([r["epoch"] for r in history_rows])

    with PdfPages(out_pdf) as pdf:
        fig, axs = plt.subplots(2, 1, figsize=(8.0, 7.0), dpi=300, sharex=True)
        ax0, ax1 = axs
        ax0.plot(ep, _v("train", "loss_total"), label="train total", color=NORD["nord10"])
        ax0.plot(ep, _v("val", "loss_total"), label="val total", color=NORD["nord9"])
        ax0.plot(ep, _v("train", "loss_expected_dirichlet_nll", fallback="loss_nll"), label=r"train $\mathbb{E}[\mathrm{NLL}_{Dir}]$", color=NORD["nord14"])
        ax0.plot(ep, _v("val", "loss_expected_dirichlet_nll", fallback="loss_nll"), label=r"val $\mathbb{E}[\mathrm{NLL}_{Dir}]$", color=NORD["nord7"])
        ax0.set_title("EDL Loss: Total and Expected Dirichlet NLL")
        ax0.set_ylabel("Loss")
        style_axes_inward(ax0)
        ax0.legend(ncol=2)

        ax1.plot(ep, _v("train", "loss_kl_unannealed", fallback="loss_kl_unscaled"), label=r"train KL (un-annealed)", color=NORD["nord11"])
        ax1.plot(ep, _v("val", "loss_kl_unannealed", fallback="loss_kl_unscaled"), label=r"val KL (un-annealed)", color=NORD["nord12"])
        ax1.plot(ep, _v("train", "loss_kl_annealed_contrib", fallback="loss_kl_annealed"), label=r"train $a(t)\lambda_{KL}\mathrm{KL}$", color=NORD["nord13"])
        ax1.plot(ep, _v("val", "loss_kl_annealed_contrib", fallback="loss_kl_annealed"), label=r"val $a(t)\lambda_{KL}\mathrm{KL}$", color=NORD["nord15"])
        ax1.plot(ep, _v("train", "loss_evidence_suppressor", fallback="loss_suppressor"), label="train suppressor", color=NORD["nord3"])
        ax1.plot(ep, _v("val", "loss_evidence_suppressor", fallback="loss_suppressor"), label="val suppressor", color=NORD["nord8"])
        ax1.plot(ep, _v("train", "loss_margin"), label="train margin", color=NORD["nord4"])
        ax1.plot(ep, _v("val", "loss_margin"), label="val margin", color=NORD["nord0"])
        ax1.plot(ep, _v("train", "annealing_coef", fallback="anneal_coef"), label=r"train annealing $a(t)$", color=NORD["nord6"], ls="--")
        ax1.plot(ep, _v("val", "annealing_coef", fallback="anneal_coef"), label=r"val annealing $a(t)$", color=NORD["nord2"], ls="--")
        ax1.set_title("KL/Suppressor Components and Annealing")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Value")
        style_axes_inward(ax1)
        ax1.legend(ncol=2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2), dpi=300)
        ax.plot(ep, _v("train", "macro_auprc"), label="train macro-AUPRC")
        ax.plot(ep, _v("val", "macro_auprc"), label="val macro-AUPRC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro-AUPRC")
        ax.set_title("Macro-AUPRC vs Epoch")
        style_axes_inward(ax, grid_y=True)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2), dpi=300)
        ax.plot(ep, _v("val", "calib_ece", fallback="ece"), label="ECE")
        ax.plot(ep, _v("val", "calib_nll", fallback="nll"), label="NLL")
        ax.plot(ep, _v("val", "calib_brier", fallback="brier"), label="Brier")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title("Validation Calibration (Dirichlet Mean Probabilities)")
        style_axes_inward(ax, grid_y=True)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.4), dpi=300)
        ax.plot(ep, _v("train", "mean_total_evidence"), label=r"train $\bar{E}=\overline{\alpha_0-C}$")
        ax.plot(ep, _v("val", "mean_total_evidence"), label=r"val $\bar{E}=\overline{\alpha_0-C}$")
        ax.plot(ep, _v("train", "mean_vacuity"), label=r"train vacuity $u=C/\sum\alpha$")
        ax.plot(ep, _v("val", "mean_vacuity"), label=r"val vacuity $u=C/\sum\alpha$")
        ax.plot(ep, _v("train", "mean_predictive_entropy"), label="train entropy")
        ax.plot(ep, _v("val", "mean_predictive_entropy"), label="val entropy")
        ax.plot(ep, _v("train", "mean_mutual_information"), label="train MI")
        ax.plot(ep, _v("val", "mean_mutual_information"), label="val MI")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Statistic")
        ax.set_title("Evidence and Uncertainty")
        style_axes_inward(ax)
        ax.legend(ncol=2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def train_evidential(
    cfg: Config,
    *,
    taxonomy: Taxonomy = DEFAULT_TAXONOMY,
    pretrained_encoder_ckpt: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[EvidentialClassifier, TrainHistory, Dict]:
    set_seed(cfg.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_ld, val_ld, test_ld, stats, meta = make_loaders(cfg, taxonomy)

    enc = EventTransformerEncoder(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        band_mode=cfg.band_mode,
        band_embed_dim=cfg.band_embed_dim,
        time_encoding=cfg.time_encoding,
    )

    model = EvidentialClassifier(
        enc,
        num_classes=taxonomy.num_classes,
        dropout=cfg.dropout,
        pool=cfg.pool,
        use_global_features=cfg.use_global_features,
        global_dim=global_feature_dim(cfg.global_feature_set),
        global_hidden_dim=cfg.global_hidden_dim,
    ).to(device)

    if pretrained_encoder_ckpt is not None:
        sd = torch.load(pretrained_encoder_ckpt, map_location="cpu")
        missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
        (cfg.out_dir / "pretrained_load.json").write_text(json.dumps({"missing": missing, "unexpected": unexpected}, indent=2))

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_ld)
    warmup_steps = int(0.05 * total_steps)

    cw = None
    if cfg.use_class_balanced_loss:
        cw = _compute_effective_num_weights(meta["train_labels_raw"], taxonomy.num_classes, cfg.class_balance_beta)

    crit = EvidentialDirichletLoss(
        num_classes=taxonomy.num_classes,
        anneal_epochs=cfg.edl_anneal_epochs,
        kl_strength=cfg.edl_kl_strength,
        sup_strength=cfg.edl_sup_strength,
        label_smoothing=cfg.label_smoothing,
        focal_gamma=cfg.focal_gamma,
        class_weights=cw,
        margin_strength=cfg.margin_loss_strength,
        margin_delta=cfg.margin_delta,
        margin_pairs=cfg.margin_pairs,
    )
    crit = crit.to(device)

    focus_class_id = _resolve_focus_class_id(taxonomy, cfg.ckpt_select_focus_class)
    best = {
        "epoch": -1,
        "va_loss": float("inf"),
        "va_auprc_macro": -1.0,
        "va_select_score": float("-inf"),
        "selection_metric": cfg.ckpt_select_metric,
        "selection_focus_class": cfg.ckpt_select_focus_class,
        "used_ema": False,
    }
    history_rows = []

    ema = EMAState(model, decay=cfg.ema_decay) if cfg.use_ema else None

    bad = 0
    step = 0
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        if hasattr(train_ld.dataset, "set_epoch"):
            train_ld.dataset.set_epoch(ep)
        tr, step = _run_epoch(
            model,
            train_ld,
            crit,
            device=device,
            epoch=ep,
            optim=optim,
            cfg=cfg,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            step=step,
            ema=ema,
            focus_class_id=focus_class_id,
            focus_class_name=cfg.ckpt_select_focus_class,
        )

        va, step = _run_epoch(
            model,
            val_ld,
            crit,
            device=device,
            epoch=ep,
            step=step,
            focus_class_id=focus_class_id,
            focus_class_name=cfg.ckpt_select_focus_class,
        )

        eval_variant = "model"
        va_for_select = va
        score_model = _selection_score(va, cfg.ckpt_select_metric)
        score_for_select = score_model
        if ema is not None:
            ema.apply(model)
            va_ema, _ = _run_epoch(
                model,
                val_ld,
                crit,
                device=device,
                epoch=ep,
                step=step,
                focus_class_id=focus_class_id,
                focus_class_name=cfg.ckpt_select_focus_class,
            )
            ema.restore(model)
            score_ema = _selection_score(va_ema, cfg.ckpt_select_metric)
            if score_ema > score_model:
                va_for_select = va_ema
                eval_variant = "ema"
                score_for_select = score_ema

        improved = (score_for_select > best["va_select_score"] + 1e-6) or (
            abs(score_for_select - best["va_select_score"]) < 1e-6
            and va_for_select["loss_total"] < best["va_loss"]
        )

        if improved:
            best.update(
                {
                    "epoch": ep,
                    "va_loss": float(va_for_select["loss_total"]),
                    "va_auprc_macro": float(va_for_select["macro_auprc"]),
                    "va_select_score": float(score_for_select),
                    "used_ema": eval_variant == "ema",
                }
            )
            if eval_variant == "ema" and ema is not None:
                ema.apply(model)
                torch.save(model.state_dict(), cfg.out_dir / "best_evidential.pt")
                torch.save(model.state_dict(), cfg.out_dir / "best_evidential_ema.pt")
                ema.restore(model)
            else:
                torch.save(model.state_dict(), cfg.out_dir / "best_evidential.pt")
            bad = 0
        else:
            bad += 1

        row = {
            "epoch": ep,
            "train": tr,
            "val": va,
            "val_selected": va_for_select,
            "selected_variant": eval_variant,
            "selection_metric": cfg.ckpt_select_metric,
            "selection_score": float(score_for_select),
            "lr": cfg.lr * _cosine_warmup(step, total_steps, warmup_steps),
            "elapsed_min": (time.time() - t0) / 60.0,
            "best": best,
        }
        history_rows.append(row)
        (cfg.out_dir / "train_log.jsonl").open("a", encoding="utf-8").write(json.dumps(row) + "\n")

        if bad >= cfg.patience:
            break

    model.load_state_dict(torch.load(cfg.out_dir / "best_evidential.pt", map_location=device))

    history = TrainHistory(rows=history_rows)
    (cfg.out_dir / "history.json").write_text(json.dumps(history.to_dict(), indent=2))
    (cfg.out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (cfg.out_dir / "seeds.json").write_text(json.dumps({"seed": cfg.seed}, indent=2))

    report_dir = cfg.out_dir / "report"
    _plot_learning_curves(history_rows, report_dir / "learning_curves.pdf")

    return model, history, {
        "best": best,
        "device": str(device),
        "train_counts_raw": meta["counts_raw"].tolist(),
        "quality_filter": meta.get("quality_filter", {}),
    }
