from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score

from .calibration import EvidenceTemperatureScaler, EvidenceVectorScaler
from .config import Config
from .data import filter_manifest_quality, global_feature_dim, preprocess_photometry_array, read_manifest_csv
from .evaluate import compute_basic_metrics, plot_abstention_overview, write_report_tables
from .plotting import NORD, marker_colors, savefig_pdf, setup_mpl_paper, style_axes_inward
from .postprocess import (
    adjust_for_priors,
    aurc,
    fit_prob_thresholds,
    pick_uncertainty_cut,
    predict_with_thresholds,
    risk_coverage_curve,
)
from .taxonomy import DEFAULT_TAXONOMY, Taxonomy
from .uncertainty import (
    alpha_from_evidence,
    dirichlet_mean,
    dirichlet_std,
    expected_categorical_entropy,
    mutual_information,
    predictive_entropy,
    vacuity,
    total_uncertainty_trace,
)

BAND_NAME = {0: "g", 1: "r", 2: "i"}
BAND_COLOR = {0: marker_colors["ztfg"], 1: marker_colors["ztfr"], 2: marker_colors["ztfi"]}


@torch.no_grad()
def collect_evidential(model, loader, *, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    ys, evids, alphas, probs = [], [], [], []
    u_list, ent_list, exp_ent_list, mi_list, conf_list = [], [], [], [], []
    alpha0_list, tevid_list, trU_list, pstd_list = [], [], [], []

    for batch in loader:
        if len(batch) == 3:
            xb, yb, mb = batch
            gb = None
        else:
            xb, yb, mb, gb = batch
        xb = xb.to(device)
        mb = mb.to(device)
        gb = gb.to(device) if gb is not None else None

        evidence = model(xb, mb, gb)
        alpha = alpha_from_evidence(evidence)
        p = dirichlet_mean(alpha)

        u = vacuity(alpha)
        ent = predictive_entropy(p)
        exp_ent = expected_categorical_entropy(alpha)
        mi = mutual_information(alpha)
        conf = p.max(dim=1).values
        alpha0 = alpha.sum(dim=1)
        tevid = torch.clamp(alpha0 - alpha.size(1), min=0.0)
        trU = total_uncertainty_trace(alpha)
        pstd = dirichlet_std(alpha)

        ys.append(yb.numpy())
        evids.append(evidence.cpu().numpy())
        alphas.append(alpha.cpu().numpy())
        probs.append(p.cpu().numpy())
        u_list.append(u.cpu().numpy())
        ent_list.append(ent.cpu().numpy())
        exp_ent_list.append(exp_ent.cpu().numpy())
        mi_list.append(mi.cpu().numpy())
        conf_list.append(conf.cpu().numpy())
        alpha0_list.append(alpha0.cpu().numpy())
        tevid_list.append(tevid.cpu().numpy())
        trU_list.append(trU.cpu().numpy())
        pstd_list.append(pstd.cpu().numpy())

    return {
        "y": np.concatenate(ys),
        "evidence": np.concatenate(evids),
        "alpha": np.concatenate(alphas),
        "probs": np.concatenate(probs),
        "vacuity": np.concatenate(u_list),
        "entropy": np.concatenate(ent_list),
        "expected_entropy": np.concatenate(exp_ent_list),
        "mi": np.concatenate(mi_list),
        "confidence": np.concatenate(conf_list),
        "alpha0": np.concatenate(alpha0_list),
        "total_evidence": np.concatenate(tevid_list),
        "trace_uncertainty": np.concatenate(trU_list),
        "p_std": np.concatenate(pstd_list),
    }


def _plot_split_hist(values: np.ndarray, correct: np.ndarray, *, xlabel: str, title: str) -> plt.Figure:
    setup_mpl_paper(usetex=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)
    ax.hist(values[correct], bins=40, alpha=0.75, label="correct", color=NORD["nord14"], edgecolor=NORD["nord0"], linewidth=1.2, density=True)
    ax.hist(values[~correct], bins=40, alpha=0.75, label="incorrect", color=NORD["nord11"], edgecolor=NORD["nord0"], linewidth=1.2, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    style_axes_inward(ax)
    fig.tight_layout()
    return fig


def _safe_auroc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return float("nan")


def _broad_train_prior_from_manifest(cfg: Config, taxonomy: Taxonomy) -> np.ndarray:
    manifest_dir = Path(cfg.manifest_dir) if cfg.manifest_dir is not None else Path(cfg.data_dir)
    df_raw = read_manifest_csv(manifest_dir / "manifest_train.csv", data_dir=cfg.data_dir, path_prefix=cfg.path_prefix)
    needs_quality_filter = bool(
        cfg.drop_i_band
        or cfg.min_obs_total > 0
        or cfg.min_obs_g > 0
        or cfg.min_obs_r > 0
        or cfg.min_obs_i > 0
        or cfg.min_bands_observed > 0
    )
    if needs_quality_filter:
        df, _ = filter_manifest_quality(
            df_raw,
            horizon_days=cfg.horizon_days,
            drop_i_band=cfg.drop_i_band,
            min_obs_total=cfg.min_obs_total,
            min_obs_g=cfg.min_obs_g,
            min_obs_r=cfg.min_obs_r,
            min_obs_i=cfg.min_obs_i,
            min_bands_observed=cfg.min_bands_observed,
        )
    else:
        df = df_raw
    if len(df) == 0:
        raise RuntimeError("No train samples left after quality cuts when computing train prior.")
    y_sub = df["label"].astype(int).values
    id2b = taxonomy.id2broad_id
    y = np.array([id2b[int(z)] for z in y_sub], dtype=int)
    C = taxonomy.num_classes
    return np.bincount(y, minlength=C) / max(1, len(y))


def _stratified_bootstrap(y_true: np.ndarray, n_boot: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    idx_by_class = {c: np.where(y_true == c)[0] for c in np.unique(y_true)}
    N = len(y_true)
    out = np.zeros((n_boot, N), dtype=int)
    for b in range(n_boot):
        chunks = []
        for _, idxs in idx_by_class.items():
            chunks.append(rng.choice(idxs, size=len(idxs), replace=True))
        sample = np.concatenate(chunks)
        rng.shuffle(sample)
        out[b] = sample
    return out


def _bootstrap_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: Sequence[str], n_boot: int = 500, seed: int = 123) -> Dict:
    C = probs.shape[1]
    bt_idx = _stratified_bootstrap(y_true, n_boot=n_boot, seed=seed)
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "auprc_macro", "ece", "nll"]
    vals = {k: [] for k in keys}
    per_cls = {name: {"precision": [], "recall": []} for name in class_names}

    for s in bt_idx:
        m = compute_basic_metrics(y_true[s], probs[s], C)
        for k in keys:
            vals[k].append(float(m[k]))

        pred = probs[s].argmax(axis=1)
        rep = classification_report(
            y_true[s],
            pred,
            labels=list(range(len(class_names))),
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        )
        for n in class_names:
            per_cls[n]["precision"].append(float(rep[n]["precision"]))
            per_cls[n]["recall"].append(float(rep[n]["recall"]))

    out = {"metrics": {}, "per_class": {}}
    for k, v in vals.items():
        arr = np.asarray(v)
        out["metrics"][k] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }

    for n in class_names:
        out["per_class"][n] = {}
        for kk in ["precision", "recall"]:
            arr = np.asarray(per_cls[n][kk])
            out["per_class"][n][kk] = {
                "mean": float(arr.mean()),
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5)),
            }
    return out


def _write_bootstrap_md(boot: Dict, out_path: Path) -> None:
    lines = ["# Bootstrap 95% CI", "", "## Global Metrics", "", "| metric | mean | 95% CI |", "|---|---:|---:|"]
    for k, d in boot["metrics"].items():
        lines.append(f"| {k} | {d['mean']:.4f} | [{d['ci95_low']:.4f}, {d['ci95_high']:.4f}] |")

    lines += ["", "## Per-Class Precision/Recall", "", "| class | precision mean | precision 95% CI | recall mean | recall 95% CI |", "|---|---:|---:|---:|---:|"]
    for cls, d in boot["per_class"].items():
        p = d["precision"]
        r = d["recall"]
        lines.append(
            f"| {cls} | {p['mean']:.4f} | [{p['ci95_low']:.4f}, {p['ci95_high']:.4f}] | {r['mean']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] |"
        )
    out_path.write_text("\n".join(lines))


def _uncertainty_score_dict(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "vacuity": d["vacuity"],
        "entropy": d["entropy"],
        "expected_entropy": d["expected_entropy"],
        "mi": d["mi"],
        "trace_uncertainty": d["trace_uncertainty"],
    }


def _robust_loc_scale(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-8:
        std = float(np.std(x))
        scale = std if np.isfinite(std) and std > 1e-8 else 1.0
    return med, scale


def _fit_uncertainty_fusion(
    score_map_val: Dict[str, np.ndarray],
    y_val: np.ndarray,
    pred_val: np.ndarray,
    *,
    base_keys: Optional[Sequence[str]] = None,
) -> Dict:
    keys = list(base_keys) if base_keys is not None else ["vacuity", "entropy", "expected_entropy", "mi", "trace_uncertainty"]
    err = (np.asarray(pred_val, dtype=int) != np.asarray(y_val, dtype=int)).astype(int)
    medians: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    auroc_map: Dict[str, float] = {}

    for k in keys:
        u = np.asarray(score_map_val[k], dtype=float)
        med, sc = _robust_loc_scale(u)
        au = _safe_auroc_binary(err, u)
        w = max(0.0, float(au) - 0.5) if np.isfinite(au) else 0.0
        medians[k] = med
        scales[k] = sc
        auroc_map[k] = float(au)
        weights[k] = float(w)

    wsum = float(sum(weights.values()))
    if wsum <= 1e-12:
        weights = {k: 1.0 for k in keys}

    return {
        "base_keys": keys,
        "medians": medians,
        "scales": scales,
        "weights": weights,
        "auroc_error_val": auroc_map,
        "rule": "weighted_robust_zsum",
        "rule_note": "fused = sum_k w_k * ((u_k - median_k)/scale_k) / sum_k w_k, where w_k=max(AUROC(error|u_k)-0.5,0).",
    }


def _apply_uncertainty_fusion(score_map: Dict[str, np.ndarray], fusion: Dict) -> np.ndarray:
    keys = list(fusion["base_keys"])
    zsum = np.zeros_like(np.asarray(score_map[keys[0]], dtype=float), dtype=float)
    wsum = 0.0
    for k in keys:
        u = np.asarray(score_map[k], dtype=float)
        med = float(fusion["medians"][k])
        sc = float(fusion["scales"][k])
        w = float(fusion["weights"][k])
        zsum += w * ((u - med) / max(sc, 1e-8))
        wsum += w
    if wsum <= 1e-12:
        return zsum
    return zsum / wsum


def _fit_ood_reference(
    score_map_val: Dict[str, np.ndarray],
    *,
    inlier_mask: np.ndarray,
    feature_keys: Sequence[str],
    score_quantile: float = 0.995,
    vote_quantile: float = 0.995,
    min_votes: int = 2,
) -> Dict:
    keys = list(feature_keys)
    X = np.column_stack([np.asarray(score_map_val[k], dtype=float) for k in keys])
    inlier = np.asarray(inlier_mask, dtype=bool)
    if inlier.sum() < 30:
        inlier = np.ones(X.shape[0], dtype=bool)

    Xin = X[inlier]
    center = np.median(Xin, axis=0)
    mad = np.median(np.abs(Xin - center[None, :]), axis=0)
    scale = 1.4826 * mad
    std = np.std(Xin, axis=0)
    scale = np.where((np.isfinite(scale)) & (scale > 1e-8), scale, np.where((np.isfinite(std)) & (std > 1e-8), std, 1.0))

    z_val = (X - center[None, :]) / scale[None, :]
    score_val = np.sqrt(np.sum(z_val ** 2, axis=1))
    ref_score = score_val[inlier] if inlier.any() else score_val
    score_thr = float(np.quantile(ref_score, float(np.clip(score_quantile, 0.5, 0.9999))))

    per_feature_thr = {}
    for k in keys:
        u = np.asarray(score_map_val[k], dtype=float)
        ur = u[inlier] if inlier.any() else u
        per_feature_thr[k] = float(np.quantile(ur, float(np.clip(vote_quantile, 0.5, 0.9999))))

    return {
        "feature_keys": keys,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "score_threshold": score_thr,
        "feature_thresholds": per_feature_thr,
        "min_votes": int(max(1, min_votes)),
        "score_quantile": float(score_quantile),
        "vote_quantile": float(vote_quantile),
    }


def _apply_ood_reference(score_map: Dict[str, np.ndarray], ref: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = list(ref["feature_keys"])
    X = np.column_stack([np.asarray(score_map[k], dtype=float) for k in keys])
    center = np.asarray(ref["center"], dtype=float)
    scale = np.asarray(ref["scale"], dtype=float)
    z = (X - center[None, :]) / np.maximum(scale[None, :], 1e-8)
    score = np.sqrt(np.sum(z ** 2, axis=1))

    votes = np.zeros(X.shape[0], dtype=int)
    for k in keys:
        u = np.asarray(score_map[k], dtype=float)
        votes += (u > float(ref["feature_thresholds"][k])).astype(int)

    flag = (score > float(ref["score_threshold"])) | (votes >= int(ref["min_votes"]))
    return score, votes, flag


def _plot_uncertainty_separation(
    out_pdf: Path,
    *,
    val_y: np.ndarray,
    val_pred: np.ndarray,
    val_scores: Dict[str, np.ndarray],
    test_y: np.ndarray,
    test_pred: np.ndarray,
    test_scores: Dict[str, np.ndarray],
    val_quantile_thresholds: Dict[str, float],
    chosen_score: str,
    chosen_threshold: float,
) -> None:
    setup_mpl_paper(usetex=True)
    metrics = list(val_scores.keys())
    n = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    with PdfPages(out_pdf) as pdf:
        for split_name, y, pred, scores in [
            ("val", val_y, val_pred, val_scores),
            ("test", test_y, test_pred, test_scores),
        ]:
            fig, axs = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.3 * nrows), dpi=300)
            axs = np.atleast_1d(axs).ravel()
            err = (np.asarray(pred, dtype=int) != np.asarray(y, dtype=int)).astype(int)

            for i, k in enumerate(metrics):
                ax = axs[i]
                u = np.asarray(scores[k], dtype=float)
                m_corr = err == 0
                m_err = err == 1
                ax.hist(u[m_corr], bins=40, alpha=0.6, density=True, color=NORD["nord14"], label="correct")
                ax.hist(u[m_err], bins=40, alpha=0.6, density=True, color=NORD["nord11"], label="misclassified")
                qthr = float(val_quantile_thresholds[k])
                ax.axvline(qthr, color=NORD["nord0"], ls="--", lw=1.2, label="val q-threshold")
                if (k == chosen_score) and np.isfinite(chosen_threshold):
                    ax.axvline(float(chosen_threshold), color=NORD["nord10"], ls="-", lw=1.3, label="selected threshold")
                au = _safe_auroc_binary(err, u)
                ax.set_title(f"{k} | AUROC(error|u)={au:.3f}")
                ax.set_xlabel("uncertainty score")
                ax.set_ylabel("density")
                style_axes_inward(ax, grid_y=True)
                ax.legend(fontsize=7)

            for j in range(n, len(axs)):
                axs[j].axis("off")

            fig.suptitle(f"Uncertainty Separation ({split_name})", y=0.995)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
            pdf.savefig(fig)
            plt.close(fig)


def _build_split_rows(
    split_df,
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    pred_thr: np.ndarray,
    keep_mask: np.ndarray,
    score_map: Dict[str, np.ndarray],
    class_names: Sequence[str],
    ood_score: Optional[np.ndarray] = None,
    ood_votes: Optional[np.ndarray] = None,
    ood_flag: Optional[np.ndarray] = None,
) -> List[Dict]:
    sdf = split_df.reset_index(drop=True)
    if len(sdf) != len(y_true):
        raise RuntimeError(f"Split dataframe length {len(sdf)} does not match predictions length {len(y_true)}")

    rows: List[Dict] = []
    for i in range(len(sdf)):
        y = int(y_true[i])
        p = np.asarray(probs[i], dtype=float)
        pred = int(pred_thr[i])
        top = np.argsort(-p)[:3]
        sr = sdf.iloc[i]
        obj = sr.get("obj_id", f"row_{i}")
        row = {
            "row_index": int(i),
            "obj_id": str(obj),
            "filepath": str(sr.filepath),
            "true_id": y,
            "pred_id": pred,
            "true_name": str(class_names[y]),
            "pred_name": str(class_names[pred]),
            "correct": bool(pred == y),
            "abstained": bool(not keep_mask[i]),
            "confidence": float(np.max(p)),
            "top1_prob": float(p[top[0]]),
            "top2_prob": float(p[top[1]]) if len(top) > 1 else float("nan"),
            "margin_top1_top2": float((p[top[0]] - p[top[1]]) if len(top) > 1 else float("nan")),
            "true_prob": float(p[y]),
            "top_classes": [str(class_names[int(top[j])]) for j in range(len(top))],
            "top_probs": [float(p[int(top[j])]) for j in range(len(top))],
        }
        for k, v in score_map.items():
            row[k] = float(np.asarray(v, dtype=float)[i])
        if ood_score is not None:
            row["ood_score"] = float(ood_score[i])
        if ood_votes is not None:
            row["ood_votes"] = int(ood_votes[i])
        if ood_flag is not None:
            row["ood_flag"] = bool(ood_flag[i])
        rows.append(row)
    return rows


def _select_stratified_misclassified(
    rows: Sequence[Dict],
    *,
    uncertainty_key: str,
    max_per_stratum: int = 6,
) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    mis = [r for r in rows if not r["correct"]]
    if len(mis) == 0:
        return [], {}

    conf = np.asarray([r["confidence"] for r in mis], dtype=float)
    unc = np.asarray([r[uncertainty_key] for r in mis], dtype=float)
    q1, q2 = np.quantile(conf, [0.33, 0.66])
    uq = float(np.quantile(unc, 0.50))

    def _conf_bin(x: float) -> str:
        if x < q1:
            return "low"
        if x < q2:
            return "mid"
        return "high"

    def _unc_bin(x: float) -> str:
        return "high" if x >= uq else "low"

    strata: Dict[str, List[Dict]] = {}
    for r in mis:
        abst = "abstained" if r["abstained"] else "kept"
        st = f"{abst}|conf_{_conf_bin(float(r['confidence']))}|unc_{_unc_bin(float(r[uncertainty_key]))}"
        strata.setdefault(st, []).append(r)

    selected: List[Dict] = []
    summary: Dict[str, Dict[str, int]] = {}
    for st in sorted(strata.keys()):
        rs = list(strata[st])
        if st.startswith("kept"):
            rs = sorted(rs, key=lambda r: (-float(r["confidence"]), float(r[uncertainty_key]), int(r["row_index"])))
        else:
            rs = sorted(rs, key=lambda r: (-float(r[uncertainty_key]), -float(r["confidence"]), int(r["row_index"])))
        take = rs[: int(max(1, max_per_stratum))]
        selected.extend(take)
        summary[st] = {"count": int(len(rs)), "selected": int(len(take))}
    return selected, summary


def _plot_light_curve(ax, *, filepath: str, horizon_days: float, drop_i_band: bool) -> None:
    raw = np.load(filepath, allow_pickle=False)
    arr = raw["data"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
    arr = preprocess_photometry_array(arr, horizon_days=horizon_days, drop_i_band=drop_i_band, allow_empty=False)

    t = arr[:, 0]
    band = arr[:, 2].astype(int)
    flux = np.exp(arr[:, 3])
    ferr = np.exp(arr[:, 4])
    for b in [0, 1, 2]:
        m = band == b
        if not np.any(m):
            continue
        ax.errorbar(
            t[m],
            flux[m],
            yerr=ferr[m],
            fmt="o",
            ms=3.0,
            lw=0.8,
            alpha=0.85,
            color=BAND_COLOR[b],
            label=BAND_NAME[b],
        )
    ax.set_xlabel("time since first detection (days)")
    ax.set_ylabel("flux")
    style_axes_inward(ax, grid_y=True)
    ax.legend(loc="upper right", fontsize=7, ncol=3)


def _plot_misclassification_gallery(
    out_pdf: Path,
    *,
    rows: Sequence[Dict],
    class_names: Sequence[str],
    uncertainty_key: str,
    uncertainty_thresholds: Dict[str, float],
    horizon_days: float,
    drop_i_band: bool,
) -> None:
    setup_mpl_paper(usetex=True)
    metrics = [k for k in ["vacuity", "entropy", "expected_entropy", "mi", "trace_uncertainty", "fused_uncertainty"] if k in uncertainty_thresholds]

    with PdfPages(out_pdf) as pdf:
        if len(rows) == 0:
            fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
            ax.axis("off")
            ax.text(0.5, 0.5, "No misclassified samples selected for gallery.", ha="center", va="center")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            return
        for r in rows:
            fig = plt.figure(figsize=(11.0, 8.0), dpi=300)
            gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.25, 1.0], hspace=0.33, wspace=0.25)
            ax_lc = fig.add_subplot(gs[0, :])
            ax_prob = fig.add_subplot(gs[1, 0])
            ax_unc = fig.add_subplot(gs[1, 1])

            try:
                _plot_light_curve(ax_lc, filepath=str(r["filepath"]), horizon_days=horizon_days, drop_i_band=drop_i_band)
            except Exception as exc:
                ax_lc.axis("off")
                ax_lc.text(0.01, 0.95, f"Failed to load light curve:\n{exc}", va="top", ha="left", fontsize=9, transform=ax_lc.transAxes)

            probs = np.asarray(r["top_probs"], dtype=float)
            names = list(r["top_classes"])
            colors = [NORD["nord11"] if names[i] == r["pred_name"] else NORD["nord9"] for i in range(len(names))]
            ax_prob.barh(np.arange(len(names)), probs, color=colors, alpha=0.85)
            ax_prob.set_yticks(np.arange(len(names)))
            ax_prob.set_yticklabels(names)
            ax_prob.invert_yaxis()
            ax_prob.set_xlim(0.0, 1.0)
            ax_prob.set_xlabel("probability")
            ax_prob.set_title("Top Predicted Classes")
            style_axes_inward(ax_prob, grid_y=True)

            unc_vals = np.asarray([float(r[k]) for k in metrics], dtype=float)
            unc_thr = np.asarray([float(uncertainty_thresholds[k]) for k in metrics], dtype=float)
            unc_color = [NORD["nord11"] if unc_vals[i] > unc_thr[i] else NORD["nord14"] for i in range(len(metrics))]
            ax_unc.barh(np.arange(len(metrics)), unc_vals, color=unc_color, alpha=0.85)
            ax_unc.scatter(unc_thr, np.arange(len(metrics)), marker="|", color=NORD["nord0"], s=170, label="val q-threshold")
            ax_unc.set_yticks(np.arange(len(metrics)))
            ax_unc.set_yticklabels(metrics)
            ax_unc.invert_yaxis()
            ax_unc.set_xlabel("uncertainty score")
            ax_unc.set_title("Uncertainty Snapshot")
            style_axes_inward(ax_unc, grid_y=True)
            ax_unc.legend(fontsize=7, loc="lower right")

            ood_txt = ""
            if "ood_flag" in r:
                ood_txt = f" | ood_flag={bool(r['ood_flag'])} | ood_score={float(r.get('ood_score', float('nan'))):.3f}"
            fig.suptitle(
                f"{r['obj_id']} | true={r['true_name']} pred={r['pred_name']} | conf={r['confidence']:.3f} | "
                f"{uncertainty_key}={float(r[uncertainty_key]):.3f} | abstained={bool(r['abstained'])}{ood_txt}",
                y=0.995,
            )
            fig.subplots_adjust(top=0.92, hspace=0.33, wspace=0.25)
            pdf.savefig(fig)
            plt.close(fig)


def _write_rows_csv(rows: Sequence[Dict], out_csv: Path, *, fieldnames: Sequence[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            wr = {k: r.get(k, None) for k in fieldnames}
            if isinstance(wr.get("top_classes"), list):
                wr["top_classes"] = "|".join([str(x) for x in wr["top_classes"]])
            if isinstance(wr.get("top_probs"), list):
                wr["top_probs"] = "|".join([f"{float(x):.6f}" for x in wr["top_probs"]])
            w.writerow(wr)


def _write_misclassification_summary_md(
    out_md: Path,
    *,
    rows_test: Sequence[Dict],
    selected_summary: Dict[str, Dict[str, int]],
    uncertainty_key: str,
) -> Dict:
    mis = [r for r in rows_test if not r["correct"]]
    kept_mis = [r for r in mis if not r["abstained"]]
    abst_mis = [r for r in mis if r["abstained"]]
    pair_counts: Dict[str, int] = {}
    pair_counts_kept: Dict[str, int] = {}

    for r in mis:
        k = f"{r['true_name']}->{r['pred_name']}"
        pair_counts[k] = pair_counts.get(k, 0) + 1
    for r in kept_mis:
        k = f"{r['true_name']}->{r['pred_name']}"
        pair_counts_kept[k] = pair_counts_kept.get(k, 0) + 1

    top_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_pairs_kept = sorted(pair_counts_kept.items(), key=lambda kv: kv[1], reverse=True)[:10]

    lines = [
        "# Misclassification Summary",
        "",
        f"- total test samples: {len(rows_test)}",
        f"- total misclassified: {len(mis)}",
        f"- misclassified kept (not abstained): {len(kept_mis)}",
        f"- misclassified abstained: {len(abst_mis)}",
        f"- uncertainty key used for stratification: `{uncertainty_key}`",
        "",
        "## Stratified Gallery Counts",
        "",
        "| stratum | count | selected |",
        "|---|---:|---:|",
    ]
    for st, d in sorted(selected_summary.items()):
        lines.append(f"| {st} | {d['count']} | {d['selected']} |")

    lines += ["", "## Top Confusion Pairs (All Misclassified)", "", "| pair | count |", "|---|---:|"]
    for p, c in top_pairs:
        lines.append(f"| {p} | {c} |")

    lines += ["", "## Top Confusion Pairs (Kept Misclassified)", "", "| pair | count |", "|---|---:|"]
    for p, c in top_pairs_kept:
        lines.append(f"| {p} | {c} |")

    if len(kept_mis) > 0:
        hi_conf_kept = sorted(kept_mis, key=lambda r: -float(r["confidence"]))[:15]
        lines += ["", "## High-Confidence Kept Errors (Top 15)", "", "| obj_id | true | pred | conf | uncertainty | ood_flag |", "|---|---|---|---:|---:|---|"]
        for r in hi_conf_kept:
            lines.append(
                f"| {r['obj_id']} | {r['true_name']} | {r['pred_name']} | {float(r['confidence']):.3f} | {float(r[uncertainty_key]):.3f} | {bool(r.get('ood_flag', False))} |"
            )

    out_md.write_text("\n".join(lines))
    return {
        "num_total": int(len(rows_test)),
        "num_misclassified": int(len(mis)),
        "num_misclassified_kept": int(len(kept_mis)),
        "num_misclassified_abstained": int(len(abst_mis)),
        "strata": selected_summary,
        "top_pairs": [{"pair": p, "count": int(c)} for p, c in top_pairs],
        "top_pairs_kept": [{"pair": p, "count": int(c)} for p, c in top_pairs_kept],
    }


def _choose_uncertainty_gate(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    score_map: Dict[str, np.ndarray],
    *,
    objective: str,
    target_coverage: float,
    beta: float,
    labels: np.ndarray,
    strategy: str = "global",
    min_true_class_coverage: Optional[Dict[int, float]] = None,
) -> Dict:
    if strategy not in {"global", "classwise_pred"}:
        raise ValueError(f"Unknown uncertainty gate strategy: {strategy}")
    best = None
    error = (pred_val != y_val).astype(int)
    C = int(len(labels))
    min_true_class_coverage = dict(min_true_class_coverage or {})

    def _global_keep(u: np.ndarray, ut: float) -> np.ndarray:
        return u <= ut

    def _classwise_pred_keep(u: np.ndarray, pred: np.ndarray, cov_target: float) -> Tuple[np.ndarray, Dict[str, float]]:
        keep = np.zeros_like(u, dtype=bool)
        thrs: Dict[str, float] = {}
        q = float(np.clip(cov_target, 0.0, 1.0))
        for c in range(C):
            m = pred == c
            if not np.any(m):
                thrs[str(c)] = float("inf")
                continue
            uc = u[m]
            ut_c = float(np.quantile(uc, q))
            keep[m] = uc <= ut_c
            thrs[str(c)] = ut_c
        return keep, thrs

    def _coverage_violations_by_true_class(y_true: np.ndarray, keep_mask: np.ndarray) -> Dict[str, Dict[str, float]]:
        violations: Dict[str, Dict[str, float]] = {}
        if not min_true_class_coverage:
            return violations
        y_true = np.asarray(y_true, dtype=int)
        keep_mask = np.asarray(keep_mask, dtype=bool)
        for c, floor in min_true_class_coverage.items():
            c = int(c)
            floor = float(floor)
            m = y_true == c
            n = int(m.sum())
            if n <= 0:
                continue
            cov = float(np.mean(keep_mask[m]))
            if cov + 1e-12 < floor:
                violations[str(c)] = {"coverage": cov, "floor": floor, "count": n}
        return violations

    for name, u in score_map.items():
        cov_curve, risk_curve = risk_coverage_curve(u, pred_val == y_val)
        aurc_val = float(np.trapz(risk_curve, cov_curve))

        thresholds_by_class = None
        if strategy == "global":
            if objective == "aurc":
                q = int(np.clip(round(target_coverage * 100.0), 1, 100))
                ut = float(np.percentile(u, q))
                keep = _global_keep(u, ut)
                fbeta = float(fbeta_score(y_val[keep], pred_val[keep], beta=beta, average="macro", labels=labels, zero_division=0))
                crit = -aurc_val
            else:
                ut, fbeta, cov = pick_uncertainty_cut(u, y_val, pred_val, beta=beta, labels=labels, min_keep=max(25, int(0.05 * len(y_val))))
                keep = _global_keep(u, ut)
                crit = fbeta
        else:
            # Class-conditional thresholds by predicted class to avoid systematic
            # over-rejection of intrinsically high-uncertainty classes.
            keep, thresholds_by_class = _classwise_pred_keep(u, pred_val, target_coverage)
            cov = float(keep.mean())
            if keep.sum() < max(25, int(0.05 * len(y_val))):
                continue
            fbeta = float(fbeta_score(y_val[keep], pred_val[keep], beta=beta, average="macro", labels=labels, zero_division=0))
            if objective == "aurc":
                crit = -aurc_val
            else:
                # Prefer macro-quality while softly matching requested coverage.
                cov_penalty = abs(cov - float(target_coverage))
                crit = fbeta - 0.1 * cov_penalty
            ut = float("nan")

        cov = float(keep.mean())
        floor_violations = _coverage_violations_by_true_class(y_val, keep)
        if floor_violations:
            continue
        cand = {
            "name": name,
            "threshold": float(ut),
            "thresholds_by_pred_class": thresholds_by_class,
            "coverage": cov,
            "macro_fbeta_kept": float(fbeta),
            "aurc": aurc_val,
            "auroc_error": _safe_auroc_binary(error, u),
            "curve": {"coverage": cov_curve.tolist(), "risk": risk_curve.tolist()},
            "criterion": float(crit),
            "strategy": strategy,
        }
        if best is None or cand["criterion"] > best["criterion"]:
            best = cand

    return best


def _build_keep_mask(
    u: np.ndarray,
    pred: np.ndarray,
    chosen: Dict,
) -> np.ndarray:
    strategy = chosen.get("strategy", "global")
    if strategy == "global":
        return u <= float(chosen["threshold"])
    if strategy == "classwise_pred":
        tb = chosen.get("thresholds_by_pred_class") or {}
        keep = np.zeros_like(u, dtype=bool)
        for i in range(len(u)):
            t = float(tb.get(str(int(pred[i])), np.inf))
            keep[i] = bool(u[i] <= t)
        return keep
    raise ValueError(f"Unknown strategy in chosen gate: {strategy}")


def _class_coverage(y: np.ndarray, keep: np.ndarray, class_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c, n in enumerate(class_names):
        m = y == c
        total = int(m.sum())
        kept = int((m & keep).sum())
        out[n] = {
            "count": total,
            "kept": kept,
            "coverage": float(kept / max(1, total)),
        }
    return out


def _score_map_from_alpha(alpha: np.ndarray) -> Dict[str, np.ndarray]:
    a = torch.from_numpy(np.asarray(alpha, dtype=np.float32))
    p = dirichlet_mean(a)
    return {
        "vacuity": vacuity(a).cpu().numpy(),
        "entropy": predictive_entropy(p).cpu().numpy(),
        "expected_entropy": expected_categorical_entropy(a).cpu().numpy(),
        "mi": mutual_information(a).cpu().numpy(),
        "trace_uncertainty": total_uncertainty_trace(a).cpu().numpy(),
    }


def _aggregate_leaf_columns(x_leaf: np.ndarray, leaf_to_group: np.ndarray, n_groups: int) -> np.ndarray:
    x_leaf = np.asarray(x_leaf, dtype=float)
    out = np.zeros((x_leaf.shape[0], int(n_groups)), dtype=float)
    for g in range(int(n_groups)):
        cols = np.where(leaf_to_group == g)[0]
        if cols.size == 0:
            continue
        out[:, g] = x_leaf[:, cols].sum(axis=1)
    den = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out / den


def _aggregate_leaf_alpha(alpha_leaf: np.ndarray, leaf_to_group: np.ndarray, n_groups: int) -> np.ndarray:
    alpha_leaf = np.asarray(alpha_leaf, dtype=float)
    out = np.zeros((alpha_leaf.shape[0], int(n_groups)), dtype=float)
    for g in range(int(n_groups)):
        cols = np.where(leaf_to_group == g)[0]
        if cols.size == 0:
            continue
        out[:, g] = alpha_leaf[:, cols].sum(axis=1)
    return np.clip(out, 1.0 + 1e-6, None)


def _default_uncertainty_choice(
    *,
    y_val: np.ndarray,
    pred_val: np.ndarray,
    score_map_val: Dict[str, np.ndarray],
    strategy: str,
    target_coverage: float,
) -> Dict:
    # Robust fallback: expected-entropy quantile gate.
    name = "expected_entropy" if "expected_entropy" in score_map_val else list(score_map_val.keys())[0]
    u = np.asarray(score_map_val[name], dtype=float)
    ut = float(np.quantile(u, float(np.clip(target_coverage, 0.0, 1.0))))
    keep = u <= ut
    return {
        "name": name,
        "threshold": ut if strategy == "global" else float("nan"),
        "thresholds_by_pred_class": None,
        "coverage": float(keep.mean()),
        "macro_fbeta_kept": float(fbeta_score(y_val[keep], pred_val[keep], beta=1.0, average="macro", labels=np.arange(int(np.max(y_val) + 1)), zero_division=0)),
        "aurc": float(aurc(u, pred_val == y_val)),
        "auroc_error": _safe_auroc_binary((pred_val != y_val).astype(int), u),
        "curve": {},
        "criterion": float("nan"),
        "strategy": "global",
    }


def _generate_hierarchy_level_reports(
    *,
    out_dir: Path,
    taxonomy: Taxonomy,
    val_y_leaf: np.ndarray,
    test_y_leaf: np.ndarray,
    val_probs_leaf: np.ndarray,
    test_probs_leaf: np.ndarray,
    val_alpha_leaf: np.ndarray,
    test_alpha_leaf: np.ndarray,
    objective: str,
    uncertainty_gate_strategy: str,
    target_coverage: float,
    uncertainty_beta: float,
) -> Dict:
    hier_dir = out_dir / "hierarchy_levels"
    hier_dir.mkdir(parents=True, exist_ok=True)

    leaf_names = list(taxonomy.broad_classes)
    specs = taxonomy.hierarchy_level_specs()
    levels_payload: List[Dict] = []

    for spec in specs:
        level_idx = int(spec["level_index"])
        level_name = str(spec["level_name"])
        node_names = list(spec["node_names"])
        n_nodes = len(node_names)
        level_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in level_name).strip("_") or f"level_{level_idx+1}"
        level_dir = hier_dir / f"level_{level_idx+1:02d}_{level_slug}"
        level_dir.mkdir(parents=True, exist_ok=True)

        leaf_to_node = np.asarray(
            [int(spec["broad_to_node"][leaf_names[i]]) for i in range(len(leaf_names))],
            dtype=int,
        )
        y_val = leaf_to_node[np.asarray(val_y_leaf, dtype=int)]
        y_test = leaf_to_node[np.asarray(test_y_leaf, dtype=int)]
        p_val = _aggregate_leaf_columns(val_probs_leaf, leaf_to_node, n_nodes)
        p_test = _aggregate_leaf_columns(test_probs_leaf, leaf_to_node, n_nodes)
        a_val = _aggregate_leaf_alpha(val_alpha_leaf, leaf_to_node, n_nodes)
        a_test = _aggregate_leaf_alpha(test_alpha_leaf, leaf_to_node, n_nodes)

        pred_val = p_val.argmax(axis=1)
        pred_test = p_test.argmax(axis=1)

        base_val_metrics = write_report_tables(y_val, p_val, node_names, level_dir, "val_base", pred_override=pred_val)
        base_test_metrics = write_report_tables(y_test, p_test, node_names, level_dir, "test_base", pred_override=pred_test)

        score_map_val = _score_map_from_alpha(a_val)
        score_map_test = _score_map_from_alpha(a_test)
        chosen = _choose_uncertainty_gate(
            y_val,
            pred_val,
            score_map_val,
            objective=objective,
            target_coverage=target_coverage,
            beta=uncertainty_beta,
            labels=np.arange(n_nodes),
            strategy=uncertainty_gate_strategy,
            min_true_class_coverage=None,
        )
        if chosen is None and uncertainty_gate_strategy != "global":
            chosen = _choose_uncertainty_gate(
                y_val,
                pred_val,
                score_map_val,
                objective=objective,
                target_coverage=target_coverage,
                beta=uncertainty_beta,
                labels=np.arange(n_nodes),
                strategy="global",
                min_true_class_coverage=None,
            )
        if chosen is None:
            chosen = _default_uncertainty_choice(
                y_val=y_val,
                pred_val=pred_val,
                score_map_val=score_map_val,
                strategy=uncertainty_gate_strategy,
                target_coverage=target_coverage,
            )

        keep_val = _build_keep_mask(score_map_val[chosen["name"]], pred_val, chosen)
        keep_test = _build_keep_mask(score_map_test[chosen["name"]], pred_test, chosen)
        if not np.any(keep_val):
            keep_val = np.ones_like(keep_val, dtype=bool)
        if not np.any(keep_test):
            keep_test = np.ones_like(keep_test, dtype=bool)

        kept_val_metrics = write_report_tables(
            y_val[keep_val],
            p_val[keep_val],
            node_names,
            level_dir,
            "val_post_kept",
            pred_override=pred_val[keep_val],
        )
        kept_test_metrics = write_report_tables(
            y_test[keep_test],
            p_test[keep_test],
            node_names,
            level_dir,
            "test_post_kept",
            pred_override=pred_test[keep_test],
        )

        fig_val = plot_abstention_overview(
            y_val,
            pred_val,
            keep_val,
            node_names,
            title=f"Hierarchy {level_name}: VAL post-kept abstention",
        )
        savefig_pdf(fig_val, level_dir / "val_post_kept_abstention_overview.pdf")
        plt.close(fig_val)

        fig_test = plot_abstention_overview(
            y_test,
            pred_test,
            keep_test,
            node_names,
            title=f"Hierarchy {level_name}: TEST post-kept abstention",
        )
        savefig_pdf(fig_test, level_dir / "test_post_kept_abstention_overview.pdf")
        plt.close(fig_test)

        rep_base_test = classification_report(
            y_test,
            pred_test,
            labels=list(range(n_nodes)),
            target_names=node_names,
            output_dict=True,
            zero_division=0,
        )
        rep_kept_test = classification_report(
            y_test[keep_test],
            pred_test[keep_test],
            labels=list(range(n_nodes)),
            target_names=node_names,
            output_dict=True,
            zero_division=0,
        )
        per_class_base_test = {
            k: {
                "precision": float(rep_base_test[k]["precision"]),
                "recall": float(rep_base_test[k]["recall"]),
                "f1": float(rep_base_test[k]["f1-score"]),
                "support": int(rep_base_test[k]["support"]),
            }
            for k in node_names
        }
        per_class_kept_test = {
            k: {
                "precision": float(rep_kept_test[k]["precision"]),
                "recall": float(rep_kept_test[k]["recall"]),
                "f1": float(rep_kept_test[k]["f1-score"]),
                "support": int(rep_kept_test[k]["support"]),
            }
            for k in node_names
        }

        selected_score = str(chosen["name"])
        selected_u_val = np.asarray(score_map_val[selected_score], dtype=float)
        cov_curve, risk_curve = risk_coverage_curve(selected_u_val, pred_val == y_val)
        level_payload = {
            "level_index": level_idx,
            "level_name": level_name,
            "node_names": node_names,
            "leaf_to_node": {leaf: node_names[int(spec["broad_to_node"][leaf])] for leaf in leaf_names},
            "out_dir": str(level_dir),
            "base_val": base_val_metrics,
            "base_test": base_test_metrics,
            "post_kept_val": kept_val_metrics,
            "post_kept_test": kept_test_metrics,
            "post_kept_coverage_val": float(np.mean(keep_val)),
            "post_kept_coverage_test": float(np.mean(keep_test)),
            "coverage_by_true_class": {
                "val": _class_coverage(y_val, keep_val, node_names),
                "test": _class_coverage(y_test, keep_test, node_names),
            },
            "selected_uncertainty": {
                "score": selected_score,
                "strategy": chosen.get("strategy", uncertainty_gate_strategy),
                "coverage_val": float(np.mean(keep_val)),
                "aurc_val": float(np.trapz(risk_curve, cov_curve)),
                "auroc_error_val": _safe_auroc_binary((pred_val != y_val).astype(int), selected_u_val),
            },
            "base_test_per_class": per_class_base_test,
            "post_kept_test_per_class": per_class_kept_test,
        }
        (level_dir / "summary.json").write_text(json.dumps(level_payload, indent=2))
        levels_payload.append(level_payload)

    md_lines = [
        "# Hierarchy Level Diagnostics",
        "",
        "| level | nodes | base test macro-F1 | post-kept test macro-F1 | post-kept coverage | selected uncertainty score |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for lv in levels_payload:
        md_lines.append(
            f"| {lv['level_name']} | {len(lv['node_names'])} | "
            f"{float(lv['base_test']['macro_f1']):.4f} | "
            f"{float(lv['post_kept_test']['macro_f1']):.4f} | "
            f"{float(lv['post_kept_coverage_test']):.4f} | "
            f"{lv['selected_uncertainty']['score']} |"
        )
    (hier_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    payload = {
        "broad_classes": leaf_names,
        "levels": levels_payload,
    }
    (out_dir / "hierarchy_levels_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def _write_metrics_tables_md(out_dir: Path, table: Dict[str, Dict]) -> None:
    rows = ["# Metrics Comparison", "", "| variant | split | acc | bal_acc | macro_f1 | auprc_macro | ece | nll |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant, splits in table.items():
        for split, m in splits.items():
            rows.append(
                f"| {variant} | {split} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | {m['auprc_macro']:.4f} | {m['ece']:.4f} | {m['nll']:.4f} |"
            )
    (out_dir / "metrics_tables.md").write_text("\n".join(rows))


def _write_per_class_md(out_dir: Path, y_true: np.ndarray, probs: np.ndarray, class_names: Sequence[str], title: str) -> None:
    pred = probs.argmax(axis=1)
    rep = classification_report(
        y_true,
        pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    rows = [f"# {title}", "", "| class | precision | recall | f1 | support |", "|---|---:|---:|---:|---:|"]
    for n in class_names:
        d = rep[n]
        rows.append(f"| {n} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1-score']:.4f} | {int(d['support'])} |")
    (out_dir / "per_class_metrics.md").write_text("\n".join(rows))


def generate_publishable_report(
    cfg: Config,
    *,
    taxonomy: Taxonomy = DEFAULT_TAXONOMY,
    ckpt_path: Path,
    out_dir: Path,
    device: Optional[torch.device] = None,
    calibrate_temperature: bool = True,
    do_prior_adjust: bool = True,
    threshold_beta_by_class: Optional[List[float]] = None,
    enable_vector_scaling: bool = True,
    uncertainty_objective: str = "aurc",
    uncertainty_gate_strategy: str = "global",
    target_coverage: float = 0.85,
    uncertainty_beta: float = 1.0,
    min_true_class_coverage: Optional[Dict[int, float]] = None,
    bootstrap_samples: int = 500,
) -> Dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    from .models import EventTransformerEncoder, EvidentialClassifier
    from .train import make_loaders

    train_ld, val_ld, test_ld, _, loader_meta = make_loaders(cfg, taxonomy)

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
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val = collect_evidential(model, val_ld, device=device)
    test = collect_evidential(model, test_ld, device=device)

    class_names = taxonomy.broad_classes
    C = taxonomy.num_classes

    metrics_table: Dict[str, Dict] = {}

    base_val_metrics = write_report_tables(val["y"], val["probs"], class_names, out_dir, "val_base")
    base_test_metrics = write_report_tables(test["y"], test["probs"], class_names, out_dir, "test_base")
    metrics_table["baseline"] = {"val": base_val_metrics, "test": base_test_metrics}

    p_val_cur = val["probs"]
    p_test_cur = test["probs"]

    if calibrate_temperature:
        ts = EvidenceTemperatureScaler().fit(val["evidence"], val["y"])
        p_val_T = ts.transform(val["evidence"])
        p_test_T = ts.transform(test["evidence"])
        (out_dir / "temperature_scaler.json").write_text(json.dumps({"temperature": ts.temperature}, indent=2))
        mvt = write_report_tables(val["y"], p_val_T, class_names, out_dir, "val_temp")
        mtt = write_report_tables(test["y"], p_test_T, class_names, out_dir, "test_temp")
        metrics_table["temp_scaling"] = {"val": mvt, "test": mtt}
        p_val_cur, p_test_cur = p_val_T, p_test_T

    if enable_vector_scaling:
        vs = EvidenceVectorScaler().fit(val["evidence"], val["y"])
        p_val_V = vs.transform(val["evidence"])
        p_test_V = vs.transform(test["evidence"])
        (out_dir / "vector_scaler.json").write_text(json.dumps({"temperature": np.asarray(vs.temperature).tolist()}, indent=2))
        mvv = write_report_tables(val["y"], p_val_V, class_names, out_dir, "val_vector")
        mtv = write_report_tables(test["y"], p_test_V, class_names, out_dir, "test_vector")
        metrics_table["vector_scaling"] = {"val": mvv, "test": mtv}
        # pick better calibrator on val NLL, tie by ECE
        if (mvv["nll"] < compute_basic_metrics(val["y"], p_val_cur, C)["nll"]) or (
            abs(mvv["nll"] - compute_basic_metrics(val["y"], p_val_cur, C)["nll"]) < 1e-9 and mvv["ece"] < compute_basic_metrics(val["y"], p_val_cur, C)["ece"]
        ):
            p_val_cur, p_test_cur = p_val_V, p_test_V

    if do_prior_adjust:
        pi_train = _broad_train_prior_from_manifest(cfg, taxonomy)
        pi_val = np.bincount(val["y"], minlength=C) / max(1, len(val["y"]))
        # Use the same deploy prior estimated from VAL when transferring to TEST.
        # This avoids leaking test label distribution into test-time post-processing.
        p_val_adj = adjust_for_priors(p_val_cur, pi_train, pi_val)
        p_test_adj = adjust_for_priors(p_test_cur, pi_train, pi_val)
        (out_dir / "priors.json").write_text(
            json.dumps({"pi_train": pi_train.tolist(), "pi_deploy_val": pi_val.tolist()}, indent=2)
        )
    else:
        p_val_adj = p_val_cur
        p_test_adj = p_test_cur

    thr = fit_prob_thresholds(val["y"], p_val_adj, beta_by_class=threshold_beta_by_class)
    pred_val_thr = predict_with_thresholds(p_val_adj, thr)
    pred_test_thr = predict_with_thresholds(p_test_adj, thr)

    score_map_val = _uncertainty_score_dict(val)
    score_map_test = _uncertainty_score_dict(test)
    fusion = _fit_uncertainty_fusion(score_map_val, val["y"], pred_val_thr)
    score_map_val["fused_uncertainty"] = _apply_uncertainty_fusion(score_map_val, fusion)
    score_map_test["fused_uncertainty"] = _apply_uncertainty_fusion(score_map_test, fusion)
    (out_dir / "uncertainty_fusion.json").write_text(json.dumps(fusion, indent=2))

    requested_min_true_class_coverage = dict(min_true_class_coverage or {})
    applied_min_true_class_coverage = dict(requested_min_true_class_coverage)
    gate_selection_warnings: List[str] = []

    chosen = _choose_uncertainty_gate(
        val["y"],
        pred_val_thr,
        score_map_val,
        objective=uncertainty_objective,
        target_coverage=target_coverage,
        beta=uncertainty_beta,
        labels=np.arange(C),
        strategy=uncertainty_gate_strategy,
        min_true_class_coverage=applied_min_true_class_coverage,
    )
    if chosen is None and applied_min_true_class_coverage:
        # Floors can be infeasible in tiny-support classes (e.g., TDE) for some
        # model variants. Fallback keeps reports reproducible and avoids job loss.
        chosen = _choose_uncertainty_gate(
            val["y"],
            pred_val_thr,
            score_map_val,
            objective=uncertainty_objective,
            target_coverage=target_coverage,
            beta=uncertainty_beta,
            labels=np.arange(C),
            strategy=uncertainty_gate_strategy,
            min_true_class_coverage=None,
        )
        if chosen is not None:
            gate_selection_warnings.append(
                "No feasible uncertainty gate satisfied requested min_true_class_coverage on VAL; "
                "fell back to unconstrained selection."
            )
            applied_min_true_class_coverage = {}

    if chosen is None and uncertainty_gate_strategy != "global":
        chosen = _choose_uncertainty_gate(
            val["y"],
            pred_val_thr,
            score_map_val,
            objective=uncertainty_objective,
            target_coverage=target_coverage,
            beta=uncertainty_beta,
            labels=np.arange(C),
            strategy="global",
            min_true_class_coverage=None,
        )
        if chosen is not None:
            gate_selection_warnings.append(
                "No feasible classwise_pred uncertainty gate found on VAL; "
                "fell back to unconstrained global gate selection."
            )
            applied_min_true_class_coverage = {}

    if chosen is None:
        raise RuntimeError(
            "Failed to choose an uncertainty gate on validation data "
            f"(objective={uncertainty_objective}, strategy={uncertainty_gate_strategy}, "
            f"target_coverage={target_coverage}, min_true_class_coverage={requested_min_true_class_coverage})."
        )

    u_val = score_map_val[chosen["name"]]
    u_test = score_map_test[chosen["name"]]
    keep_val = _build_keep_mask(u_val, pred_val_thr, chosen)
    keep_test = _build_keep_mask(u_test, pred_test_thr, chosen)

    kept_val_metrics = write_report_tables(
        val["y"][keep_val],
        p_val_adj[keep_val],
        class_names,
        out_dir,
        "val_post_kept",
        pred_override=pred_val_thr[keep_val],
    )
    kept_test_metrics = write_report_tables(
        test["y"][keep_test],
        p_test_adj[keep_test],
        class_names,
        out_dir,
        "test_post_kept",
        pred_override=pred_test_thr[keep_test],
    )
    metrics_table["post_kept"] = {"val": kept_val_metrics, "test": kept_test_metrics}

    fig_abs_val = plot_abstention_overview(
        val["y"],
        pred_val_thr,
        keep_val,
        class_names,
        title="VAL post-kept abstention overview",
    )
    savefig_pdf(fig_abs_val, out_dir / "val_post_kept_abstention_overview.pdf")
    plt.close(fig_abs_val)

    fig_abs_test = plot_abstention_overview(
        test["y"],
        pred_test_thr,
        keep_test,
        class_names,
        title="TEST post-kept abstention overview",
    )
    savefig_pdf(fig_abs_test, out_dir / "test_post_kept_abstention_overview.pdf")
    plt.close(fig_abs_test)

    hierarchy_summary = _generate_hierarchy_level_reports(
        out_dir=out_dir,
        taxonomy=taxonomy,
        val_y_leaf=val["y"],
        test_y_leaf=test["y"],
        val_probs_leaf=p_val_adj,
        test_probs_leaf=p_test_adj,
        val_alpha_leaf=val["alpha"],
        test_alpha_leaf=test["alpha"],
        objective=uncertainty_objective,
        uncertainty_gate_strategy=uncertainty_gate_strategy,
        target_coverage=target_coverage,
        uncertainty_beta=uncertainty_beta,
    )

    uncertainty_metrics = {
        "objective": uncertainty_objective,
        "target_coverage": target_coverage,
        "beta": uncertainty_beta,
        "min_true_class_coverage": {str(int(k)): float(v) for k, v in applied_min_true_class_coverage.items()},
        "min_true_class_coverage_requested": {str(int(k)): float(v) for k, v in requested_min_true_class_coverage.items()},
        "min_true_class_coverage_applied": {str(int(k)): float(v) for k, v in applied_min_true_class_coverage.items()},
        "selection_warnings": gate_selection_warnings,
        "selected": {
            "score": chosen["name"],
            "strategy": chosen.get("strategy", uncertainty_gate_strategy),
            "threshold": chosen["threshold"],
            "thresholds_by_pred_class": chosen.get("thresholds_by_pred_class"),
            "coverage_val": chosen["coverage"],
            "macro_fbeta_val": chosen["macro_fbeta_kept"],
            "aurc_val": chosen["aurc"],
            "reason": f"Chosen by best objective value on VAL among {{{', '.join(score_map_val.keys())}}}",
        },
        "coverage_by_true_class": {
            "val": _class_coverage(val["y"], keep_val, class_names),
            "test": _class_coverage(test["y"], keep_test, class_names),
        },
        "scores": {},
    }

    for n, u in score_map_val.items():
        cov, risk = risk_coverage_curve(u, pred_val_thr == val["y"])
        uncertainty_metrics["scores"][n] = {
            "aurc_val": float(np.trapz(risk, cov)),
            "auroc_error_val": _safe_auroc_binary((pred_val_thr != val["y"]).astype(int), u),
            "curve_val": {"coverage": cov.tolist(), "risk": risk.tolist()},
        }

    (out_dir / "uncertainty_metrics.json").write_text(json.dumps(uncertainty_metrics, indent=2))
    val_q_thr = {k: float(np.quantile(np.asarray(v, dtype=float), float(np.clip(target_coverage, 0.0, 1.0)))) for k, v in score_map_val.items()}
    chosen_thr_plot = float(chosen["threshold"]) if chosen.get("strategy", "global") == "global" else float("nan")
    _plot_uncertainty_separation(
        out_dir / "uncertainty_separation_distributions.pdf",
        val_y=val["y"],
        val_pred=pred_val_thr,
        val_scores=score_map_val,
        test_y=test["y"],
        test_pred=pred_test_thr,
        test_scores=score_map_test,
        val_quantile_thresholds=val_q_thr,
        chosen_score=chosen["name"],
        chosen_threshold=chosen_thr_plot,
    )

    ood_ref = _fit_ood_reference(
        score_map_val,
        inlier_mask=(pred_val_thr == val["y"]) & keep_val,
        feature_keys=list(score_map_val.keys()),
        score_quantile=0.995,
        vote_quantile=0.995,
        min_votes=2,
    )
    ood_score_val, ood_votes_val, ood_flag_val = _apply_ood_reference(score_map_val, ood_ref)
    ood_score_test, ood_votes_test, ood_flag_test = _apply_ood_reference(score_map_test, ood_ref)
    ood_summary = {
        "feature_keys": ood_ref["feature_keys"],
        "score_threshold": float(ood_ref["score_threshold"]),
        "feature_thresholds": ood_ref["feature_thresholds"],
        "min_votes": int(ood_ref["min_votes"]),
        "val_flag_rate": float(np.mean(ood_flag_val)),
        "test_flag_rate": float(np.mean(ood_flag_test)),
        "val_auroc_error": _safe_auroc_binary((pred_val_thr != val["y"]).astype(int), ood_score_val),
        "test_auroc_error": _safe_auroc_binary((pred_test_thr != test["y"]).astype(int), ood_score_test),
        "val_flag_rate_on_errors": float(np.mean(ood_flag_val[pred_val_thr != val["y"]])) if np.any(pred_val_thr != val["y"]) else float("nan"),
        "test_flag_rate_on_errors": float(np.mean(ood_flag_test[pred_test_thr != test["y"]])) if np.any(pred_test_thr != test["y"]) else float("nan"),
    }
    (out_dir / "ood_summary.json").write_text(json.dumps(ood_summary, indent=2))

    setup_mpl_paper(usetex=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    m_corr = pred_test_thr == test["y"]
    ax.hist(ood_score_test[m_corr], bins=40, alpha=0.60, density=True, color=NORD["nord14"], label="correct")
    ax.hist(ood_score_test[~m_corr], bins=40, alpha=0.60, density=True, color=NORD["nord11"], label="misclassified")
    ax.axvline(float(ood_ref["score_threshold"]), color=NORD["nord0"], ls="--", lw=1.2, label="OOD threshold")
    ax.set_xlabel("OOD score")
    ax.set_ylabel("density")
    ax.set_title("OOD Score Distribution (Test)")
    style_axes_inward(ax, grid_y=True)
    ax.legend()
    fig.tight_layout()
    savefig_pdf(fig, out_dir / "ood_score_distribution_test.pdf")
    plt.close(fig)

    val_df = loader_meta.get("val_df", None)
    test_df = loader_meta.get("test_df", None)
    if val_df is None or test_df is None:
        manifest_dir = Path(cfg.manifest_dir) if cfg.manifest_dir is not None else Path(cfg.data_dir)
        val_df_raw = read_manifest_csv(manifest_dir / "manifest_val.csv", data_dir=cfg.data_dir, path_prefix=cfg.path_prefix)
        test_df_raw = read_manifest_csv(manifest_dir / "manifest_test.csv", data_dir=cfg.data_dir, path_prefix=cfg.path_prefix)
        qkwargs = {
            "horizon_days": cfg.horizon_days,
            "drop_i_band": cfg.drop_i_band,
            "min_obs_total": cfg.min_obs_total,
            "min_obs_g": cfg.min_obs_g,
            "min_obs_r": cfg.min_obs_r,
            "min_obs_i": cfg.min_obs_i,
            "min_bands_observed": cfg.min_bands_observed,
        }
        val_df, _ = filter_manifest_quality(val_df_raw, **qkwargs)
        test_df, _ = filter_manifest_quality(test_df_raw, **qkwargs)

    rows_test = _build_split_rows(
        test_df,
        y_true=test["y"],
        probs=p_test_adj,
        pred_thr=pred_test_thr,
        keep_mask=keep_test,
        score_map=score_map_test,
        class_names=class_names,
        ood_score=ood_score_test,
        ood_votes=ood_votes_test,
        ood_flag=ood_flag_test,
    )

    catalog_fields = [
        "row_index",
        "obj_id",
        "filepath",
        "true_id",
        "pred_id",
        "true_name",
        "pred_name",
        "correct",
        "abstained",
        "confidence",
        "top1_prob",
        "top2_prob",
        "margin_top1_top2",
        "true_prob",
        "top_classes",
        "top_probs",
        "vacuity",
        "entropy",
        "expected_entropy",
        "mi",
        "trace_uncertainty",
        "fused_uncertainty",
        "ood_score",
        "ood_votes",
        "ood_flag",
    ]
    _write_rows_csv([r for r in rows_test if not r["correct"]], out_dir / "misclassified_catalog_test.csv", fieldnames=catalog_fields)
    _write_rows_csv([r for r in rows_test if r.get("ood_flag", False)], out_dir / "ood_flags_test.csv", fieldnames=catalog_fields)

    gallery_rows, gallery_strata = _select_stratified_misclassified(rows_test, uncertainty_key=chosen["name"], max_per_stratum=6)
    _plot_misclassification_gallery(
        out_dir / "misclassification_gallery_stratified.pdf",
        rows=gallery_rows,
        class_names=class_names,
        uncertainty_key=chosen["name"],
        uncertainty_thresholds=val_q_thr,
        horizon_days=float(cfg.horizon_days),
        drop_i_band=cfg.drop_i_band,
    )
    _write_rows_csv(gallery_rows, out_dir / "misclassification_gallery_index.csv", fieldnames=catalog_fields)
    gallery_summary = _write_misclassification_summary_md(
        out_dir / "misclassification_summary.md",
        rows_test=rows_test,
        selected_summary=gallery_strata,
        uncertainty_key=chosen["name"],
    )
    (out_dir / "misclassification_summary.json").write_text(json.dumps(gallery_summary, indent=2))

    for split_name, d, p in [("val", val, p_val_adj), ("test", test, p_test_adj)]:
        correct = p.argmax(axis=1) == d["y"]
        for k, xlabel in [
            ("confidence", "Max class probability"),
            ("vacuity", r"Vacuity $u=C/\sum\alpha$"),
            ("trace_uncertainty", r"$\sqrt{\mathrm{tr}(\mathrm{Cov}[p])}$"),
        ]:
            fig = _plot_split_hist(d[k], correct, xlabel=xlabel, title=f"{split_name}: {k} (correct vs incorrect)")
            savefig_pdf(fig, out_dir / f"{split_name}_{k}_correctness.pdf")
            plt.close(fig)

    boot = _bootstrap_metrics(test["y"], p_test_adj, class_names, n_boot=bootstrap_samples, seed=cfg.seed + 17)
    (out_dir / "bootstrap_metrics.json").write_text(json.dumps(boot, indent=2))
    _write_bootstrap_md(boot, out_dir / "bootstrap_metrics.md")

    _write_metrics_tables_md(out_dir, metrics_table)
    _write_per_class_md(out_dir, test["y"], p_test_adj, class_names, title="Per-Class Metrics (Test, Final)")

    postprocess = {
        "thresholds": thr.tolist(),
        "uncertainty_score": chosen["name"],
        "uncertainty_gate_strategy": chosen.get("strategy", uncertainty_gate_strategy),
        "uncertainty_cut": float(chosen["threshold"]),
        "uncertainty_cut_by_pred_class": chosen.get("thresholds_by_pred_class"),
        "val_macro_fbeta_at_cut": float(chosen["macro_fbeta_kept"]),
        "val_coverage_at_cut": float(chosen["coverage"]),
    }
    (out_dir / "postprocess.json").write_text(json.dumps(postprocess, indent=2))

    summary = {
        "base_test": base_test_metrics,
        "post_kept_test": kept_test_metrics,
        "post_kept_coverage_test": float(keep_test.mean()),
        "selected_uncertainty": chosen,
        "ood": {
            "score_threshold": float(ood_ref["score_threshold"]),
            "test_flag_rate": float(np.mean(ood_flag_test)),
            "test_auroc_error": _safe_auroc_binary((pred_test_thr != test["y"]).astype(int), ood_score_test),
        },
        "misclassification_gallery": {
            "num_pages": int(len(gallery_rows)),
            "strata": gallery_strata,
            "summary_file": "misclassification_summary.md",
        },
        "hierarchy_levels": {
            "summary_file": "hierarchy_levels_summary.json",
            "num_levels": int(len(hierarchy_summary.get("levels", []))),
            "level_names": [str(x.get("level_name")) for x in hierarchy_summary.get("levels", [])],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
