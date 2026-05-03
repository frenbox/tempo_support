from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .config import Config
from .data import global_feature_dim
from .evaluate import write_report_tables
from .report import collect_evidential
from .taxonomy import DEFAULT_TAXONOMY, Taxonomy
from .train import make_loaders


def _build_model(cfg: Config, *, taxonomy: Taxonomy, device: torch.device):
    from .models import EventTransformerEncoder, EvidentialClassifier

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
    return model


def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return p


def _combine_outputs(
    outputs: Sequence[Dict[str, np.ndarray]],
    *,
    split: str,
    rule: str,
) -> Dict[str, np.ndarray]:
    if rule not in {"alpha_mean", "prob_mean"}:
        raise ValueError(f"Unknown ensemble rule: {rule}")

    y_ref = outputs[0][f"{split}_y"]
    for o in outputs[1:]:
        if not np.array_equal(y_ref, o[f"{split}_y"]):
            raise RuntimeError("Seed outputs are misaligned: label order mismatch across runs")

    if rule == "alpha_mean":
        a = np.mean(np.stack([o[f"{split}_alpha"] for o in outputs], axis=0), axis=0)
        p = _normalize_probs(a / np.clip(a.sum(axis=1, keepdims=True), 1e-12, None))
        return {"y": y_ref, "alpha": a, "probs": p}

    p = np.mean(np.stack([o[f"{split}_probs"] for o in outputs], axis=0), axis=0)
    p = _normalize_probs(p)
    return {"y": y_ref, "alpha": None, "probs": p}


@torch.no_grad()
def generate_ensemble_report(
    cfg: Config,
    *,
    ckpt_paths: Sequence[Path],
    out_dir: Path,
    taxonomy: Taxonomy = DEFAULT_TAXONOMY,
    device: Optional[torch.device] = None,
    selection_rule: str = "auto",
) -> Dict:
    """Build an ensemble report from multiple trained checkpoints.

    Supported combination rules:
    - alpha_mean: average Dirichlet alpha vectors across seeds, then normalize.
    - prob_mean: average class probabilities across seeds.
    """
    if len(ckpt_paths) < 2:
        raise ValueError("Ensemble requires at least 2 checkpoints")
    for p in ckpt_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing checkpoint: {p}")
    if selection_rule not in {"auto", "alpha_mean", "prob_mean"}:
        raise ValueError(f"Invalid selection_rule={selection_rule}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, val_ld, test_ld, _, _ = make_loaders(cfg, taxonomy)
    class_names = taxonomy.broad_classes

    per_seed = []
    for ckpt in ckpt_paths:
        model = _build_model(cfg, taxonomy=taxonomy, device=device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        val = collect_evidential(model, val_ld, device=device)
        test = collect_evidential(model, test_ld, device=device)
        per_seed.append(
            {
                "ckpt": str(ckpt),
                "val_y": val["y"],
                "test_y": test["y"],
                "val_alpha": val["alpha"],
                "test_alpha": test["alpha"],
                "val_probs": val["probs"],
                "test_probs": test["probs"],
            }
        )

    rules = ["alpha_mean", "prob_mean"] if selection_rule == "auto" else [selection_rule]
    results = {}
    for rule in rules:
        val_c = _combine_outputs(per_seed, split="val", rule=rule)
        test_c = _combine_outputs(per_seed, split="test", rule=rule)
        m_val = write_report_tables(val_c["y"], val_c["probs"], class_names, out_dir, f"val_ensemble_{rule}")
        m_test = write_report_tables(test_c["y"], test_c["probs"], class_names, out_dir, f"test_ensemble_{rule}")
        results[rule] = {"val": m_val, "test": m_test}

    if selection_rule == "auto":
        ranked = sorted(
            results.items(),
            key=lambda kv: (
                float(kv[1]["val"]["macro_f1"]),
                float(kv[1]["val"]["balanced_accuracy"]),
                -float(kv[1]["val"]["nll"]),
            ),
            reverse=True,
        )
        chosen_rule = ranked[0][0]
    else:
        chosen_rule = selection_rule

    summary = {
        "selection_rule": selection_rule,
        "chosen_rule": chosen_rule,
        "rule_note": {
            "alpha_mean": "Ensemble alpha(x)=mean_m alpha_m(x); probabilities from normalized alpha.",
            "prob_mean": "Ensemble probabilities p(x)=mean_m p_m(x).",
        },
        "checkpoints": [str(p) for p in ckpt_paths],
        "metrics_by_rule": results,
        "selected": results[chosen_rule],
    }
    (out_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

