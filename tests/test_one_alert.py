"""Smoke test: run TEMPO on a single ZTF alert sampled from oracle_support/data."""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tempo_support.tempo_boom_ztf import (
    get_taxonomy,
    prv_candidates_to_photometry_df,
    run_tempo,
)
from tempo_support.plot_tempo import save_sunburst

ALERT_PATH = REPO_ROOT / "data" / "alert.json"
SUNBURST_PATH = REPO_ROOT / "data" / "tempo_sunburst.png"


def main():
    blob = json.loads(ALERT_PATH.read_text())
    ztf_id = blob.get("_id") or blob.get("objectId")
    prv = blob.get("prv_candidates") or []
    print(f"object: {ztf_id}, prv_candidates: {len(prv)}")

    df = prv_candidates_to_photometry_df(prv)
    print("converted photometry:")
    print(df.to_string(index=False))

    result = run_tempo(ztf_id, prv)
    if result is None:
        print("run_tempo returned None")
        return

    leaf = result["leaf"]
    decision = result["decision"]
    print(f"\nn_events: {result['n_events']}, horizon: {result['final_horizon_days']:.2f} days")
    print("leaf class probabilities (final ± calibrated std):")
    rows = list(zip(leaf["class_names"], leaf["probs_final"], leaf["prob_std_calibrated"]))
    rows.sort(key=lambda r: -float(r[1]))
    for name, p, s in rows:
        print(f"  {name:<8}{float(p) * 100:7.2f}%  ±{float(s) * 100:5.2f}%")
    print(
        f"\nleaf prediction: {leaf['pred_thresholded_name']} "
        f"({leaf['pred_thresholded_prob'] * 100:.2f}%), kept={leaf['kept']}"
    )
    fb = decision["fallback"]
    print(
        f"decision: label={fb.get('label')} level={fb.get('level_name')} "
        f"abstain={decision['abstain_completely']}"
    )

    print("\nuncertainty scores (leaf):")
    sm = leaf["score_map"]
    sel_name = leaf["uncertainty_selected_name"]
    sel_val = leaf["uncertainty_selected_value"]
    for k in ("vacuity", "entropy", "expected_entropy", "mi", "trace_uncertainty", "fused_uncertainty"):
        if k in sm:
            marker = "  <- gate" if k == sel_name else ""
            print(f"  {k:<22}{sm[k]:8.4f}{marker}")
    print(f"  selected gate: {sel_name} = {sel_val:.4f}")
    if leaf.get("ood"):
        print(f"OOD: score={leaf['ood']['score']:.3f} votes={leaf['ood']['votes']} flag={leaf['ood']['flag']}")

    print("\nhierarchy levels (top prob ± calibrated std):")
    for lv in result["hierarchy"]:
        pid = int(lv["pred_id"])
        std = float(lv["prob_std_calibrated"][pid])
        print(
            f"  {lv['level_name']:<8} pred={lv['pred_name']:<10} "
            f"prob={lv['top_prob'] * 100:6.2f}% ±{std * 100:5.2f}%  "
            f"kept={lv['kept']}  unc({lv['chosen_uncertainty_name']})={lv['chosen_uncertainty_value']:.4f}"
        )

    out = save_sunburst(
        result, get_taxonomy(), SUNBURST_PATH, title=f"TEMPO — {ztf_id}", font_size=16
    )
    print(f"\nsunburst saved to {out}")


if __name__ == "__main__":
    main()
