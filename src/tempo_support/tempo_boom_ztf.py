"""Run TEMPO inference on a ZTF alert (prv_candidates from BOOM Mongo).

The vendored ``photometry_edl`` package, helpers, and ``model_bundle/``
ship inside this package, so no external bundle directory is required.
Override the bundle location with env var ``TEMPO_BUNDLE_DIR`` if you
keep weights/runtime cache elsewhere.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_default_device("cpu")

logger = logging.getLogger(__name__)

ZTF_FID_TO_BAND = {1: "g", 2: "r", 3: "i"}
BAND_TO_FRITZ_FILTER = {"g": "ztfg", "r": "ztfr", "i": "ztfi"}
ZTF_FRITZ_ZP = 23.9
LOG10 = math.log(10.0)

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE_DIR = _PACKAGE_DIR / "model_bundle"


def _bundle_dir() -> Path:
    env = os.environ.get("TEMPO_BUNDLE_DIR")
    return Path(env) if env else DEFAULT_BUNDLE_DIR


_bundle_state = None


def _load_bundle():
    global _bundle_state
    if _bundle_state is not None:
        return _bundle_state

    bundle_root = _bundle_dir()
    if not bundle_root.exists():
        raise FileNotFoundError(
            f"TEMPO model_bundle not found at {bundle_root}. "
            "Set TEMPO_BUNDLE_DIR if it lives elsewhere."
        )

    from tempo_support._tempo_helpers import (
        build_runtime_postprocess_state,
        load_tempo_bundle,
    )

    run_dir = bundle_root / "run"
    report_dir = bundle_root / "report"
    cache_json = bundle_root / "runtime" / "p26_classwise_no_vector_runtime_cache.json"

    bundle = load_tempo_bundle(run_dir, report_dir=report_dir, device="cpu")
    state = build_runtime_postprocess_state(
        bundle, cache_json=cache_json, force_recompute=False
    )
    _bundle_state = (bundle, state)
    logger.info(
        "Loaded TEMPO bundle from %s (taxonomy=%s)",
        bundle_root,
        list(bundle.taxonomy.broad_classes),
    )
    return _bundle_state


def get_taxonomy():
    bundle, _ = _load_bundle()
    return bundle.taxonomy


def get_class_names():
    return list(get_taxonomy().broad_classes)


def _resolve_band(row):
    band = row.get("band")
    if isinstance(band, str) and band in BAND_TO_FRITZ_FILTER:
        return BAND_TO_FRITZ_FILTER[band]
    fid = row.get("fid")
    if fid is not None:
        try:
            band = ZTF_FID_TO_BAND.get(int(fid))
        except (TypeError, ValueError):
            band = None
        if band:
            return BAND_TO_FRITZ_FILTER[band]
    return None


def _is_finite(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


PUBLIC_PROGRAMIDS = {1, 2}


def prv_candidates_to_photometry_df(prv_candidates):
    """Convert ZTF ``prv_candidates`` rows to a Fritz-style flux dataframe.

    Drops rows with ``programid`` not in ``{1, 2}`` (i.e. excludes 0 and 3
    private/proprietary streams), matching oracle_support's behavior.

    Output columns: ``mjd, flux, flux_error, filter`` with filter in
    ``{ztfg, ztfr, ztfi}`` and flux in microjansky (zeropoint 23.9, AB-like).
    """
    rows = []
    for c in prv_candidates or []:
        programid = c.get("programid")
        if programid is not None:
            try:
                if int(programid) not in PUBLIC_PROGRAMIDS:
                    continue
            except (TypeError, ValueError):
                continue
        magpsf = c.get("magpsf")
        sigmapsf = c.get("sigmapsf")
        jd = c.get("jd")
        if not (_is_finite(magpsf) and _is_finite(sigmapsf) and _is_finite(jd)):
            continue
        if sigmapsf is None or float(sigmapsf) <= 0:
            continue
        filt = _resolve_band(c)
        if filt is None:
            continue
        isdiffpos = c.get("isdiffpos")
        if isdiffpos is False or (
            isinstance(isdiffpos, str) and isdiffpos.lower() in {"f", "0", "false", "-"}
        ):
            continue
        mjd = float(jd) - 2400000.5
        mag = float(magpsf)
        sigma = float(sigmapsf)
        flux = 10.0 ** (-0.4 * (mag - ZTF_FRITZ_ZP))
        flux_error = flux * sigma * LOG10 / 2.5
        rows.append(
            {
                "mjd": mjd,
                "flux": float(flux),
                "flux_error": float(flux_error),
                "filter": filt,
            }
        )
    df = pd.DataFrame(rows, columns=["mjd", "flux", "flux_error", "filter"])
    if df.empty:
        return df
    return df.sort_values("mjd").reset_index(drop=True)


def run_tempo(
    ztf_id,
    prv_candidates,
    *,
    delta_t_hours=12.0,
    window_start_day=0.0,
    window_end_day=100.0,
):
    """Run TEMPO classification on a ZTF alert's prv_candidates.

    Returns the helper's full result dict, or ``None`` if input is unusable.
    """
    df = prv_candidates_to_photometry_df(prv_candidates)
    if df.empty:
        logger.warning("[%s] no usable prv_candidates after conversion", ztf_id)
        return None

    bundle, state = _load_bundle()

    from tempo_support._tempo_helpers import (
        infer_from_photometry_dataframe,
        select_light_curve_window,
    )

    selected_df, selection = select_light_curve_window(
        df,
        window_start_day=window_start_day,
        window_end_day=window_end_day,
        reference_mode="first_point_in_input",
    )
    if selected_df.empty:
        logger.warning(
            "[%s] all photometry removed by window (%s, %s)",
            ztf_id,
            window_start_day,
            window_end_day,
        )
        return None

    logger.info(
        "[%s] running TEMPO (rows_in=%d, rows_in_window=%d, horizon=%.1fd)",
        ztf_id,
        selection["rows_in"],
        selection["rows_after_window"],
        selection.get("selected_horizon_days") or 0.0,
    )

    return infer_from_photometry_dataframe(
        bundle,
        selected_df,
        runtime_state=state,
        object_id=str(ztf_id),
        already_merged=False,
        delta_t_hours=delta_t_hours,
    )


def leaf_class_probs(result):
    """Extract a {class_name: probability} dict from a TEMPO result."""
    leaf = result["leaf"]
    return dict(zip(leaf["class_names"], [float(p) for p in leaf["probs_final"]]))
