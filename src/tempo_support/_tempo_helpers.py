from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nbformat
import numpy as np
import pandas as pd
import requests
import torch
from matplotlib import patheffects as pe
from matplotlib.colors import to_rgba

BUNDLE_ROOT = Path(__file__).resolve().parent

from tempo_support.photometry_edl.calibration import EvidenceTemperatureScaler
from tempo_support.photometry_edl.config import Config
from tempo_support.photometry_edl.data import FeatureStats, _global_features_from_sequence, build_event_tensor, global_feature_dim
from tempo_support.photometry_edl.models import EventTransformerEncoder, EvidentialClassifier
from tempo_support.photometry_edl.plotting import NORD, marker_colors, setup_mpl_paper, style_axes_inward
from tempo_support.photometry_edl.postprocess import adjust_for_priors, fit_prob_thresholds, predict_with_thresholds
from tempo_support.photometry_edl.report import (
    _aggregate_leaf_alpha,
    _aggregate_leaf_columns,
    _apply_ood_reference,
    _apply_uncertainty_fusion,
    _build_keep_mask,
    _choose_uncertainty_gate,
    _fit_ood_reference,
    _fit_uncertainty_fusion,
    _score_map_from_alpha,
    _uncertainty_score_dict,
    collect_evidential,
)
from tempo_support.photometry_edl.taxonomy import Taxonomy, get_taxonomy
from tempo_support.photometry_edl.train import make_loaders
from tempo_support.photometry_edl.uncertainty import dirichlet_mean, dirichlet_std, expected_categorical_entropy, mutual_information, predictive_entropy, total_uncertainty_trace, vacuity


BAND2ID = {'ztfg': 0, 'ztfr': 1, 'ztfi': 2}
FILTER_COLORS = {
    'ztfg': marker_colors['ztfg'],
    'ztfr': marker_colors['ztfr'],
    'ztfi': marker_colors['ztfi'],
}
LOG_CONST = 1.0 / np.log(10.0)


@dataclass
class LoadedTempoBundle:
    run_dir: Path
    report_dir: Path
    config: Config
    taxonomy: Taxonomy
    stats: FeatureStats
    model: EvidentialClassifier
    device: torch.device


@dataclass
class RuntimePostprocessState:
    temperature: Optional[float]
    use_temperature: bool
    use_prior_adjust: bool
    prob_thresholds: np.ndarray
    pi_train: Optional[np.ndarray]
    pi_deploy_val: Optional[np.ndarray]
    leaf_chosen: Dict[str, Any]
    uncertainty_objective: str
    uncertainty_beta: float
    target_coverage: float
    uncertainty_gate_strategy: str
    uncertainty_fusion: Optional[Dict[str, Any]]
    hierarchy_levels: List[Dict[str, Any]]
    ood_reference: Optional[Dict[str, Any]]


def _jsonify(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonify(v) for v in x]
    if isinstance(x, tuple):
        return [_jsonify(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        xf = float(x)
        return None if math.isnan(xf) else xf
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def extract_fritz_token_from_legacy_notebook(legacy_notebook_path: str | Path) -> str:
    path = Path(legacy_notebook_path)
    nb = nbformat.read(path, as_version=4)
    pat = re.compile(r'TOKEN\s*=\s*["\']([^"\']+)["\']')
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        m = pat.search(cell.source)
        if m:
            return m.group(1)
    raise RuntimeError(f'Could not find TOKEN assignment in {path}')


def fetch_fritz_photometry(obj_id: str, token: str, *, base_url: str = 'https://fritz.science', timeout: float = 30.0) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/sources/{obj_id}/photometry"
    headers = {'Authorization': f'token {token}'}
    params = {'format': 'flux', 'individualOrSeries': 'individual'}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get('data')
    if not isinstance(data, list):
        raise RuntimeError(f'Unexpected Fritz payload for {obj_id}: {payload}')
    return data


def _metadata_contains_forced_marker(x: Any) -> bool:
    if isinstance(x, dict):
        return any(_metadata_contains_forced_marker(k) or _metadata_contains_forced_marker(v) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return any(_metadata_contains_forced_marker(v) for v in x)
    if isinstance(x, str):
        return 'forced' in x.lower()
    return False


def infer_forced_photometry_flag(obs: Dict[str, Any]) -> Tuple[Optional[bool], Optional[str]]:
    """Infer whether a Fritz photometry row is explicitly marked as forced.

    The current Fritz payload often does not expose a dedicated forced-photometry field.
    We therefore distinguish:
      - True / False when an explicit metadata signal is present
      - None when the payload does not tell us
    """
    direct_keys = [
        'forced',
        'is_forced',
        'isForced',
        'forcedphot',
        'forced_phot',
        'is_forced_photometry',
    ]
    for key in direct_keys:
        if key in obs and obs.get(key) is not None:
            return bool(obs.get(key)), f'explicit:{key}'

    origin = obs.get('origin')
    if isinstance(origin, str) and origin:
        lo = origin.lower()
        if 'forced' in lo:
            return True, 'origin'
        return False, 'origin_nonforced'

    altdata = obs.get('altdata')
    if altdata is not None and _metadata_contains_forced_marker(altdata):
        return True, 'altdata_marker'

    for key, value in obs.items():
        if key in {'altdata', 'origin'}:
            continue
        if isinstance(key, str) and 'forced' in key.lower():
            return bool(value), f'key:{key}'

    return None, None


def build_fritz_photometry_dataframe(raw_data: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for obs in raw_data:
        filt = str(obs.get('filter', ''))
        if filt not in BAND2ID:
            continue
        flux = obs.get('flux')
        fluxerr = obs.get('fluxerr', obs.get('flux_error'))
        mjd = obs.get('mjd')
        if flux is None or fluxerr is None or mjd is None:
            continue
        try:
            flux = float(flux)
            fluxerr = float(fluxerr)
            mjd = float(mjd)
        except Exception:
            continue
        if not np.isfinite(flux) or not np.isfinite(fluxerr) or not np.isfinite(mjd):
            continue
        if fluxerr <= 0:
            continue
        forced_flag, forced_reason = infer_forced_photometry_flag(obs)
        snr = obs.get('snr')
        try:
            snr = float(snr) if snr is not None else None
        except Exception:
            snr = None
        rows.append({
            'mjd': mjd,
            'flux': flux,
            'flux_error': fluxerr,
            'filter': filt,
            'snr': snr,
            'zp': obs.get('zp'),
            'phot_id': obs.get('id'),
            'instrument_name': obs.get('instrument_name'),
            'created_at': obs.get('created_at'),
            'origin': obs.get('origin'),
            'groups': obs.get('groups'),
            'forced_flag': forced_flag,
            'forced_flag_reason': forced_reason,
            'has_explicit_forced_flag': forced_flag is not None,
            'raw': obs,
        })
    return pd.DataFrame(rows).sort_values('mjd').reset_index(drop=True)


def select_fritz_photometry(
    df: pd.DataFrame,
    *,
    forced_policy: str = 'keep_all',
    window_start_day: Optional[float] = 0.0,
    window_end_day: Optional[float] = 100.0,
    reference_mode: str = 'first_point_in_input',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply point-selection policies before merge/inference.

    Parameters
    ----------
    forced_policy:
        - keep_all: use all usable points
        - drop_explicit_forced: remove only rows explicitly marked as forced
        - keep_only_explicit_non_forced: keep only rows explicitly marked as non-forced
    window_start_day, window_end_day:
        Relative day window. Set both to None to keep the full selected light curve.
    reference_mode:
        - first_point_in_input: anchor window to the earliest point before filtering
        - first_point_after_forced_filter: anchor after forced-phot selection
    """
    if df.empty:
        return df.copy(), {
            'rows_in': 0,
            'rows_after_forced_policy': 0,
            'rows_after_window': 0,
            'forced_policy': forced_policy,
            'forced_metadata_available': False,
            'forced_rows_explicit_true': 0,
            'forced_rows_explicit_false': 0,
            'window_start_day': window_start_day,
            'window_end_day': window_end_day,
            'reference_mode': reference_mode,
            'window_reference_mjd': None,
        }

    out = df.copy().sort_values('mjd').reset_index(drop=True)
    explicit = out['forced_flag'].notna()
    forced_true = explicit & out['forced_flag'].astype(bool)
    forced_false = explicit & (~out['forced_flag'].astype(bool))

    if forced_policy == 'keep_all':
        out = out.copy()
    elif forced_policy == 'drop_explicit_forced':
        out = out.loc[~forced_true].copy()
    elif forced_policy == 'keep_only_explicit_non_forced':
        out = out.loc[forced_false].copy()
    else:
        raise ValueError(f'Unknown forced_policy={forced_policy}')
    rows_after_forced_policy = int(len(out))

    if out.empty:
        return out.reset_index(drop=True), {
            'rows_in': int(len(df)),
            'rows_after_forced_policy': rows_after_forced_policy,
            'rows_after_window': 0,
            'forced_policy': forced_policy,
            'forced_metadata_available': bool(explicit.any()),
            'forced_rows_explicit_true': int(forced_true.sum()),
            'forced_rows_explicit_false': int(forced_false.sum()),
            'window_start_day': window_start_day,
            'window_end_day': window_end_day,
            'reference_mode': reference_mode,
            'window_reference_mjd': None,
        }

    if reference_mode == 'first_point_in_input':
        ref_mjd = float(df['mjd'].min())
    elif reference_mode == 'first_point_after_forced_filter':
        ref_mjd = float(out['mjd'].min())
    else:
        raise ValueError(f'Unknown reference_mode={reference_mode}')

    out['days_from_reference'] = out['mjd'].astype(float) - ref_mjd
    if window_start_day is not None:
        out = out.loc[out['days_from_reference'] >= float(window_start_day)].copy()
    if window_end_day is not None:
        out = out.loc[out['days_from_reference'] <= float(window_end_day)].copy()
    out = out.sort_values('mjd').reset_index(drop=True)

    return out, {
        'rows_in': int(len(df)),
        'rows_after_forced_policy': rows_after_forced_policy,
        'rows_after_window': int(len(out)),
        'forced_policy': forced_policy,
        'forced_metadata_available': bool(explicit.any()),
        'forced_rows_explicit_true': int(forced_true.sum()),
        'forced_rows_explicit_false': int(forced_false.sum()),
        'window_start_day': None if window_start_day is None else float(window_start_day),
        'window_end_day': None if window_end_day is None else float(window_end_day),
        'reference_mode': reference_mode,
        'window_reference_mjd': float(ref_mjd),
        'selected_horizon_days': float(out['days_from_reference'].max()) if len(out) else None,
    }


def select_light_curve_window(
    df: pd.DataFrame,
    *,
    window_start_day: Optional[float] = 0.0,
    window_end_day: Optional[float] = 100.0,
    reference_mode: str = 'first_point_in_input',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select a relative time window from a usable photometry dataframe."""
    if df.empty:
        return df.copy(), {
            'rows_in': 0,
            'rows_after_window': 0,
            'window_start_day': window_start_day,
            'window_end_day': window_end_day,
            'reference_mode': reference_mode,
            'window_reference_mjd': None,
        }

    out = df.copy().sort_values('mjd').reset_index(drop=True)
    if reference_mode == 'first_point_in_input':
        ref_mjd = float(out['mjd'].min())
    else:
        raise ValueError(f'Unknown reference_mode={reference_mode}')

    out['days_from_reference'] = out['mjd'].astype(float) - ref_mjd
    if window_start_day is not None:
        out = out.loc[out['days_from_reference'] >= float(window_start_day)].copy()
    if window_end_day is not None:
        out = out.loc[out['days_from_reference'] <= float(window_end_day)].copy()
    out = out.sort_values('mjd').reset_index(drop=True)

    return out, {
        'rows_in': int(len(df)),
        'rows_after_window': int(len(out)),
        'window_start_day': None if window_start_day is None else float(window_start_day),
        'window_end_day': None if window_end_day is None else float(window_end_day),
        'reference_mode': reference_mode,
        'window_reference_mjd': float(ref_mjd),
        'selected_horizon_days': float(out['days_from_reference'].max()) if len(out) else None,
    }


def _merge_weighted(t: np.ndarray, f: np.ndarray, e: np.ndarray, *, dt_days: float, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tout: List[float] = []
    fout: List[float] = []
    eout: List[float] = []
    i = 0
    n = len(t)
    while i < n:
        j = i
        while j + 1 < n and (t[j + 1] - t[i]) <= dt_days:
            j += 1
        w = 1.0 / np.maximum(e[i:j+1], eps)
        w = w / np.sum(w)
        tout.append(float(np.sum(w * t[i:j+1])))
        fout.append(float(np.sum(w * f[i:j+1])))
        eout.append(float(np.sum(w * e[i:j+1])))
        i = j + 1
    return np.asarray(tout), np.asarray(fout), np.asarray(eout)


def merge_fritz_photometry(df: pd.DataFrame, *, delta_t_hours: float = 12.0) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out: List[pd.DataFrame] = []
    dt_days = float(delta_t_hours) / 24.0
    for filt, grp in df.groupby('filter'):
        grp2 = grp.sort_values('mjd').reset_index(drop=True)
        t, f, e = _merge_weighted(
            grp2['mjd'].to_numpy(dtype=float),
            grp2['flux'].to_numpy(dtype=float),
            grp2['flux_error'].to_numpy(dtype=float),
            dt_days=dt_days,
        )
        out.append(pd.DataFrame({'mjd': t, 'flux': f, 'flux_error': e, 'filter': filt}))
    return pd.concat(out, ignore_index=True).sort_values('mjd').reset_index(drop=True)


def build_training_like_event_array(df: pd.DataFrame, *, drop_i_band: bool = False) -> np.ndarray:
    if df.empty:
        raise ValueError('Cannot build event array from empty photometry')
    use = df.copy()
    if drop_i_band:
        use = use[use['filter'] != 'ztfi'].copy()
    use = use.sort_values('mjd').reset_index(drop=True)
    if use.empty:
        raise ValueError('All points were removed by drop_i_band')
    t0 = float(use['mjd'].iloc[0])
    dt = use['mjd'].to_numpy(dtype=np.float32) - np.float32(t0)
    dt_prev = np.diff(np.r_[np.float32(t0), use['mjd'].to_numpy(dtype=np.float32)]).astype(np.float32)
    flux = np.clip(use['flux'].to_numpy(dtype=np.float32), 1e-6, None)
    fluxerr = np.clip(use['flux_error'].to_numpy(dtype=np.float32), 1e-12, None)
    logf = np.log10(flux).astype(np.float32)
    logf_err = (fluxerr * np.float32(LOG_CONST) / flux).astype(np.float32)
    band_id = use['filter'].map(BAND2ID).to_numpy(dtype=np.int64).astype(np.float32)
    return np.stack([dt, dt_prev, band_id, logf, logf_err], axis=1).astype(np.float32)


def load_tempo_bundle(run_dir: str | Path, *, report_dir: str | Path, device: Optional[str | torch.device] = None) -> LoadedTempoBundle:
    run_dir = Path(run_dir)
    report_dir = Path(report_dir)
    cfg = Config.from_dict(json.loads((run_dir / 'config.json').read_text()))
    taxonomy = get_taxonomy(getattr(cfg, 'taxonomy_preset', 'default'))
    stats_candidates = [
        run_dir / cfg.stats_file,
        run_dir.parent / cfg.stats_file,
        report_dir.parent / cfg.stats_file,
        BUNDLE_ROOT / cfg.stats_file,
        Path(cfg.data_dir) / cfg.stats_file,
    ]
    stats_path = next((p for p in stats_candidates if p.exists()), None)
    if stats_path is None:
        tried = '\n'.join(str(p) for p in stats_candidates)
        raise FileNotFoundError(f'Could not resolve stats file {cfg.stats_file}. Tried:\n{tried}')
    stats = FeatureStats.load(stats_path)
    dev = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    encoder = EventTransformerEncoder(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        band_mode=cfg.band_mode,
        band_embed_dim=cfg.band_embed_dim,
        time_encoding=cfg.time_encoding,
    )
    model = EvidentialClassifier(
        encoder,
        num_classes=taxonomy.num_classes,
        dropout=cfg.dropout,
        pool=cfg.pool,
        use_global_features=cfg.use_global_features,
        global_dim=global_feature_dim(cfg.global_feature_set),
        global_hidden_dim=cfg.global_hidden_dim,
    ).to(dev)
    model.load_state_dict(torch.load(run_dir / 'best_evidential.pt', map_location=dev))
    model.eval()
    return LoadedTempoBundle(run_dir=run_dir, report_dir=report_dir, config=cfg, taxonomy=taxonomy, stats=stats, model=model, device=dev)


def _prepare_single_inputs(bundle: LoadedTempoBundle, arr: np.ndarray):
    seq = build_event_tensor(arr, band_mode=bundle.config.band_mode)
    cont = (seq[:, :4].float() - bundle.stats.mean.view(1, 4)) / (bundle.stats.std.view(1, 4) + 1e-8)
    if bundle.config.band_mode == 'onehot':
        x = torch.cat([cont, seq[:, 4:].float()], dim=-1).unsqueeze(0)
    else:
        x = torch.cat([cont, seq[:, 4:5].float()], dim=-1).unsqueeze(0)
    pad_mask = torch.zeros((1, x.shape[1]), dtype=torch.bool)
    g = None
    if bundle.config.use_global_features:
        g = _global_features_from_sequence(seq, band_mode=bundle.config.band_mode, feature_set=bundle.config.global_feature_set).unsqueeze(0)
    return x.to(bundle.device), pad_mask.to(bundle.device), g.to(bundle.device) if g is not None else None, seq


def _load_saved_report_state(bundle: LoadedTempoBundle) -> RuntimePostprocessState:
    report_dir = bundle.report_dir
    post = json.loads((report_dir / 'postprocess.json').read_text())
    unc = json.loads((report_dir / 'uncertainty_metrics.json').read_text())
    temp_path = report_dir / 'temperature_scaler.json'
    priors_path = report_dir / 'priors.json'
    fusion_path = report_dir / 'uncertainty_fusion.json'
    report_name = report_dir.name.lower()
    temperature = float(json.loads(temp_path.read_text())['temperature']) if temp_path.exists() else None
    pi_train = None
    pi_deploy_val = None
    if priors_path.exists():
        pri = json.loads(priors_path.read_text())
        pi_train = np.asarray(pri['pi_train'], dtype=np.float64)
        pi_deploy_val = np.asarray(pri['pi_deploy_val'], dtype=np.float64)
    return RuntimePostprocessState(
        temperature=temperature,
        use_temperature=temperature is not None and 'no_temp' not in report_name,
        use_prior_adjust=pi_train is not None and pi_deploy_val is not None and 'no_prior' not in report_name,
        prob_thresholds=np.asarray(post['thresholds'], dtype=np.float64),
        pi_train=pi_train,
        pi_deploy_val=pi_deploy_val,
        leaf_chosen={
            'name': unc['selected']['score'],
            'strategy': unc['selected'].get('strategy', post.get('uncertainty_gate_strategy', 'global')),
            'threshold': post.get('uncertainty_cut'),
            'thresholds_by_pred_class': post.get('uncertainty_cut_by_pred_class'),
            'coverage': unc['selected'].get('coverage_val'),
            'macro_fbeta_kept': post.get('val_macro_fbeta_at_cut'),
            'aurc': unc['selected'].get('aurc_val'),
            'criterion': None,
        },
        uncertainty_objective=str(unc.get('objective', 'aurc')),
        uncertainty_beta=float(unc.get('beta', 1.0)),
        target_coverage=float(unc.get('target_coverage', 0.85)),
        uncertainty_gate_strategy=str(post.get('uncertainty_gate_strategy', unc['selected'].get('strategy', 'global'))),
        uncertainty_fusion=json.loads(fusion_path.read_text()) if fusion_path.exists() else None,
        hierarchy_levels=[],
        ood_reference=None,
    )


def build_runtime_postprocess_state(bundle: LoadedTempoBundle, *, cache_json: Optional[str | Path] = None, force_recompute: bool = False) -> RuntimePostprocessState:
    cache_path = Path(cache_json) if cache_json is not None else (bundle.report_dir / 'single_object_runtime_cache.json')
    if cache_path.exists() and not force_recompute:
        cached = json.loads(cache_path.read_text())
        return RuntimePostprocessState(
            temperature=cached.get('temperature'),
            use_temperature=bool(cached.get('use_temperature', False)),
            use_prior_adjust=bool(cached.get('use_prior_adjust', False)),
            prob_thresholds=np.asarray(cached['prob_thresholds'], dtype=np.float64),
            pi_train=np.asarray(cached['pi_train'], dtype=np.float64) if cached.get('pi_train') is not None else None,
            pi_deploy_val=np.asarray(cached['pi_deploy_val'], dtype=np.float64) if cached.get('pi_deploy_val') is not None else None,
            leaf_chosen=cached['leaf_chosen'],
            uncertainty_objective=str(cached['uncertainty_objective']),
            uncertainty_beta=float(cached['uncertainty_beta']),
            target_coverage=float(cached['target_coverage']),
            uncertainty_gate_strategy=str(cached['uncertainty_gate_strategy']),
            uncertainty_fusion=cached.get('uncertainty_fusion'),
            hierarchy_levels=cached.get('hierarchy_levels', []),
            ood_reference=cached.get('ood_reference'),
        )

    data_dir = Path(str(bundle.config.data_dir))
    required = [data_dir / 'manifest_train.csv', data_dir / 'manifest_val.csv', data_dir / 'feature_stats_day100.npz']
    if not all(p.exists() for p in required):
        missing = [str(p) for p in required if not p.exists()]
        raise RuntimeError(
            'Runtime cache is missing and validation-based reconstruction is not possible in this standalone bundle.\n'
            f'Missing required training-data artifacts: {missing}\n'
            'For portable inference, keep the bundled runtime cache file under model_bundle/runtime/.'
        )

    state = _load_saved_report_state(bundle)
    _, val_loader, _, _, _ = make_loaders(bundle.config, bundle.taxonomy)
    val = collect_evidential(bundle.model, val_loader, device=bundle.device)

    p_val_cur = np.asarray(val['probs'], dtype=np.float64)
    if state.use_temperature and state.temperature is not None:
        p_val_cur = EvidenceTemperatureScaler(temperature=state.temperature).transform(val['evidence'])
    p_val_adj = adjust_for_priors(p_val_cur, state.pi_train, state.pi_deploy_val) if state.use_prior_adjust and state.pi_train is not None and state.pi_deploy_val is not None else p_val_cur.copy()

    thr = fit_prob_thresholds(val['y'], p_val_adj)
    if thr.shape == state.prob_thresholds.shape:
        thr = state.prob_thresholds
    pred_val_thr = predict_with_thresholds(p_val_adj, thr)

    score_map_val = _uncertainty_score_dict(val)
    if state.uncertainty_fusion is None:
        state.uncertainty_fusion = _fit_uncertainty_fusion(score_map_val, val['y'], pred_val_thr)
    score_map_val['fused_uncertainty'] = _apply_uncertainty_fusion(score_map_val, state.uncertainty_fusion)
    keep_val = _build_keep_mask(np.asarray(score_map_val[state.leaf_chosen['name']], dtype=float), pred_val_thr, state.leaf_chosen)

    state.ood_reference = _jsonify(_fit_ood_reference(
        score_map_val,
        inlier_mask=(pred_val_thr == val['y']) & keep_val,
        feature_keys=list(score_map_val.keys()),
        score_quantile=0.995,
        vote_quantile=0.995,
        min_votes=2,
    ))

    levels_out: List[Dict[str, Any]] = []
    leaf_names = list(bundle.taxonomy.broad_classes)
    for spec in bundle.taxonomy.hierarchy_level_specs():
        level_idx = int(spec['level_index'])
        level_name = str(spec['level_name'])
        node_names = list(spec['node_names'])
        leaf_to_node = np.asarray([int(spec['broad_to_node'][leaf]) for leaf in leaf_names], dtype=int)
        n_nodes = len(node_names)
        y_val_level = leaf_to_node[np.asarray(val['y'], dtype=int)]
        p_val_level = _aggregate_leaf_columns(p_val_adj, leaf_to_node, n_nodes)
        a_val_level = _aggregate_leaf_alpha(val['alpha'], leaf_to_node, n_nodes)
        pred_val_level = p_val_level.argmax(axis=1)
        score_level = _score_map_from_alpha(a_val_level)
        chosen = _choose_uncertainty_gate(
            y_val_level,
            pred_val_level,
            score_level,
            objective=state.uncertainty_objective,
            target_coverage=state.target_coverage,
            beta=state.uncertainty_beta,
            labels=np.arange(n_nodes),
            strategy=state.uncertainty_gate_strategy,
            min_true_class_coverage=None,
        )
        if chosen is None and state.uncertainty_gate_strategy != 'global':
            chosen = _choose_uncertainty_gate(
                y_val_level,
                pred_val_level,
                score_level,
                objective=state.uncertainty_objective,
                target_coverage=state.target_coverage,
                beta=state.uncertainty_beta,
                labels=np.arange(n_nodes),
                strategy='global',
                min_true_class_coverage=None,
            )
        if chosen is None:
            raise RuntimeError(f'Failed to rebuild hierarchy gate for level {level_name}')
        levels_out.append(_jsonify({
            'level_index': level_idx,
            'level_name': level_name,
            'node_names': node_names,
            'leaf_to_node_index': {leaf: int(spec['broad_to_node'][leaf]) for leaf in leaf_names},
            'chosen': chosen,
        }))
    state.hierarchy_levels = levels_out

    cache_path.write_text(json.dumps(_jsonify({
        'temperature': state.temperature,
        'use_temperature': state.use_temperature,
        'use_prior_adjust': state.use_prior_adjust,
        'prob_thresholds': state.prob_thresholds,
        'pi_train': state.pi_train,
        'pi_deploy_val': state.pi_deploy_val,
        'leaf_chosen': state.leaf_chosen,
        'uncertainty_objective': state.uncertainty_objective,
        'uncertainty_beta': state.uncertainty_beta,
        'target_coverage': state.target_coverage,
        'uncertainty_gate_strategy': state.uncertainty_gate_strategy,
        'uncertainty_fusion': state.uncertainty_fusion,
        'hierarchy_levels': state.hierarchy_levels,
        'ood_reference': state.ood_reference,
    }), indent=2))
    return state


def _apply_calibration(alpha: np.ndarray, evidence: np.ndarray, state: RuntimePostprocessState) -> Dict[str, np.ndarray]:
    probs_base = dirichlet_mean(torch.from_numpy(alpha.astype(np.float32))).cpu().numpy()
    std_base = dirichlet_std(torch.from_numpy(alpha.astype(np.float32))).cpu().numpy()
    alpha_cal = np.asarray(alpha, dtype=np.float64)
    probs_cal = np.asarray(probs_base, dtype=np.float64)
    std_cal = np.asarray(std_base, dtype=np.float64)
    if state.use_temperature and state.temperature is not None:
        alpha_cal = EvidenceTemperatureScaler(temperature=state.temperature).transform_alpha(alpha)
        probs_cal = alpha_cal / alpha_cal.sum(axis=1, keepdims=True)
        std_cal = dirichlet_std(torch.from_numpy(alpha_cal.astype(np.float32))).cpu().numpy()
    probs_final = adjust_for_priors(probs_cal, state.pi_train, state.pi_deploy_val) if state.use_prior_adjust and state.pi_train is not None and state.pi_deploy_val is not None else probs_cal.copy()
    return {
        'alpha_base': np.asarray(alpha, dtype=np.float64),
        'probs_base': np.asarray(probs_base, dtype=np.float64),
        'std_base': np.asarray(std_base, dtype=np.float64),
        'alpha_calibrated': np.asarray(alpha_cal, dtype=np.float64),
        'probs_calibrated': np.asarray(probs_cal, dtype=np.float64),
        'std_calibrated': np.asarray(std_cal, dtype=np.float64),
        'probs_final': np.asarray(probs_final, dtype=np.float64),
    }


def infer_event_array(bundle: LoadedTempoBundle, arr: np.ndarray, *, runtime_state: RuntimePostprocessState, object_id: str = 'unknown') -> Dict[str, Any]:
    x, pad_mask, g, seq = _prepare_single_inputs(bundle, arr)
    with torch.no_grad():
        evidence_t = bundle.model(x, pad_mask, g)
        alpha_t = bundle.model.alpha_from_evidence(evidence_t)
    evidence = evidence_t.detach().cpu().numpy()
    alpha = alpha_t.detach().cpu().numpy()
    pack = _apply_calibration(alpha, evidence, runtime_state)
    probs_final = pack['probs_final']
    class_names = list(bundle.taxonomy.broad_classes)
    pred_thr = int(predict_with_thresholds(probs_final, runtime_state.prob_thresholds)[0])
    pred_argmax = int(np.argmax(probs_final[0]))

    score_map = {
        'vacuity': vacuity(alpha_t).cpu().numpy(),
        'entropy': predictive_entropy(dirichlet_mean(alpha_t)).cpu().numpy(),
        'expected_entropy': expected_categorical_entropy(alpha_t).cpu().numpy(),
        'mi': mutual_information(alpha_t).cpu().numpy(),
        'trace_uncertainty': total_uncertainty_trace(alpha_t).cpu().numpy(),
    }
    score_map = _uncertainty_score_dict(score_map)
    if runtime_state.uncertainty_fusion is not None:
        score_map['fused_uncertainty'] = _apply_uncertainty_fusion(score_map, runtime_state.uncertainty_fusion)

    chosen_name = str(runtime_state.leaf_chosen['name'])
    keep_leaf = bool(_build_keep_mask(np.asarray(score_map[chosen_name], dtype=float), np.asarray([pred_thr], dtype=int), runtime_state.leaf_chosen)[0])
    ood = None
    if runtime_state.ood_reference is not None:
        ood_score, ood_votes, ood_flag = _apply_ood_reference(score_map, runtime_state.ood_reference)
        ood = {'score': float(ood_score[0]), 'votes': int(ood_votes[0]), 'flag': bool(ood_flag[0])}

    hierarchy_rows: List[Dict[str, Any]] = []
    for level in runtime_state.hierarchy_levels:
        node_names = list(level['node_names'])
        leaf_to_node = np.asarray([level['leaf_to_node_index'][leaf] for leaf in class_names], dtype=int)
        p_level_final = _aggregate_leaf_columns(pack['probs_final'], leaf_to_node, len(node_names))
        p_level_cal = _aggregate_leaf_columns(pack['probs_calibrated'], leaf_to_node, len(node_names))
        p_level_base = _aggregate_leaf_columns(pack['probs_base'], leaf_to_node, len(node_names))
        a_level_base = _aggregate_leaf_alpha(pack['alpha_base'], leaf_to_node, len(node_names))
        a_level_cal = _aggregate_leaf_alpha(pack['alpha_calibrated'], leaf_to_node, len(node_names))
        std_level_base = dirichlet_std(torch.from_numpy(a_level_base.astype(np.float32))).cpu().numpy()
        std_level_cal = dirichlet_std(torch.from_numpy(a_level_cal.astype(np.float32))).cpu().numpy()
        pred_level = int(np.argmax(p_level_final[0]))
        score_level = _score_map_from_alpha(a_level_base)
        keep_level = bool(_build_keep_mask(np.asarray(score_level[level['chosen']['name']], dtype=float), np.asarray([pred_level], dtype=int), level['chosen'])[0])
        hierarchy_rows.append(_jsonify({
            'level_index': level['level_index'],
            'level_name': level['level_name'],
            'node_names': node_names,
            'alpha_base': a_level_base[0],
            'alpha_calibrated': a_level_cal[0],
            'probs_base': p_level_base[0],
            'probs_calibrated': p_level_cal[0],
            'probs_final': p_level_final[0],
            'prob_std_base': std_level_base[0],
            'prob_std_calibrated': std_level_cal[0],
            'final_prob_variance_defined': not bool(runtime_state.use_prior_adjust),
            'pred_id': pred_level,
            'pred_name': node_names[pred_level],
            'top_prob': float(p_level_final[0, pred_level]),
            'kept': keep_level,
            'chosen_uncertainty_name': level['chosen']['name'],
            'chosen_uncertainty_value': float(np.asarray(score_level[level['chosen']['name']], dtype=float)[0]),
            'score_map': {k: float(np.asarray(v, dtype=float)[0]) for k, v in score_level.items()},
        }))

    kept_levels = [lv for lv in hierarchy_rows if lv['kept']]
    if keep_leaf:
        fallback = {'level_name': 'leaf', 'label': class_names[pred_thr], 'accepted': True}
    elif kept_levels:
        deepest = sorted(kept_levels, key=lambda d: d['level_index'], reverse=True)[0]
        fallback = {'level_name': deepest['level_name'], 'label': deepest['pred_name'], 'accepted': True}
    else:
        fallback = {'level_name': None, 'label': None, 'accepted': False}

    return _jsonify({
        'object_id': object_id,
        'n_events': int(arr.shape[0]),
        'final_horizon_days': float(arr[-1, 0]),
        'leaf': {
            'class_names': class_names,
            'evidence': evidence[0],
            'alpha_base': pack['alpha_base'][0],
            'alpha_calibrated': pack['alpha_calibrated'][0],
            'probs_base': pack['probs_base'][0],
            'probs_calibrated': pack['probs_calibrated'][0],
            'probs_final': pack['probs_final'][0],
            'prob_std_base': pack['std_base'][0],
            'prob_std_calibrated': pack['std_calibrated'][0],
            'final_prob_variance_defined': not bool(runtime_state.use_prior_adjust),
            'pred_argmax_id': pred_argmax,
            'pred_argmax_name': class_names[pred_argmax],
            'pred_thresholded_id': pred_thr,
            'pred_thresholded_name': class_names[pred_thr],
            'pred_thresholded_prob': float(pack['probs_final'][0, pred_thr]),
            'kept': keep_leaf,
            'uncertainty_selected_name': chosen_name,
            'uncertainty_selected_value': float(np.asarray(score_map[chosen_name], dtype=float)[0]),
            'score_map': {k: float(np.asarray(v, dtype=float)[0]) for k, v in score_map.items()},
            'ood': ood,
        },
        'hierarchy': hierarchy_rows,
        'decision': {
            'leaf_accept': keep_leaf,
            'fallback': fallback,
            'abstain_completely': not bool(fallback and fallback.get('accepted')),
        },
        'photometry': {'raw_event_array': arr.tolist(), 'token_tensor_unnormalized': seq.cpu().numpy().tolist()},
    })


def infer_from_photometry_dataframe(bundle: LoadedTempoBundle, photometry_df: pd.DataFrame, *, runtime_state: RuntimePostprocessState, object_id: str = 'unknown', already_merged: bool = False, delta_t_hours: float = 12.0) -> Dict[str, Any]:
    df = photometry_df.copy().sort_values('mjd').reset_index(drop=True)
    if not already_merged:
        df = merge_fritz_photometry(df, delta_t_hours=delta_t_hours)
    arr = build_training_like_event_array(df, drop_i_band=bundle.config.drop_i_band)
    out = infer_event_array(bundle, arr, runtime_state=runtime_state, object_id=object_id)
    out['photometry']['dataframe'] = df.to_dict(orient='records')
    return out


def compute_prefix_evolution(bundle: LoadedTempoBundle, photometry_df: pd.DataFrame, *, runtime_state: RuntimePostprocessState, object_id: str = 'unknown', already_merged: bool = False, delta_t_hours: float = 12.0) -> pd.DataFrame:
    df = photometry_df.copy().sort_values('mjd').reset_index(drop=True)
    if not already_merged:
        df = merge_fritz_photometry(df, delta_t_hours=delta_t_hours)
    t0 = float(df['mjd'].iloc[0])
    rows: List[Dict[str, Any]] = []
    for i in range(1, len(df) + 1):
        sub = df.iloc[:i].copy()
        arr = build_training_like_event_array(sub, drop_i_band=bundle.config.drop_i_band)
        result = infer_event_array(bundle, arr, runtime_state=runtime_state, object_id=object_id)
        row = {
            'prefix_index': int(i),
            'n_events': int(i),
            'horizon_days': float(sub['mjd'].iloc[-1] - t0),
            'leaf_pred_name': result['leaf']['pred_thresholded_name'],
            'leaf_pred_prob': result['leaf']['pred_thresholded_prob'],
            'leaf_kept': bool(result['leaf']['kept']),
            'leaf_selected_uncertainty': result['leaf']['uncertainty_selected_value'],
            'decision_label': result['decision']['fallback']['label'],
            'decision_level': result['decision']['fallback']['level_name'],
            'abstain_completely': bool(result['decision']['abstain_completely']),
        }
        for cname, p in zip(result['leaf']['class_names'], result['leaf']['probs_final']):
            row[f'final_prob_{cname}'] = float(p)
        for cname, p in zip(result['leaf']['class_names'], result['leaf']['probs_base']):
            row[f'base_prob_{cname}'] = float(p)
        for cname, p in zip(result['leaf']['class_names'], result['leaf']['probs_calibrated']):
            row[f'calibrated_prob_{cname}'] = float(p)
        for cname, s in zip(result['leaf']['class_names'], result['leaf']['prob_std_base']):
            row[f'base_std_{cname}'] = float(s)
        for cname, s in zip(result['leaf']['class_names'], result['leaf']['prob_std_calibrated']):
            row[f'calibrated_std_{cname}'] = float(s)
        for k, v in result['leaf']['score_map'].items():
            row[f'score_{k}'] = float(v)
        for lv in result['hierarchy']:
            slug = str(lv['level_name']).lower()
            row[f'{slug}_pred_name'] = lv['pred_name']
            row[f'{slug}_top_prob'] = float(lv['top_prob'])
            row[f'{slug}_kept'] = bool(lv['kept'])
            row[f'{slug}_selected_uncertainty_name'] = lv['chosen_uncertainty_name']
            row[f'{slug}_selected_uncertainty'] = float(lv['chosen_uncertainty_value'])
            for cname, p in zip(lv['node_names'], lv['probs_final']):
                row[f'{slug}_final_prob_{cname}'] = float(p)
            for cname, p in zip(lv['node_names'], lv['probs_calibrated']):
                row[f'{slug}_calibrated_prob_{cname}'] = float(p)
            for cname, s in zip(lv['node_names'], lv['prob_std_calibrated']):
                row[f'{slug}_calibrated_std_{cname}'] = float(s)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_prediction(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    leaf = result['leaf']
    for i, cname in enumerate(leaf['class_names']):
        rows.append({
            'class_name': cname,
            'prob_base': leaf['probs_base'][i],
            'prob_calibrated': leaf['probs_calibrated'][i],
            'prob_final': leaf['probs_final'][i],
            'prob_std_base': leaf['prob_std_base'][i],
            'prob_std_calibrated': leaf['prob_std_calibrated'][i],
            'alpha_base': leaf['alpha_base'][i],
            'alpha_calibrated': leaf['alpha_calibrated'][i],
            'is_thresholded_prediction': i == leaf['pred_thresholded_id'],
        })
    return pd.DataFrame(rows).sort_values('prob_final', ascending=False).reset_index(drop=True)


def summarize_hierarchy(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for lv in result['hierarchy']:
        for i, cname in enumerate(lv['node_names']):
            rows.append({
                'level_index': lv['level_index'],
                'level_name': lv['level_name'],
                'class_name': cname,
                'prob_base': lv['probs_base'][i],
                'prob_calibrated': lv['probs_calibrated'][i],
                'prob_final': lv['probs_final'][i],
                'prob_std_base': lv['prob_std_base'][i],
                'prob_std_calibrated': lv['prob_std_calibrated'][i],
                'alpha_base': lv['alpha_base'][i],
                'alpha_calibrated': lv['alpha_calibrated'][i],
                'is_prediction': i == lv['pred_id'],
                'kept': lv['kept'],
                'selected_uncertainty_name': lv['chosen_uncertainty_name'],
                'selected_uncertainty_value': lv['chosen_uncertainty_value'],
            })
    return pd.DataFrame(rows)


def get_result_level(result: Dict[str, Any], level_name: str) -> Dict[str, Any]:
    slug = str(level_name).lower()
    if slug == 'leaf':
        return result['leaf']
    for lv in result['hierarchy']:
        if str(lv['level_name']).lower() == slug:
            return lv
    raise KeyError(f'Unknown level_name={level_name}')


def plot_light_curve(raw_df: pd.DataFrame, *, selected_df: Optional[pd.DataFrame] = None, merged_df: Optional[pd.DataFrame] = None, object_id: Optional[str] = None):
    setup_mpl_paper(usetex=False)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    t0 = float(raw_df['mjd'].min())
    for filt in ['ztfg', 'ztfr', 'ztfi']:
        sub = raw_df[raw_df['filter'] == filt]
        col = FILTER_COLORS[filt]
        if not sub.empty:
            ax.errorbar(sub['mjd'] - t0, sub['flux'], yerr=sub['flux_error'], fmt='o', ms=3.0, lw=0.8, alpha=0.25, color=col, label=f'raw {filt}')
        if selected_df is not None:
            subs = selected_df[selected_df['filter'] == filt]
            if not subs.empty:
                ax.errorbar(subs['mjd'] - t0, subs['flux'], yerr=subs['flux_error'], fmt='o', ms=3.6, lw=1.0, alpha=0.85, color=col, label=f'selected {filt}')
        if merged_df is not None:
            subm = merged_df[merged_df['filter'] == filt]
            if not subm.empty:
                ax.errorbar(subm['mjd'] - t0, subm['flux'], yerr=subm['flux_error'], fmt='o', ms=4.5, mew=1.0, mfc='white', lw=1.0, color=col, label=f'merged {filt}')
    ax.set_xlabel('days since first detection')
    ax.set_ylabel('flux')
    ax.set_title(f'Fritz light curve: {object_id}' if object_id else 'Fritz light curve')
    style_axes_inward(ax, grid_y=True)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_dirichlet_top3_simplex(alpha_like: Sequence[float], class_names: Sequence[str], *, title: str = 'Raw Dirichlet posterior'):
    setup_mpl_paper(usetex=False)
    alpha = np.asarray(alpha_like, dtype=np.float64).reshape(-1)
    p_all = alpha / alpha.sum()
    top3 = np.argsort(p_all)[-3:][::-1]
    a3 = alpha[top3]
    names3 = [class_names[i] for i in top3]
    mean3 = a3 / a3.sum()
    A = np.array([0.0, 0.0]); B = np.array([1.0, 0.0]); C = np.array([0.5, np.sqrt(3.0)/2.0])
    grid = np.linspace(1e-4, 1.0-1e-4, 180)
    u, v = np.meshgrid(grid, grid, indexing='ij')
    w = 1.0 - u - v
    mask = (u > 0) & (v > 0) & (w > 0)
    U, V, W = u[mask], v[mask], w[mask]
    xy = U[:, None] * A + V[:, None] * B + W[:, None] * C
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    from scipy.special import gammaln
    log_z = gammaln(np.sum(a3)) - np.sum(gammaln(a3))
    log_pdf = (a3[0]-1.0)*np.log(U) + (a3[1]-1.0)*np.log(V) + (a3[2]-1.0)*np.log(W) - log_z
    dens = np.exp(log_pdf)
    fig, axs = plt.subplots(1, 2, figsize=(13.6, 6.0), dpi=180)
    logdens = np.log(dens + 1e-300)
    for ax, field, cmap, cbar_label, subtitle in [
        (axs[0], dens, 'magma', 'density', 'density'),
        (axs[1], logdens, 'magma', 'log density', 'log density'),
    ]:
        tpc = ax.tripcolor(tri, field, shading='gouraud', cmap=cmap)
        fig.colorbar(tpc, ax=ax, shrink=0.72, label=cbar_label)
        ax.tricontour(tri, field, levels=8, colors='white', alpha=0.45, linewidths=0.8)
        outline = np.array([A, B, C, A])
        ax.plot(outline[:, 0], outline[:, 1], color='#222', lw=1.5)
        ax.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], c='#222', s=20)
        ax.text(A[0]-0.04, A[1]-0.04, names3[0], ha='right', va='top', fontsize=11, path_effects=[pe.withStroke(linewidth=2, foreground='white', alpha=0.7)])
        ax.text(B[0]+0.04, B[1]-0.04, names3[1], ha='left', va='top', fontsize=11, path_effects=[pe.withStroke(linewidth=2, foreground='white', alpha=0.7)])
        ax.text(C[0], C[1]+0.04, names3[2], ha='center', va='bottom', fontsize=11, path_effects=[pe.withStroke(linewidth=2, foreground='white', alpha=0.7)])
        mean_xy = mean3[0]*A + mean3[1]*B + mean3[2]*C
        ax.scatter(mean_xy[0], mean_xy[1], marker='v', s=60, facecolor='white', edgecolor='#111', zorder=5)
        ax.set_title(f'{title}: {subtitle}')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, np.sqrt(3.0)/2.0 + 0.06)
    fig.tight_layout()
    return fig


def plot_prediction_bars(result: Dict[str, Any], *, title: Optional[str] = None):
    setup_mpl_paper(usetex=False)
    leaf = result['leaf']
    cls = leaf['class_names']
    x = np.arange(len(cls)); width = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    ax.bar(x - width, leaf['probs_base'], width=width, color=NORD['nord10'], alpha=0.85, label='base Dirichlet mean')
    ax.bar(x, leaf['probs_calibrated'], width=width, color=NORD['nord9'], alpha=0.85, label='temperature-calibrated')
    ax.bar(x + width, leaf['probs_final'], width=width, color=NORD['nord14'], alpha=0.9, label='deployed final')
    ax.set_xticks(x); ax.set_xticklabels(cls); ax.set_ylim(0.0, 1.02)
    ax.set_ylabel('probability')
    ax.set_title(title or f"Prediction summary for {result['object_id']}")
    style_axes_inward(ax, grid_y=True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_level_probability_bars(
    level: Dict[str, Any],
    *,
    title: Optional[str] = None,
    use_calibrated: bool = True,
    show_final_markers: bool = True,
):
    setup_mpl_paper(usetex=False)
    class_names = list(level['class_names']) if 'class_names' in level else list(level['node_names'])
    probs = np.asarray(level['probs_calibrated' if use_calibrated else 'probs_base'], dtype=float)
    errs = np.asarray(level['prob_std_calibrated' if use_calibrated else 'prob_std_base'], dtype=float)
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    ax.bar(x, probs, color=NORD['nord10'], alpha=0.84, yerr=errs, capsize=4, ecolor='#111111', label='calibrated posterior mean ± std' if use_calibrated else 'base posterior mean ± std')
    if show_final_markers and 'probs_final' in level:
        ax.plot(x, np.asarray(level['probs_final'], dtype=float), marker='D', lw=0.0, ms=5.5, color=NORD['nord14'], label='deployed final probability')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel('probability')
    ax.set_title(title or 'Class probabilities')
    style_axes_inward(ax, grid_y=True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_beta_density(alpha_like: Sequence[float], class_names: Sequence[str], *, title: str = 'Beta posterior'):
    setup_mpl_paper(usetex=False)
    alpha = np.asarray(alpha_like, dtype=np.float64).reshape(-1)
    if alpha.size != 2:
        raise ValueError('plot_beta_density requires exactly 2 classes')
    grid = np.linspace(1e-4, 1.0 - 1e-4, 800)
    from scipy.special import gammaln
    a, b = float(alpha[0]), float(alpha[1])
    log_z = gammaln(a + b) - gammaln(a) - gammaln(b)
    log_pdf = (a - 1.0) * np.log(grid) + (b - 1.0) * np.log(1.0 - grid) + log_z
    dens = np.exp(log_pdf)
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=180, sharex=True)
    axs[0].plot(grid, dens, color=NORD['nord10'], lw=2.0)
    axs[0].fill_between(grid, 0.0, dens, color=NORD['nord10'], alpha=0.28)
    axs[0].set_title(f'{title}: density')
    axs[0].set_ylabel('density')
    style_axes_inward(axs[0], grid_y=True)
    axs[1].plot(grid, log_pdf, color=NORD['nord11'], lw=2.0)
    axs[1].set_title(f'{title}: log density')
    axs[1].set_ylabel('log density')
    style_axes_inward(axs[1], grid_y=True)
    for ax in axs:
        ax.set_xlabel(f'p({class_names[0]})')
    fig.tight_layout()
    return fig


def plot_level_posterior_density(level: Dict[str, Any], *, title: Optional[str] = None, use_calibrated: bool = True):
    class_names = list(level['class_names']) if 'class_names' in level else list(level['node_names'])
    alpha = np.asarray(level['alpha_calibrated' if use_calibrated else 'alpha_base'], dtype=float)
    base_title = title or 'Posterior density'
    if len(class_names) == 2:
        return plot_beta_density(alpha, class_names, title=base_title)
    return plot_dirichlet_top3_simplex(alpha, class_names, title=base_title)


def plot_topk_probability_with_uncertainty(
    evo: pd.DataFrame,
    class_names: Sequence[str],
    *,
    object_id: Optional[str] = None,
    probability_prefix: str = 'calibrated_prob_',
    std_prefix: str = 'calibrated_std_',
    top_k: int = 3,
    title: Optional[str] = None,
    keep_mask_col: Optional[str] = 'leaf_kept',
):
    setup_mpl_paper(usetex=False)
    probs_last = []
    for cname in class_names:
        col = f'{probability_prefix}{cname}'
        probs_last.append(float(evo[col].iloc[-1]) if col in evo.columns else -np.inf)
    top_idx = np.argsort(np.asarray(probs_last))[-min(int(top_k), len(class_names)):][::-1]
    chosen = [class_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9.8, 5.3), dpi=180)
    t = evo['horizon_days'].to_numpy(dtype=float)
    colors = [NORD['nord10'], NORD['nord9'], NORD['nord14'], NORD['nord8'], NORD['nord11']]
    for color, cname in zip(colors, chosen):
        p = evo[f'{probability_prefix}{cname}'].to_numpy(dtype=float)
        s = evo[f'{std_prefix}{cname}'].to_numpy(dtype=float) if f'{std_prefix}{cname}' in evo.columns else np.zeros_like(p)
        lo = np.clip(p - s, 0.0, 1.0)
        hi = np.clip(p + s, 0.0, 1.0)
        ax.plot(t, p, marker='o', lw=2.0, color=color, label=cname)
        ax.fill_between(t, lo, hi, color=color, alpha=0.18)

    if keep_mask_col is not None and keep_mask_col in evo.columns:
        kept = evo[keep_mask_col].astype(bool).to_numpy()
        ax.scatter(t[~kept], np.full(np.sum(~kept), 0.02), marker='x', s=50, color='black', label='abstained')
    ax.set_xlabel('days since first detection')
    ax.set_ylabel('posterior probability')
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title or (f'Top-{len(chosen)} probability evolution with posterior uncertainty: {object_id}' if object_id else 'Probability evolution with posterior uncertainty'))
    style_axes_inward(ax, grid_y=True)
    ax.legend(ncol=min(4, len(chosen) + 1), fontsize=8)
    fig.tight_layout()
    return fig


def plot_probability_evolution(evo: pd.DataFrame, class_names: Sequence[str], *, object_id: Optional[str] = None, probability_prefix: str = 'final_prob_'):
    setup_mpl_paper(usetex=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    t = evo['horizon_days'].to_numpy(dtype=float)
    for cname in class_names:
        ax.plot(t, evo[f'{probability_prefix}{cname}'].to_numpy(dtype=float), marker='o', lw=1.8, label=cname)
    kept = evo['leaf_kept'].astype(bool).to_numpy()
    ax.scatter(t[~kept], np.full(np.sum(~kept), 0.02), marker='x', s=50, color='black', label='leaf abstained')
    ax.set_xlabel('days since first detection')
    ax.set_ylabel('deployed probability')
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f'Probability evolution: {object_id}' if object_id else 'Probability evolution')
    style_axes_inward(ax, grid_y=True)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_probability_std_evolution(evo: pd.DataFrame, class_names: Sequence[str], *, object_id: Optional[str] = None, std_prefix: str = 'calibrated_std_'):
    setup_mpl_paper(usetex=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=180)
    t = evo['horizon_days'].to_numpy(dtype=float)
    for cname in class_names:
        col = f'{std_prefix}{cname}'
        if col in evo.columns:
            ax.plot(t, evo[col].to_numpy(dtype=float), marker='o', lw=1.8, label=cname)
    ax.set_xlabel('days since first detection')
    ax.set_ylabel(r'$\sqrt{\mathrm{Var}[p_k]}$')
    ax.set_title(f'Per-class probability uncertainty evolution: {object_id}' if object_id else 'Per-class probability uncertainty evolution')
    style_axes_inward(ax, grid_y=True)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_uncertainty_evolution(evo: pd.DataFrame, runtime_state: RuntimePostprocessState, *, object_id: Optional[str] = None):
    setup_mpl_paper(usetex=False)
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 7.0), dpi=180, sharex=True)
    t = evo['horizon_days'].to_numpy(dtype=float)
    metrics = [c for c in ['score_vacuity', 'score_entropy', 'score_expected_entropy', 'score_mi', 'score_trace_uncertainty', 'score_fused_uncertainty'] if c in evo.columns]
    for col in metrics:
        axs[0].plot(t, evo[col].to_numpy(dtype=float), marker='o', lw=1.6, label=col.replace('score_', ''))
    axs[0].set_ylabel('uncertainty score')
    axs[0].set_title(f'Uncertainty evolution: {object_id}' if object_id else 'Uncertainty evolution')
    style_axes_inward(axs[0], grid_y=True)
    axs[0].legend(ncol=3, fontsize=8)

    sel = evo['leaf_selected_uncertainty'].to_numpy(dtype=float)
    thr = np.full_like(sel, np.nan)
    class_names = [c.replace('final_prob_', '') for c in evo.filter(like='final_prob_').columns]
    for i, row in evo.reset_index(drop=True).iterrows():
        if runtime_state.leaf_chosen.get('strategy') == 'classwise_pred':
            pred_id = class_names.index(row['leaf_pred_name']) if row['leaf_pred_name'] in class_names else 0
            thr[i] = float(runtime_state.leaf_chosen['thresholds_by_pred_class'][str(pred_id)])
        else:
            thr[i] = float(runtime_state.leaf_chosen['threshold'])
    kept = evo['leaf_kept'].astype(bool).to_numpy()
    axs[1].plot(t, sel, marker='o', lw=1.8, color=NORD['nord11'], label=runtime_state.leaf_chosen['name'])
    axs[1].plot(t, thr, lw=1.4, color='black', ls='--', label='active abstention threshold')
    axs[1].scatter(t[kept], sel[kept], color=NORD['nord14'], s=30, label='kept')
    axs[1].scatter(t[~kept], sel[~kept], color=NORD['nord11'], s=30, label='abstained')
    axs[1].set_xlabel('days since first detection')
    axs[1].set_ylabel('selected uncertainty')
    style_axes_inward(axs[1], grid_y=True)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_hierarchy_evolution(
    evo: pd.DataFrame,
    *,
    object_id: Optional[str] = None,
    include_levels: Sequence[str] = ('leaf', 'domain', 'family'),
):
    setup_mpl_paper(usetex=False)
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 7.0), dpi=180, sharex=True)
    t = evo['horizon_days'].to_numpy(dtype=float)
    colors = {'leaf': NORD['nord14'], 'domain': NORD['nord8'], 'family': NORD['nord9'], 'class': NORD['nord10']}
    for lv in include_levels:
        if lv == 'leaf':
            prob_col = 'leaf_pred_prob'
            keep_col = 'leaf_kept'
        else:
            prob_col = f'{lv}_top_prob'
            keep_col = f'{lv}_kept'
        if prob_col not in evo.columns or keep_col not in evo.columns:
            continue
        axs[0].plot(t, evo[prob_col].to_numpy(dtype=float), marker='o', lw=1.8, color=colors.get(lv, NORD['nord10']), label=f'{lv} top prob')
        axs[1].step(t, evo[keep_col].astype(int).to_numpy(dtype=float), where='mid', lw=1.8, color=colors.get(lv, NORD['nord10']), label=f'{lv} kept')
    axs[0].set_ylabel('top probability')
    axs[0].set_title(f'Hierarchy evolution: {object_id}' if object_id else 'Hierarchy evolution')
    style_axes_inward(axs[0], grid_y=True)
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel('days since first detection')
    axs[1].set_ylabel('kept = 1')
    axs[1].set_ylim(-0.05, 1.05)
    style_axes_inward(axs[1], grid_y=True)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def _threshold_series_for_level(evo: pd.DataFrame, chosen: Dict[str, Any], pred_name_col: str, class_names: Sequence[str]) -> np.ndarray:
    out = np.full(len(evo), np.nan, dtype=float)
    if chosen.get('strategy') == 'classwise_pred':
        cut_map = chosen.get('thresholds_by_pred_class') or {}
        for i, pred_name in enumerate(evo[pred_name_col].tolist()):
            if pred_name in class_names:
                idx = class_names.index(pred_name)
                if str(idx) in cut_map:
                    out[i] = float(cut_map[str(idx)])
    else:
        thr = chosen.get('threshold')
        if thr is not None:
            out[:] = float(thr)
    return out


def plot_hierarchy_gate_evolution(
    evo: pd.DataFrame,
    runtime_state: RuntimePostprocessState,
    *,
    object_id: Optional[str] = None,
    include_levels: Sequence[str] = ('leaf', 'domain', 'family'),
):
    setup_mpl_paper(usetex=False)
    level_specs: List[Tuple[str, str, Dict[str, Any], Sequence[str], np.ndarray]] = []
    leaf_class_names = [c.replace('final_prob_', '') for c in evo.filter(like='final_prob_').columns]
    if 'leaf' in include_levels:
        level_specs.append((
            'leaf',
            'leaf_selected_uncertainty',
            runtime_state.leaf_chosen,
            leaf_class_names,
            evo['leaf_kept'].astype(bool).to_numpy(),
        ))
    for level in runtime_state.hierarchy_levels:
        slug = str(level['level_name']).lower()
        if slug not in include_levels:
            continue
        val_col = f'{slug}_selected_uncertainty'
        keep_col = f'{slug}_kept'
        if val_col in evo.columns and keep_col in evo.columns:
            level_specs.append((
                slug,
                val_col,
                level['chosen'],
                list(level['node_names']),
                evo[keep_col].astype(bool).to_numpy(),
            ))

    fig, axs = plt.subplots(len(level_specs), 1, figsize=(10.0, 2.7 * len(level_specs)), dpi=180, sharex=True)
    if len(level_specs) == 1:
        axs = [axs]
    t = evo['horizon_days'].to_numpy(dtype=float)
    for ax, (slug, val_col, chosen, names, kept) in zip(axs, level_specs):
        pred_col = 'leaf_pred_name' if slug == 'leaf' else f'{slug}_pred_name'
        thr = _threshold_series_for_level(evo, chosen, pred_col, names)
        vals = evo[val_col].to_numpy(dtype=float)
        ax.plot(t, vals, marker='o', lw=1.8, color=NORD['nord11'], label=f'{slug} uncertainty')
        ax.plot(t, thr, lw=1.4, color='black', ls='--', label='threshold')
        ax.scatter(t[kept], vals[kept], color=NORD['nord14'], s=28, label='kept')
        ax.scatter(t[~kept], vals[~kept], color=NORD['nord11'], s=28, label='abstained')
        ax.set_ylabel(slug)
        style_axes_inward(ax, grid_y=True)
        ax.legend(fontsize=8, loc='best')
    axs[0].set_title(f'Hierarchy gate evolution: {object_id}' if object_id else 'Hierarchy gate evolution')
    axs[-1].set_xlabel('days since first detection')
    fig.tight_layout()
    return fig


def plot_hierarchy_decision_timeline(evo: pd.DataFrame, *, object_id: Optional[str] = None):
    setup_mpl_paper(usetex=False)
    rows: List[Tuple[str, List[str], List[bool]]] = []
    rows.append(('leaf', evo['leaf_pred_name'].astype(str).tolist(), evo['leaf_kept'].astype(bool).tolist()))
    for slug in ['domain', 'family', 'class']:
        pred_col = f'{slug}_pred_name'
        keep_col = f'{slug}_kept'
        if pred_col in evo.columns and keep_col in evo.columns:
            rows.append((slug, evo[pred_col].astype(str).tolist(), evo[keep_col].astype(bool).tolist()))
    final_labels = [x if x is not None else 'ABSTAIN' for x in evo['decision_label'].tolist()]
    final_kept = (~evo['abstain_completely'].astype(bool)).tolist()
    rows.append(('final_decision', final_labels, final_kept))

    all_labels: List[str] = []
    for _, labels, _ in rows:
        for label in labels:
            if label not in all_labels:
                all_labels.append(label)
    palette = plt.cm.get_cmap('tab20', max(20, len(all_labels)))
    label2rgba = {label: palette(i % palette.N) for i, label in enumerate(all_labels)}
    label2rgba['ABSTAIN'] = (1.0, 1.0, 1.0, 1.0)

    n_rows = len(rows)
    n_cols = len(evo)
    rgba = np.ones((n_rows, n_cols, 4), dtype=float)
    for i, (_, labels, keeps) in enumerate(rows):
        for j, (label, keep) in enumerate(zip(labels, keeps)):
            color = list(to_rgba(label2rgba.get(label, (0.8, 0.8, 0.8, 1.0))))
            if label == 'ABSTAIN':
                color = [1.0, 1.0, 1.0, 1.0]
            color[3] = 1.0 if keep else 0.22
            rgba[i, j, :] = color

    fig_w = min(0.34 * max(8, n_cols) + 3.5, 24.0)
    fig, ax = plt.subplots(figsize=(fig_w, 1.15 * n_rows + 2.2), dpi=180)
    ax.imshow(rgba, aspect='auto', interpolation='nearest')

    for i, (row_name, labels, keeps) in enumerate(rows):
        for j, (label, keep) in enumerate(zip(labels, keeps)):
            txt = label.replace('Transient', 'Tr').replace('Variable', 'Var')
            txt = txt.replace('final_decision', 'final')
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.3, color='black')
            if not keep:
                ax.plot([j - 0.35, j + 0.35], [i - 0.35, i + 0.35], color='black', lw=0.8)
                ax.plot([j - 0.35, j + 0.35], [i + 0.35, i - 0.35], color='black', lw=0.8)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([str(int(x)) for x in evo['prefix_index'].tolist()], rotation=90)
    ax.set_xlabel('detection index')
    ax.set_title(f'Hierarchy decision timeline: {object_id}' if object_id else 'Hierarchy decision timeline')
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for x in np.arange(-0.5, n_cols, 1.0):
        ax.axvline(x, color='white', lw=0.4, alpha=0.8)
    for y in np.arange(-0.5, n_rows, 1.0):
        ax.axhline(y, color='white', lw=0.4, alpha=0.8)

    top = ax.secondary_xaxis('top')
    top.set_xticks(np.arange(n_cols))
    top.set_xticklabels([f'{x:.1f}' for x in evo['horizon_days'].tolist()], rotation=90)
    top.set_xlabel('days since first detection')

    fig.tight_layout()
    return fig
