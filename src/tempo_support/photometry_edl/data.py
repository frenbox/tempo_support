from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .taxonomy import DEFAULT_TAXONOMY, Taxonomy


_BAND_OH = np.eye(3, dtype=np.float32)
BAND_NAME = {0: "g", 1: "r", 2: "i"}


def _ensure_valid_band(band: np.ndarray) -> None:
    if (band < 0).any() or (band > 2).any():
        raise ValueError("band_id must be in {0,1,2}")


def build_event_tensor(arr: np.ndarray, *, band_mode: str = "onehot") -> torch.Tensor:
    """Convert raw photometry array to event-token representation.

    Expected raw columns:
      0: dt_first, 1: dt_prev, 2: band_id, 3: log_flux, 4: log_flux_err

    Output channels for `band_mode`:
      - onehot: [log1p(dt_first), log1p(dt_prev), log_flux, log_flux_err, onehot(band,3)]
      - embed : [log1p(dt_first), log1p(dt_prev), log_flux, log_flux_err, band_id]
    """
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"Expected arr shape (N,>=5), got {arr.shape}")
    dt = np.log1p(arr[:, 0]).astype(np.float32)
    dt_prev = np.log1p(arr[:, 1]).astype(np.float32)
    logf = arr[:, 3].astype(np.float32)
    logfe = arr[:, 4].astype(np.float32)
    band = arr[:, 2].astype(np.int64)
    _ensure_valid_band(band)

    vec4 = np.stack([dt, dt_prev, logf, logfe], axis=1)
    if band_mode == "onehot":
        oh = _BAND_OH[band]
        out = np.concatenate([vec4, oh], axis=1)
    elif band_mode == "embed":
        out = np.concatenate([vec4, band.astype(np.float32)[:, None]], axis=1)
    else:
        raise ValueError(f"Unknown band_mode={band_mode}")
    return torch.from_numpy(out)


def preprocess_photometry_array(
    arr: np.ndarray,
    *,
    horizon_days: Optional[float] = None,
    drop_i_band: bool = False,
    allow_empty: bool = False,
) -> np.ndarray:
    """Apply canonical per-object preprocessing prior to tensorization."""
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] < 5:
        raise ValueError(f"Expected arr shape (N,>=5), got {out.shape}")
    if horizon_days is not None:
        out = out[out[:, 0] <= float(horizon_days)]
    if drop_i_band:
        out = out[out[:, 2].astype(np.int64) != 2]
    if len(out) == 0 and not allow_empty:
        out = np.asarray([[0, 0, 0, 0, 0]], dtype=np.float32)
    return out


def _quality_pass(
    arr: np.ndarray,
    *,
    min_obs_total: int = 0,
    min_obs_g: int = 0,
    min_obs_r: int = 0,
    min_obs_i: int = 0,
    min_bands_observed: int = 0,
    drop_i_band: bool = False,
) -> bool:
    """Check whether preprocessed sequence satisfies quality constraints."""
    band = arr[:, 2].astype(np.int64)
    counts = np.array([(band == b).sum() for b in [0, 1, 2]], dtype=int)
    n_obs = int(arr.shape[0])
    n_bands = int((counts > 0).sum())

    if n_obs < int(min_obs_total):
        return False
    if counts[0] < int(min_obs_g):
        return False
    if counts[1] < int(min_obs_r):
        return False
    if not drop_i_band and counts[2] < int(min_obs_i):
        return False
    if n_bands < int(min_bands_observed):
        return False
    return True


def filter_manifest_quality(
    manifest: pd.DataFrame,
    *,
    horizon_days: Optional[float],
    drop_i_band: bool,
    min_obs_total: int = 0,
    min_obs_g: int = 0,
    min_obs_r: int = 0,
    min_obs_i: int = 0,
    min_bands_observed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter manifest rows by per-object quality criteria.

    Returns filtered DataFrame and a summary dictionary.
    """
    if min_obs_total < 0 or min_obs_g < 0 or min_obs_r < 0 or min_obs_i < 0 or min_bands_observed < 0:
        raise ValueError("Quality cut thresholds must be >= 0")
    if min_bands_observed > (2 if drop_i_band else 3):
        raise ValueError("min_bands_observed is infeasible given selected band set")

    keep_mask = np.zeros(len(manifest), dtype=bool)
    fail_counts = {
        "min_obs_total": 0,
        "min_obs_g": 0,
        "min_obs_r": 0,
        "min_obs_i": 0,
        "min_bands_observed": 0,
        "other": 0,
    }
    min_i_eff = 0 if drop_i_band else int(min_obs_i)

    for i, row in manifest.reset_index(drop=True).iterrows():
        try:
            raw = np.load(row.filepath, allow_pickle=False)
            arr = raw["data"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
            arr = preprocess_photometry_array(arr, horizon_days=horizon_days, drop_i_band=drop_i_band, allow_empty=True)
            band = arr[:, 2].astype(np.int64)
            counts = np.array([(band == b).sum() for b in [0, 1, 2]], dtype=int)
            n_obs = int(arr.shape[0])
            n_bands = int((counts > 0).sum())
            fail = False
            if n_obs < int(min_obs_total):
                fail_counts["min_obs_total"] += 1
                fail = True
            if counts[0] < int(min_obs_g):
                fail_counts["min_obs_g"] += 1
                fail = True
            if counts[1] < int(min_obs_r):
                fail_counts["min_obs_r"] += 1
                fail = True
            if counts[2] < min_i_eff:
                fail_counts["min_obs_i"] += 1
                fail = True
            if n_bands < int(min_bands_observed):
                fail_counts["min_bands_observed"] += 1
                fail = True
            keep_mask[i] = not fail
        except Exception:
            fail_counts["other"] += 1
            keep_mask[i] = False

    out = manifest.loc[keep_mask].reset_index(drop=True).copy()
    summary = {
        "rows_before": int(len(manifest)),
        "rows_after": int(len(out)),
        "rows_dropped": int(len(manifest) - len(out)),
        "drop_i_band": int(bool(drop_i_band)),
        "min_obs_total": int(min_obs_total),
        "min_obs_g": int(min_obs_g),
        "min_obs_r": int(min_obs_r),
        "min_obs_i_effective": int(min_i_eff),
        "min_bands_observed": int(min_bands_observed),
    }
    summary.update({f"dropped_{k}": int(v) for k, v in fail_counts.items()})
    return out, summary


class PhotoEventDataset(Dataset):
    """Loads tokenized photometry sequences using a CSV manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        taxonomy: Taxonomy = DEFAULT_TAXONOMY,
        horizon_days: Optional[float] = None,
        band_mode: str = "onehot",
        drop_i_band: bool = False,
    ):
        self.df = manifest.reset_index(drop=True).copy()
        self.taxonomy = taxonomy
        self.horizon_days = horizon_days
        self.band_mode = band_mode
        self.drop_i_band = drop_i_band
        self._id2broad = taxonomy.id2broad_id

        if "filepath" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("Manifest must have columns: filepath, label")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_horizon_days(self, idx: Optional[int] = None) -> Optional[float]:
        _ = idx
        return self.horizon_days

    def _load_raw_array(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        raw = np.load(row.filepath, allow_pickle=False)
        arr = raw["data"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
        return np.asarray(arr, dtype=np.float32)

    def _preprocess_array(
        self,
        arr: np.ndarray,
        *,
        idx: Optional[int] = None,
        allow_empty: bool = False,
    ) -> np.ndarray:
        return preprocess_photometry_array(
            arr,
            horizon_days=self._resolve_horizon_days(idx),
            drop_i_band=self.drop_i_band,
            allow_empty=allow_empty,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        arr = self._load_raw_array(idx)
        arr = self._preprocess_array(arr, idx=idx)
        x = build_event_tensor(arr, band_mode=self.band_mode)
        y_sub = int(row.label)
        y_broad = int(self._id2broad[y_sub])
        return x, y_broad


class EarlyHorizonCurriculumDataset(PhotoEventDataset):
    """Photo dataset with train-time horizon randomization and optional curriculum."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        taxonomy: Taxonomy = DEFAULT_TAXONOMY,
        horizon_days: Optional[float] = None,
        band_mode: str = "onehot",
        drop_i_band: bool = False,
        random_horizon: bool = True,
        random_horizon_min_days: float = 2.0,
        random_horizon_power: float = 1.8,
        curriculum_epochs: int = 0,
        curriculum_start_days: float = 5.0,
        seed: int = 0,
    ):
        super().__init__(
            manifest,
            taxonomy=taxonomy,
            horizon_days=horizon_days,
            band_mode=band_mode,
            drop_i_band=drop_i_band,
        )
        self.random_horizon = bool(random_horizon)
        self.random_horizon_min_days = float(random_horizon_min_days)
        self.random_horizon_power = float(max(0.05, random_horizon_power))
        self.curriculum_epochs = int(max(0, curriculum_epochs))
        self.curriculum_start_days = float(curriculum_start_days)
        self.seed = int(seed)
        self._epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(max(1, epoch))

    @staticmethod
    def _splitmix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return x ^ (x >> 31)

    def _uniform01(self, idx: int) -> float:
        x = (
            int(self.seed)
            + 0xD1B54A32D192ED03 * int(idx + 1)
            + 0x94D049BB133111EB * int(self._epoch + 1)
        ) & 0xFFFFFFFFFFFFFFFF
        z = self._splitmix64(x)
        return ((z >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def _current_curriculum_cap(self) -> Optional[float]:
        if self.horizon_days is None:
            return None
        full_h = float(self.horizon_days)
        if self.curriculum_epochs <= 0:
            return full_h
        start_h = min(full_h, max(0.05, float(self.curriculum_start_days)))
        if self.curriculum_epochs == 1:
            return full_h
        prog = min(1.0, max(0.0, float(self._epoch - 1) / float(self.curriculum_epochs - 1)))
        return start_h + prog * (full_h - start_h)

    def _resolve_horizon_days(self, idx: Optional[int] = None) -> Optional[float]:
        cap = self._current_curriculum_cap()
        if cap is None:
            return None
        if not self.random_horizon:
            return float(cap)
        lo = min(float(cap), max(0.05, float(self.random_horizon_min_days)))
        if idx is None or lo >= float(cap) - 1e-8:
            return float(cap)
        u = self._uniform01(int(idx))
        frac = u ** self.random_horizon_power
        return float(lo + (float(cap) - lo) * frac)


@dataclass(frozen=True)
class FeatureStats:
    mean: torch.Tensor
    std: torch.Tensor

    @staticmethod
    def load(path: Path) -> "FeatureStats":
        st = np.load(path)
        mean = torch.from_numpy(st["mean"]).float().view(-1)
        std = torch.from_numpy(st["std"]).float().view(-1)
        if mean.numel() != 4 or std.numel() != 4:
            raise ValueError("Stats file must store mean/std for 4 continuous channels")
        return FeatureStats(mean=mean, std=std)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy())


def compute_feature_stats(dataset: Dataset, *, max_items: Optional[int] = None) -> FeatureStats:
    n = 0
    s1 = torch.zeros(4)
    s2 = torch.zeros(4)
    for i, (x, _) in enumerate(dataset):
        if max_items is not None and i >= max_items:
            break
        cont = x[:, :4].float()
        n_i = cont.size(0)
        n += n_i
        s1 += cont.sum(dim=0)
        s2 += (cont ** 2).sum(dim=0)
    if n == 0:
        raise ValueError("Dataset is empty")
    mean = s1 / n
    var = (s2 / n) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return FeatureStats(mean=mean, std=std)


def global_feature_dim(feature_set: str) -> int:
    if feature_set == "basic":
        return 8
    if feature_set == "enhanced":
        return 16
    if feature_set == "physics":
        return 24
    raise ValueError(f"Unknown global feature set: {feature_set}")


def _safe_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    n = int(x.numel())
    if n < 2:
        return 0.0
    xm = torch.mean(x)
    ym = torch.mean(y)
    denom = torch.sum((x - xm) ** 2)
    if float(denom.item()) <= 1e-12:
        return 0.0
    slope = torch.sum((x - xm) * (y - ym)) / denom
    return float(slope.item())


def _global_features_from_sequence(x: torch.Tensor, *, band_mode: str, feature_set: str = "basic") -> torch.Tensor:
    """Compute global features per light curve from unnormalized token sequence."""
    if x.size(0) == 0:
        # Defensive fallback for any upstream truncation edge case.
        return torch.zeros(global_feature_dim(feature_set), dtype=torch.float32)
    cont = x[:, :4].float()
    dt_first = torch.expm1(cont[:, 0]).clamp_min(0.0)
    dt_prev = torch.expm1(cont[:, 1]).clamp_min(0.0)
    logf = cont[:, 2]

    n_obs = float(x.size(0))
    duration = float(dt_first.max().item() if x.size(0) > 0 else 0.0)
    amp = float((logf.max() - logf.min()).item() if x.size(0) > 0 else 0.0)

    if band_mode == "onehot":
        onehot = x[:, 4:7].float()
        counts = onehot.sum(dim=0)
        band_id = onehot.argmax(dim=1)
    else:
        band_id = x[:, 4].long().clamp(0, 2)
        counts = torch.stack([(band_id == k).sum() for k in range(3)], dim=0).float()

    # Color proxies: average log-flux differences by band.
    means = []
    for k in range(3):
        m = band_id == k
        means.append(logf[m].mean() if m.any() else torch.tensor(0.0, dtype=logf.dtype, device=logf.device))
    color_gr = means[0] - means[1]
    color_ri = means[1] - means[2]

    basic = [
        duration,
        n_obs,
        counts[0].item(),
        counts[1].item(),
        counts[2].item(),
        amp,
        float(color_gr.item()),
        float(color_ri.item()),
    ]
    if feature_set == "basic":
        return torch.tensor(basic, dtype=torch.float32)

    if feature_set not in {"enhanced", "physics"}:
        raise ValueError(f"Unknown global feature set: {feature_set}")

    idx_peak = int(torch.argmax(logf).item())
    peak_t = float(dt_first[idx_peak].item())
    peak_frac_h = peak_t / max(1e-6, duration)
    peak_flux = float(logf[idx_peak].item())
    med_dt_prev = float(torch.median(dt_prev).item())
    std_flux = float(torch.std(logf, unbiased=False).item()) if x.size(0) > 1 else 0.0
    p90 = float(torch.quantile(logf, 0.90).item()) if x.size(0) > 1 else peak_flux
    p10 = float(torch.quantile(logf, 0.10).item()) if x.size(0) > 1 else peak_flux

    t = dt_first
    rise_mask = t <= t[idx_peak]
    fall_mask = t >= t[idx_peak]
    rise_slope = _safe_slope(t[rise_mask], logf[rise_mask])
    fall_slope = _safe_slope(t[fall_mask], logf[fall_mask])
    rise_fall_ratio = rise_slope / max(1e-6, abs(fall_slope))

    enhanced = basic + [
        peak_frac_h,
        peak_flux,
        med_dt_prev,
        std_flux,
        p90 - p10,
        rise_slope,
        fall_slope,
        rise_fall_ratio,
    ]
    if feature_set == "enhanced":
        return torch.tensor(enhanced, dtype=torch.float32)

    # Physically motivated extras for hierarchical experiments:
    # band-occupancy fractions and per-band rise/decline slope proxies.
    n_safe = max(1.0, n_obs)
    frac_g = float(counts[0].item() / n_safe)
    frac_r = float(counts[1].item() / n_safe)
    frac_i = float(counts[2].item() / n_safe)
    slope_g = _safe_slope(t[band_id == 0], logf[band_id == 0])
    slope_r = _safe_slope(t[band_id == 1], logf[band_id == 1])
    slope_i = _safe_slope(t[band_id == 2], logf[band_id == 2])
    color_gr_slope = slope_g - slope_r
    color_ri_slope = slope_r - slope_i

    physics = enhanced + [
        frac_g,
        frac_r,
        frac_i,
        slope_g,
        slope_r,
        slope_i,
        color_gr_slope,
        color_ri_slope,
    ]
    return torch.tensor(physics, dtype=torch.float32)


def make_collate(
    stats: FeatureStats,
    *,
    band_mode: str = "onehot",
    return_global: bool = False,
    global_feature_set: str = "basic",
) -> Callable:
    """Pad sequences and return (x_norm, y, pad_mask[, global_features])."""
    mean = stats.mean.view(1, 1, 4)
    std = stats.std.view(1, 1, 4)

    def collate(batch: Sequence[Tuple[torch.Tensor, int]]):
        seqs, labels = zip(*batch)
        lens = [s.size(0) for s in seqs]
        pad = pad_sequence(seqs, batch_first=True)
        B, L, D = pad.shape

        pad_mask = torch.stack([
            torch.cat([torch.zeros(l), torch.ones(L - l)]) for l in lens
        ]).bool()

        cont = (pad[..., :4].float() - mean) / (std + 1e-8)
        if band_mode == "onehot":
            x = torch.cat([cont, pad[..., 4:].float()], dim=-1)
        elif band_mode == "embed":
            band = pad[..., 4:5].float()
            x = torch.cat([cont, band], dim=-1)
        else:
            raise ValueError(f"Unknown band_mode={band_mode}")

        y = torch.tensor(labels, dtype=torch.long)
        if not return_global:
            return x, y, pad_mask

        g = torch.stack([_global_features_from_sequence(s, band_mode=band_mode, feature_set=global_feature_set) for s in seqs], dim=0)
        gd = global_feature_dim(global_feature_set)
        if g.shape != (B, gd):
            raise RuntimeError(f"Expected global feature shape {(B, gd)}, got {tuple(g.shape)}")
        if not torch.isfinite(g).all():
            raise RuntimeError("Global features must be finite")
        return x, y, pad_mask, g

    return collate


def _resolve_path(raw_path: str, *, manifest_dir: Path, data_dir: Optional[Path], path_prefix: Optional[str]) -> str:
    p = Path(str(raw_path))

    if path_prefix is not None and str(raw_path).startswith(path_prefix):
        suffix = str(raw_path)[len(path_prefix):].lstrip("/\\")
        if data_dir is None:
            return str((manifest_dir / suffix).resolve())
        return str((data_dir / suffix).resolve())

    if p.is_absolute():
        return str(p)

    base = data_dir if data_dir is not None else manifest_dir
    return str((base / p).resolve())


def read_manifest_csv(path: Path, *, data_dir: Optional[Path] = None, path_prefix: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "filepath" in df.columns and df.filepath.dtype == object:
        df["filepath"] = df["filepath"].apply(
            lambda rp: _resolve_path(str(rp), manifest_dir=path.parent, data_dir=data_dir, path_prefix=path_prefix)
        )
    return df


def rewrite_manifest_paths(
    manifest_path: Path,
    *,
    output_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    path_prefix: Optional[str] = None,
    make_relative_to_data_dir: bool = True,
) -> Path:
    """Rewrite manifest filepaths for portability.

    If `make_relative_to_data_dir=True`, stores relative paths when possible.
    """
    df = pd.read_csv(manifest_path)
    if "filepath" not in df.columns:
        raise ValueError(f"Manifest missing filepath column: {manifest_path}")

    resolved = read_manifest_csv(manifest_path, data_dir=data_dir, path_prefix=path_prefix)

    if make_relative_to_data_dir and data_dir is not None:
        droot = Path(data_dir).resolve()

        def _to_rel(p: str) -> str:
            pp = Path(p).resolve()
            try:
                return str(pp.relative_to(droot))
            except Exception:
                return str(pp)

        resolved["filepath"] = resolved["filepath"].apply(_to_rel)

    out = output_path or manifest_path
    out.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(out, index=False)
    return out
