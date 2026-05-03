from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import t as student_t
from torch.utils.data import Dataset

from .data import PhotoEventDataset, build_event_tensor


@dataclass(frozen=True)
class AugmentConfig:
    p_token_dropout: float = 0.10
    jitter_scale: float = 0.10
    flux_nu: int = 8
    flux_jitter_frac: float = 0.15


def _drop_tokens(arr: np.ndarray, p_drop: float) -> np.ndarray:
    if len(arr) <= 1 or p_drop <= 0:
        return arr
    keep = np.random.rand(len(arr)) > p_drop
    keep[0] = True  # keep first observation
    out = arr[keep]
    return out if len(out) else arr[:1]


def _jitter_times(arr: np.ndarray, jitter_scale: float) -> np.ndarray:
    if len(arr) <= 1 or jitter_scale <= 0:
        return arr
    t0 = arr[:, 0].copy()
    ints = np.diff(np.concatenate([[0.0], t0]))
    noise = np.random.randn(len(ints)) * (jitter_scale * np.maximum(ints, 1e-6))
    ints = np.clip(ints + noise, 0.0, None)
    tnew = np.cumsum(ints)
    arr = arr.copy()
    arr[:, 0] = tnew
    arr[:, 1] = np.concatenate([[0.0], ints[:-1]])
    return arr


def _jitter_flux(arr: np.ndarray, flux_nu: int, flux_jitter_frac: float) -> np.ndarray:
    if flux_jitter_frac <= 0:
        return arr
    arr = arr.copy()
    logf = arr[:, 3]
    logfe = arr[:, 4]
    f = np.exp(logf)
    ferr = np.exp(logfe)
    scale = flux_jitter_frac * ferr
    fnew = student_t(df=flux_nu, loc=f, scale=scale).rvs()
    arr[:, 3] = np.log(np.clip(fnew, 1e-10, None))
    return arr


def augment_raw_sequence(arr: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    arr = _drop_tokens(arr, cfg.p_token_dropout)
    arr = _jitter_times(arr, cfg.jitter_scale)
    arr = _jitter_flux(arr, cfg.flux_nu, cfg.flux_jitter_frac)
    return arr


class OversampleAugmentDataset(Dataset):
    """Oversample minority classes by cloning samples and applying augmentation.

    This implements the exact strategy used in your notebook (originally only for TDE),
    but generalized to any set of classes.

    The oversampling is done *once* at construction time to build an index list.
    """

    def __init__(
        self,
        base: PhotoEventDataset,
        labels: np.ndarray,
        *,
        target_per_class: int,
        augment_cfg: AugmentConfig,
        oversample_classes: Optional[Sequence[int]] = None,
        seed: int = 0,
    ):
        self.base = base
        self.labels = np.asarray(labels, dtype=int)
        self.target_per_class = int(target_per_class)
        self.augment_cfg = augment_cfg
        self.rng = np.random.default_rng(seed)

        C = int(self.labels.max()) + 1
        oversample_set = set(range(C)) if oversample_classes is None else set(map(int, oversample_classes))

        idx_by_class: Dict[int, List[int]] = {c: np.where(self.labels == c)[0].tolist() for c in range(C)}

        indices: List[int] = list(range(len(self.base)))  # all originals
        # add extras for classes below target_per_class
        for c, idxs in idx_by_class.items():
            if c not in oversample_set:
                continue
            need = max(0, self.target_per_class - len(idxs))
            if need == 0 or len(idxs) == 0:
                continue
            extra = self.rng.choice(idxs, size=need, replace=True).tolist()
            indices.extend(extra)

        self.indices = indices
        self._is_extra = np.zeros(len(self.indices), dtype=bool)
        self._is_extra[len(self.base):] = True

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(int(epoch))

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x, y = self.base[idx]
        if self._is_extra[i]:
            # rebuild from raw and augment (ensures augmentation is applied in raw space)
            row = self.base.df.iloc[idx]
            raw = np.load(row.filepath, allow_pickle=False)
            arr = raw["data"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
            arr = np.asarray(arr, dtype=np.float32)
            if hasattr(self.base, "_preprocess_array"):
                # Keep at least one fallback token when horizon truncation removes
                # all events, so downstream global-feature extraction is well-defined.
                arr = self.base._preprocess_array(arr, idx=idx, allow_empty=False)
            else:
                if self.base.horizon_days is not None:
                    arr = arr[arr[:, 0] <= float(self.base.horizon_days)]
                if len(arr) == 0:
                    arr = np.asarray([[0, 0, 0, 0, 0]], dtype=np.float32)
            arr = augment_raw_sequence(arr, self.augment_cfg)
            x = build_event_tensor(arr.astype(np.float32), band_mode=self.base.band_mode)
        return x, y
