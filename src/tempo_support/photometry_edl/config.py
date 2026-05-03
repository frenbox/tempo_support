from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Config:
    # Data
    data_dir: Path
    stats_file: str = "feature_stats_day100.npz"
    manifest_dir: Optional[Path] = None
    path_prefix: Optional[str] = None
    horizon_days: float = 100.0
    drop_i_band: bool = False
    min_obs_total: int = 0
    min_obs_g: int = 0
    min_obs_r: int = 0
    min_obs_i: int = 0
    min_bands_observed: int = 0
    batch_size: int = 256
    num_workers: int = 8

    # Model
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.4
    max_len: int = 257
    pool: str = "attn"  # cls | attn | mean
    band_mode: str = "embed"  # embed | onehot
    band_embed_dim: int = 8
    time_encoding: str = "dt_both"  # dt_first | dt_both
    use_global_features: bool = True
    global_feature_set: str = "basic"  # basic | enhanced | physics
    global_hidden_dim: int = 64

    # Optim / schedule
    lr: float = 2e-4
    weight_decay: float = 1e-2
    epochs: int = 160
    patience: int = 10
    grad_clip_norm: float = 1.0

    # Augmentation
    p_token_dropout: float = 0.10
    jitter_scale: float = 0.10
    flux_nu: int = 8
    flux_jitter_frac: float = 0.15

    # Early-time training strategy
    train_random_horizon: bool = False
    train_random_horizon_min_days: float = 2.0
    train_random_horizon_power: float = 1.8
    train_curriculum_epochs: int = 0
    train_curriculum_start_days: float = 5.0

    # Balancing / sampling
    use_weighted_sampler: bool = True
    oversample: bool = True
    oversample_classes: Optional[List[int]] = None
    oversample_target: str = "median"  # median | max | fixed
    oversample_target_count: int = 0

    # EDL loss
    edl_anneal_epochs: int = 10
    edl_kl_strength: float = 1e-3
    edl_sup_strength: float = 2e-2
    label_smoothing: float = 0.02
    class_balance_beta: float = 0.999
    use_class_balanced_loss: bool = False
    focal_gamma: float = 0.0
    margin_loss_strength: float = 0.0
    margin_delta: float = 0.0
    margin_pairs: Optional[List[List[int]]] = None

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Repro
    seed: int = 42
    taxonomy_preset: str = "default"

    # Best-checkpoint selection
    ckpt_select_metric: str = "macro_auprc"  # macro_auprc | macro_f1 | balanced_accuracy | composite_sota | composite_prod
    ckpt_select_focus_class: str = "TDE"

    # Output
    out_dir: Path = Path("runs/default")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["out_dir"] = str(self.out_dir)
        d["manifest_dir"] = str(self.manifest_dir) if self.manifest_dir is not None else None
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        d = dict(d)
        d["data_dir"] = Path(d["data_dir"])
        d["out_dir"] = Path(d.get("out_dir", "runs/default"))
        if d.get("manifest_dir") is not None:
            d["manifest_dir"] = Path(d["manifest_dir"])
        return Config(**d)
