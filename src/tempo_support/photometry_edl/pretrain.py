from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import Config
from .data import FeatureStats, PhotoEventDataset, make_collate, read_manifest_csv
from .models import EventTransformerEncoder, MaskedEventPretrainer
from .utils import set_seed


@dataclass
class PretrainConfig:
    mask_p: float = 0.30
    epochs: int = 50
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 1e-2
    # loss weights
    lambda_flux: float = 5.0
    lambda_band: float = 3.0
    lambda_dt: float = 5.0

    def to_dict(self) -> Dict:
        return asdict(self)


def mask_batch_balanced(x: torch.Tensor, pad_mask: torch.Tensor, mask_p: float) -> torch.Tensor:
    """Balanced masking across g/r/i. Returns masked_positions (B, L) bool.

    This follows your notebook logic: mask ~mask_p of valid tokens, with ~1/3 from each band.
    Masked tokens have channels [2:7] zeroed: logf, logf_err, and band one-hot.
    """
    masked = torch.zeros_like(pad_mask)
    B, L, _ = x.shape
    for b in range(B):
        valid = (~pad_mask[b]).nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            continue
        k = max(int(len(valid) * mask_p), 3)
        num_each = k // 3
        extras = k - 3 * num_each

        bands = x[b, :, 4:7].argmax(-1)  # 0/1/2
        idxs = []
        for band in [0, 1, 2]:
            valid_b = valid[bands[valid] == band]
            if len(valid_b) > 0:
                take = min(len(valid_b), num_each)
                perm = torch.randperm(len(valid_b), device=x.device)[:take]
                idxs.append(valid_b[perm])

        if extras > 0 and len(idxs) > 0:
            remaining = torch.cat(idxs)
            pool = valid[~torch.isin(valid, remaining)]
            if len(pool) > 0:
                perm = torch.randperm(len(pool), device=x.device)[:extras]
                idxs.append(pool[perm])

        if len(idxs) == 0:
            continue
        idx = torch.cat(idxs)
        x[b, idx, 2:7] = 0.0
        masked[b, idx] = True
    return masked


def pretrain_mpt(cfg: Config, pt_cfg: PretrainConfig, *, out_file: Path) -> Dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg.data_dir)
    manifest_dir = Path(cfg.manifest_dir) if cfg.manifest_dir is not None else data_dir
    train_df = read_manifest_csv(manifest_dir / "manifest_train.csv", data_dir=data_dir, path_prefix=cfg.path_prefix)
    ds = PhotoEventDataset(
        train_df,
        horizon_days=cfg.horizon_days,  # pretrain on same horizon by default
        drop_i_band=cfg.drop_i_band,
    )
    stats = FeatureStats.load(data_dir / cfg.stats_file)
    collate = make_collate(stats)

    loader = DataLoader(
        ds,
        batch_size=pt_cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    enc = EventTransformerEncoder(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    mpt = MaskedEventPretrainer(enc).to(device)

    opt = AdamW(mpt.parameters(), lr=pt_cfg.lr, weight_decay=pt_cfg.weight_decay)

    flux_losses, band_losses, dt_losses, total_losses = [], [], [], []

    for ep in range(1, pt_cfg.epochs + 1):
        mpt.train()
        ep_f = ep_b = ep_d = ep_t = 0.0
        n_batches = 0

        for xb, _, mb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            x_orig = xb.clone()

            masked_tok = mask_batch_balanced(xb, mb, pt_cfg.mask_p)
            mf = masked_tok.view(-1)

            f_hat, b_hat, dt_hat = mpt(xb, mb)
            # flux MSE on masked
            true_f = x_orig[..., 2].view(-1)
            loss_f = F.mse_loss(f_hat.view(-1)[mf], true_f[mf])

            # band CE on masked
            true_b = x_orig[..., 4:7].argmax(-1).view(-1)
            loss_b = F.cross_entropy(b_hat.view(-1, 3)[mf], true_b[mf])

            # dt_prev MSE on masked (predict next dt_prev)
            dt_gt = torch.roll(x_orig[..., 1], shifts=-1, dims=1)
            dt_gt[:, -1] = 0.0
            dt_gt = dt_gt.reshape(-1)
            loss_dt = F.mse_loss(dt_hat[..., 0].reshape(-1)[mf], dt_gt[mf])

            loss = pt_cfg.lambda_flux * loss_f + pt_cfg.lambda_band * loss_b + pt_cfg.lambda_dt * loss_dt

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_f += float(loss_f.item())
            ep_b += float(loss_b.item())
            ep_d += float(loss_dt.item())
            ep_t += float(loss.item())
            n_batches += 1

        flux_losses.append(ep_f / max(1, n_batches))
        band_losses.append(ep_b / max(1, n_batches))
        dt_losses.append(ep_d / max(1, n_batches))
        total_losses.append(ep_t / max(1, n_batches))

    # Save encoder weights only (so fine-tuning can load strict=False)
    torch.save(enc.state_dict(), out_file)

    log = {
        "device": str(device),
        "pt_cfg": pt_cfg.to_dict(),
        "losses": {
            "flux": flux_losses,
            "band": band_losses,
            "dt": dt_losses,
            "total": total_losses,
        },
    }
    (out_file.parent / "pretrain_log.json").write_text(json.dumps(log, indent=2))
    return log
