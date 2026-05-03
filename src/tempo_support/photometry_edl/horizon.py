from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import FeatureStats, build_event_tensor
from .uncertainty import alpha_from_evidence, dirichlet_mean, vacuity


@torch.no_grad()
def horizon_sweep_single(
    model,
    *,
    raw_file: Path,
    stats: FeatureStats,
    horizons: Sequence[float],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Evaluate one object truncated at multiple horizons (days)."""
    raw = np.load(raw_file, allow_pickle=False)
    arr = raw["data"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
    arr = np.asarray(arr, dtype=np.float32)

    mean = stats.mean.numpy().reshape(1, 1, 4)
    std = stats.std.numpy().reshape(1, 1, 4)

    ps = []
    us = []
    for H in horizons:
        arr_h = arr[arr[:, 0] <= float(H)]
        if len(arr_h) == 0:
            arr_h = arr[:1]
        x = build_event_tensor(arr_h).unsqueeze(0).float()  # (1,L,7)
        cont = (x[..., :4].numpy() - mean) / (std + 1e-8)
        x_norm = torch.from_numpy(np.concatenate([cont, x[..., 4:].numpy()], axis=-1)).to(device)
        pad_mask = torch.zeros((1, x_norm.size(1)), dtype=torch.bool, device=device)

        evidence = model(x_norm, pad_mask)
        alpha = alpha_from_evidence(evidence)
        p = dirichlet_mean(alpha)
        u = vacuity(alpha)
        ps.append(p.squeeze(0).cpu().numpy())
        us.append(float(u.squeeze(0).cpu().numpy()))

    return {"horizons": np.asarray(horizons, dtype=float), "probs": np.stack(ps), "vacuity": np.asarray(us)}
