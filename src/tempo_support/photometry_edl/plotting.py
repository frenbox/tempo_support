from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap


NORD: Dict[str, str] = {
    "nord0":  "#2E3440",
    "nord1":  "#3B4252",
    "nord2":  "#434C5E",
    "nord3":  "#4C566A",
    "nord4":  "#D8DEE9",
    "nord5":  "#E5E9F0",
    "nord6":  "#ECEFF4",
    "nord7":  "#8FBCBB",
    "nord8":  "#88C0D0",
    "nord9":  "#81A1C1",
    "nord10": "#5E81AC",
    "nord11": "#BF616A",
    "nord12": "#D08770",
    "nord13": "#EBCB8B",
    "nord14": "#A3BE8C",
    "nord15": "#B48EAD",
}

marker_colors = {"ztfg": '#2A9D8F', "ztfr": '#E63946', "ztfi": '#F4A261'} # ztf filters

NORD_CYCLE = [NORD[k] for k in ["nord9", "nord10", "nord7", "nord14", "nord15", "nord12", "nord11", "nord13"]]


def nord_cmap(name: str = "nord_blues") -> LinearSegmentedColormap:
    if name == "nord_blues":
        return LinearSegmentedColormap.from_list("nord_blues", [NORD["nord6"], NORD["nord8"], NORD["nord10"], NORD["nord0"]])
    if name == "nord_reds":
        return LinearSegmentedColormap.from_list("nord_reds", [NORD["nord6"], NORD["nord12"], NORD["nord11"], NORD["nord0"]])
    return LinearSegmentedColormap.from_list(name, list(NORD.values()))


def setup_mpl_paper(*, usetex: bool = True) -> None:
    """Paper-friendly Matplotlib defaults.

    - Palatino (serif) for plots + math
    - Nord color cycle
    - Slightly larger labels / lines for print
    """
    # Cluster nodes may not provide a LaTeX installation; gracefully fall back.
    usetex = bool(usetex and shutil.which("latex"))
    has_palatino = any("Palatino" in f.name for f in fm.fontManager.ttflist)
    font_serif = ["Palatino", "DejaVu Serif"] if has_palatino else ["DejaVu Serif"]
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": font_serif,
        "mathtext.fontset": "custom" if has_palatino else "dejavuserif",
        "mathtext.rm": "Palatino" if has_palatino else "DejaVu Serif",
        "mathtext.it": "Palatino:italic" if has_palatino else "DejaVu Serif:italic",
        "mathtext.bf": "Palatino:bold" if has_palatino else "DejaVu Serif:bold",
        "text.usetex": usetex,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.prop_cycle": mpl.cycler(color=NORD_CYCLE),
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "legend.frameon": False,
        "grid.alpha": 0.25,
        "grid.linestyle": (0, (2, 4)),
    })
    if usetex:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{newpxtext,newpxmath,mathpazo}"

def savefig_pdf(fig: mpl.figure.Figure, path: Path, *, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight" if tight else None, dpi=300)


def style_axes_inward(ax: mpl.axes.Axes, *, minor: bool = True, grid_y: bool = False) -> None:
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, length=6, width=1.0)
    if minor:
        ax.tick_params(axis="both", which="minor", length=3)
        ax.minorticks_on()
    if grid_y:
        ax.grid(axis="y", which="major")
