"""Visualizations for TEMPO classification results.

Mirrors the look of ``oracle_support/plot_oracle.py`` — a plotly sunburst
walking the model's hierarchy. Branch values come from the leaf-level
final probabilities propagated up through the taxonomy paths.
"""
import math
from pathlib import Path

import plotly.graph_objects as go


def _fmt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NaN"
    if 0 < p < 1e-4:
        return "<0.01%"
    return f"{p * 100:.2f}%"


def _hierarchy_scores(result, taxonomy):
    """Build a flat ``{node_name: probability}`` dict for every taxonomy node.

    Internal-node probabilities are obtained by summing the leaf-final
    probabilities of all descendants — guaranteeing the sunburst's
    branchvalues='total' is exactly consistent.
    """
    leaf = result["leaf"]
    leaf_names = list(leaf["class_names"])
    leaf_probs = [float(p) for p in leaf["probs_final"]]
    paths = {
        name: list(taxonomy.broad2hierarchy_path.get(name, [name]))
        for name in leaf_names
    }
    for name in leaf_names:
        if not paths[name] or paths[name][-1] != name:
            paths[name] = paths[name] + [name]

    scores = {}
    for name, p in zip(leaf_names, leaf_probs):
        for ancestor in dict.fromkeys(paths[name]):
            scores[ancestor] = scores.get(ancestor, 0.0) + p
    return scores, paths, leaf_names


def _level_order_with_parents(paths, leaf_names):
    """Return (ids, parents) in level-order so siblings stay adjacent."""
    depth = max(len(paths[name]) for name in leaf_names)
    seen = set()
    ids = []
    parents = []
    for li in range(depth):
        emitted_at_level = []
        for name in leaf_names:
            if li >= len(paths[name]):
                continue
            node = paths[name][li]
            if node in seen:
                continue
            seen.add(node)
            emitted_at_level.append(node)
            parent = paths[name][li - 1] if li > 0 else ""
            ids.append(node)
            parents.append(parent)
    return ids, parents


def plot_tempo_sunburst(result, taxonomy, *, title=None, font_size=16):
    """Create a sunburst figure for TEMPO hierarchical classification.

    Args:
        result: dict returned by :func:`tempo_support.tempo_boom_ztf.run_tempo`.
        taxonomy: TEMPO taxonomy (``tempo_boom_ztf.get_taxonomy()``).
        title: optional title at the top of the figure.
        font_size: leaf-label point size.

    Returns:
        plotly.graph_objects.Figure
    """
    scores, paths, leaf_names = _hierarchy_scores(result, taxonomy)
    ids, parents = _level_order_with_parents(paths, leaf_names)

    labels, values, texts = [], [], []
    for node in ids:
        p = float(scores.get(node, 0.0))
        labels.append(f"<b>{node}</b>")
        values.append(p)
        texts.append(f"<b>{_fmt(p)}</b>")

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            text=texts,
            textinfo="label+text",
            textfont=dict(size=font_size),
            hovertemplate="<b>%{label}</b><br>P(class) = %{text}<extra></extra>",
            branchvalues="total",
            marker=dict(line=dict(width=2, color="white")),
            insidetextorientation="radial",
        )
    )

    if title:
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=24, family="Arial")),
            margin=dict(t=60, l=10, r=10, b=10),
            width=750,
            height=750,
        )
    else:
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), width=750, height=750)
    return fig


def save_sunburst(result, taxonomy, out_path, *, title=None, font_size=12, scale=2):
    fig = plot_tempo_sunburst(result, taxonomy, title=title, font_size=font_size)
    out_path = Path(out_path)
    fig.write_image(str(out_path), scale=scale)
    return out_path


if __name__ == "__main__":
    sample_result = {
        "leaf": {
            "class_names": ["SNI", "SNII", "TDE", "AGN", "CV"],
            "probs_final": [0.9774, 0.0172, 0.0017, 0.0019, 0.0018],
        }
    }

    class _StubTaxonomy:
        broad2hierarchy_path = {
            "SNI": ["Transient", "SN", "SNI"],
            "SNII": ["Transient", "SN", "SNII"],
            "TDE": ["Transient", "TDE", "TDE"],
            "AGN": ["Variable", "AGN", "AGN"],
            "CV": ["Variable", "CV", "CV"],
        }

    fig = plot_tempo_sunburst(sample_result, _StubTaxonomy(), title="TEMPO — demo")
    fig.write_image("demo_tempo_sunburst.png", scale=2)
    print("saved: demo_tempo_sunburst.png")
