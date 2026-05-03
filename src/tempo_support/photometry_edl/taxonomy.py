from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Taxonomy:
    broad_classes: List[str]
    orig2broad: Dict[str, str]
    subclass_id2name: List[str]
    hierarchy_level_names: List[str] = field(default_factory=list)
    broad2hierarchy_path: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.broad_classes)

    @property
    def broad2id(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.broad_classes)}

    @property
    def id2broad_id(self) -> Dict[int, int]:
        b2id = self.broad2id
        return {i: b2id[self.orig2broad[name]] for i, name in enumerate(self.subclass_id2name)}

    @property
    def hierarchy_depth(self) -> int:
        return len(self.hierarchy_level_specs())

    def hierarchy_level_specs(self) -> List[Dict]:
        # Default fallback: single-level hierarchy equal to broad classes.
        raw_paths: Dict[str, List[str]] = {}
        for broad in self.broad_classes:
            p = self.broad2hierarchy_path.get(broad)
            raw_paths[broad] = list(p) if p else [broad]

        depth = max((len(v) for v in raw_paths.values()), default=1)
        paths: Dict[str, List[str]] = {}
        for broad, path in raw_paths.items():
            if len(path) < depth:
                path = path + [path[-1]] * (depth - len(path))
            paths[broad] = path

        names = list(self.hierarchy_level_names)
        if len(names) < depth:
            names += [f"level_{i + 1}" for i in range(len(names), depth)]
        names = names[:depth]

        out: List[Dict] = []
        for li in range(depth):
            node_names: List[str] = []
            broad_to_node: Dict[str, int] = {}
            for broad in self.broad_classes:
                node = paths[broad][li]
                if node not in node_names:
                    node_names.append(node)
                broad_to_node[broad] = node_names.index(node)
            out.append(
                {
                    "level_index": li,
                    "level_name": names[li],
                    "node_names": node_names,
                    "broad_to_node": broad_to_node,
                }
            )
        return out


SUBCLASS_ID2NAME = [
    "SN Ia",
    "SN Ib",
    "SN Ic",
    "SN II",
    "SN IIP",
    "SN IIn",
    "SN IIb",
    "Cataclysmic",
    "AGN",
    "Tidal Disruption Event",
]


DEFAULT_TAXONOMY = Taxonomy(
    broad_classes=["SNI", "SNII", "CV", "AGN", "TDE"],
    orig2broad={
        "SN Ia": "SNI",
        "SN Ib": "SNI",
        "SN Ic": "SNI",
        "SN II": "SNII",
        "SN IIP": "SNII",
        "SN IIn": "SNII",
        "SN IIb": "SNII",
        "Cataclysmic": "CV",
        "AGN": "AGN",
        "Tidal Disruption Event": "TDE",
    },
    subclass_id2name=SUBCLASS_ID2NAME,
    hierarchy_level_names=["domain", "family", "class"],
    broad2hierarchy_path={
        "SNI": ["Transient", "SN", "SNI"],
        "SNII": ["Transient", "SN", "SNII"],
        "TDE": ["Transient", "TDE", "TDE"],
        "AGN": ["Variable", "AGN", "AGN"],
        "CV": ["Variable", "CV", "CV"],
    },
)


PHOTOID_SN5_TAXONOMY = Taxonomy(
    broad_classes=["SNIa", "CCSN", "CV", "AGN", "TDE"],
    orig2broad={
        "SN Ia": "SNIa",
        "SN Ib": "CCSN",
        "SN Ic": "CCSN",
        "SN II": "CCSN",
        "SN IIP": "CCSN",
        "SN IIn": "CCSN",
        "SN IIb": "CCSN",
        "Cataclysmic": "CV",
        "AGN": "AGN",
        "Tidal Disruption Event": "TDE",
    },
    subclass_id2name=SUBCLASS_ID2NAME,
    hierarchy_level_names=["domain", "family", "class"],
    broad2hierarchy_path={
        "SNIa": ["Transient", "SN", "SNIa"],
        "CCSN": ["Transient", "SN", "CCSN"],
        "TDE": ["Transient", "TDE", "TDE"],
        "AGN": ["Variable", "AGN", "AGN"],
        "CV": ["Variable", "CV", "CV"],
    },
)


PHOTOID_SN6_TAXONOMY = Taxonomy(
    broad_classes=["SNIa", "SESN", "HRichCCSN", "CV", "AGN", "TDE"],
    orig2broad={
        "SN Ia": "SNIa",
        "SN Ib": "SESN",
        "SN Ic": "SESN",
        "SN II": "HRichCCSN",
        "SN IIP": "HRichCCSN",
        "SN IIn": "HRichCCSN",
        "SN IIb": "SESN",
        "Cataclysmic": "CV",
        "AGN": "AGN",
        "Tidal Disruption Event": "TDE",
    },
    subclass_id2name=SUBCLASS_ID2NAME,
    hierarchy_level_names=["domain", "family", "class"],
    broad2hierarchy_path={
        "SNIa": ["Transient", "SN", "SNIa"],
        "SESN": ["Transient", "SN", "SESN"],
        "HRichCCSN": ["Transient", "SN", "HRichCCSN"],
        "TDE": ["Transient", "TDE", "TDE"],
        "AGN": ["Variable", "AGN", "AGN"],
        "CV": ["Variable", "CV", "CV"],
    },
)


TRANSIENT_VARIABLE_5C_TAXONOMY = Taxonomy(
    broad_classes=["SNI", "SNII", "TDE", "AGN", "CV"],
    orig2broad={
        "SN Ia": "SNI",
        "SN Ib": "SNI",
        "SN Ic": "SNI",
        "SN II": "SNII",
        "SN IIP": "SNII",
        "SN IIn": "SNII",
        "SN IIb": "SNII",
        "Cataclysmic": "CV",
        "AGN": "AGN",
        "Tidal Disruption Event": "TDE",
    },
    subclass_id2name=SUBCLASS_ID2NAME,
    hierarchy_level_names=["domain", "family", "class"],
    broad2hierarchy_path={
        "SNI": ["Transient", "SN", "SNI"],
        "SNII": ["Transient", "SN", "SNII"],
        "TDE": ["Transient", "TDE", "TDE"],
        "AGN": ["Variable", "AGN", "AGN"],
        "CV": ["Variable", "CV", "CV"],
    },
)


TRANSIENT_VARIABLE_4C_TAXONOMY = Taxonomy(
    broad_classes=["SN", "TDE", "AGN", "CV"],
    orig2broad={
        "SN Ia": "SN",
        "SN Ib": "SN",
        "SN Ic": "SN",
        "SN II": "SN",
        "SN IIP": "SN",
        "SN IIn": "SN",
        "SN IIb": "SN",
        "Cataclysmic": "CV",
        "AGN": "AGN",
        "Tidal Disruption Event": "TDE",
    },
    subclass_id2name=SUBCLASS_ID2NAME,
    hierarchy_level_names=["domain", "family", "class"],
    broad2hierarchy_path={
        "SN": ["Transient", "SN", "SN"],
        "TDE": ["Transient", "TDE", "TDE"],
        "AGN": ["Variable", "AGN", "AGN"],
        "CV": ["Variable", "CV", "CV"],
    },
)


TAXONOMY_PRESETS: Dict[str, Taxonomy] = {
    "default": DEFAULT_TAXONOMY,
    "sn_photoid_5c": PHOTOID_SN5_TAXONOMY,
    "sn_photoid_6c": PHOTOID_SN6_TAXONOMY,
    "transient_variable_5c": TRANSIENT_VARIABLE_5C_TAXONOMY,
    "transient_variable_4c": TRANSIENT_VARIABLE_4C_TAXONOMY,
}


TAXONOMY_ALIASES: Dict[str, str] = {
    "orig": "default",
    "legacy": "default",
    "sn5": "sn_photoid_5c",
    "photoid5": "sn_photoid_5c",
    "photoid_5": "sn_photoid_5c",
    "sn6": "sn_photoid_6c",
    "photoid6": "sn_photoid_6c",
    "photoid_6": "sn_photoid_6c",
    "tv5": "transient_variable_5c",
    "transient_variable5": "transient_variable_5c",
    "transientvariable5": "transient_variable_5c",
    "tv4": "transient_variable_4c",
    "transient_variable4": "transient_variable_4c",
    "transientvariable4": "transient_variable_4c",
}


def list_taxonomy_presets() -> List[str]:
    return sorted(TAXONOMY_PRESETS.keys())


def get_taxonomy(preset: Optional[str]) -> Taxonomy:
    if preset is None:
        return DEFAULT_TAXONOMY
    key = str(preset).strip().lower()
    key = TAXONOMY_ALIASES.get(key, key)
    if key not in TAXONOMY_PRESETS:
        raise ValueError(
            f"Unknown taxonomy preset `{preset}`. "
            f"Available: {sorted(TAXONOMY_PRESETS.keys())}"
        )
    return TAXONOMY_PRESETS[key]
