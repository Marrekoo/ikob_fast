"""User-authored mapping from raw GeoPackage columns onto the per-zone
SEGS variables IKOB needs, plus the small urbanization-grade lookup
tables (WelAuto, GeenAuto, GeenRijbewijs, Voorkeuren, VoorkeurenGeenAuto).

Why this exists
----------------
CBS opendata buurten GeoPackages contain dozens of attribute columns
(population, cars, income, urbanization, ...) all on one 'buurten'
layer, joined by buurtcode. Nothing in that raw table is named the way
IKOB's internal SEGS ids are (e.g. "CBS_autos_per_huishouden", "WelAuto"),
and several fields IKOB needs are *derived* -- a ratio of two raw
columns, or a total column split into per-class shares -- rather than a
single column that can be read verbatim.

This module lets the wizard capture, once, how each required IKOB input
is obtained from the source data:

  - "direct":  the value already sits in one column, read verbatim.
  - "ratio":   value = teller_kolom / noemer_kolom * schaal
               (e.g. cars per household = totaal_autos / totaal_huishoudens).
  - "shares":  for a multi-class variable, value_j = totaal_kolom *
               (aandeel_kolom_j / sum_of_shares) -- shares may be given
               as fractions or percentages, since they are normalised
               per row before being applied.

and resolves that mapping into the plain per-zone arrays / small lookup
tables ikob.datasource.SegsSource ultimately needs -- replacing a
convention-based "guess the layer name" approach with an explicit,
wizard-reviewed mapping.

The mapping is stored as a plain (JSON-serialisable) dict in the project
configuration, under project.paden.segs_mapping, mirroring the
config-as-plain-dict convention used throughout ikob (see e.g.
ikob.tolerance_curves.spec_to_dict/spec_from_dict for the same
philosophy applied to tolerance curves).

Note on urbanization-grade lookup tables: WelAuto, GeenAuto,
GeenRijbewijs, Voorkeuren, and VoorkeurenGeenAuto are small survey-based
tables with one row per *stedelijkheidsgraad* (urbanization grade,
1-5) -- see distribute_over_groups.py, where e.g.
with_car_segs[urbanization[i]] selects the row for zone i's
urbanization grade. They are not per-buurt data, so they are mapped
separately from the per-zone variables above: either from a small gpkg
layer keyed by grade, or typed in directly.
"""

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class SegsMappingError(ValueError):
    """Raised for an incomplete or inconsistent SEGS column mapping."""


PER_ZONE_VARIABLES = [
    ("Stedelijkheidsgraad", "Stedelijkheidsgraad (1 = zeer sterk stedelijk .. 5 = niet stedelijk)"),
    ("CBS_autos_per_huishouden", "Autobezit: auto's per 100 huishoudens"),
]

URBANIZATION_LOOKUP_VARIABLES = [
    ("WelAuto", "Percentage met auto per inkomensklasse", ["laag", "middellaag", "middelhoog", "hoog"]),
    ("GeenAuto", "Percentage zonder auto (wel rijbewijs) per inkomensklasse",
     ["laag", "middellaag", "middelhoog", "hoog"]),
    ("GeenRijbewijs", "Percentage zonder rijbewijs per inkomensklasse",
     ["laag", "middellaag", "middelhoog", "hoog"]),
    ("Voorkeuren", "Modaliteitsvoorkeur (met auto) per klasse", ["Auto", "Neutraal", "Fiets", "OV"]),
    ("VoorkeurenGeenAuto", "Modaliteitsvoorkeur (zonder auto) per klasse", ["Neutraal", "Fiets", "OV"]),
]

MOTIVE_CLASS_VARIABLES = [
    # (config field under project.motief, label, income-class labels)
    ("reizende populatie", "Reizende populatie per inkomensklasse", ["laag", "middellaag", "middelhoog", "hoog"]),
    ("bestemmingsplaatsen", "Bestemmingen (bv. arbeidsplaatsen) per inkomensklasse",
     ["laag", "middellaag", "middelhoog", "hoog"]),
]


def empty_mapping() -> dict:
    return {"per_zone": {}, "urbanization_lookup": {}, "motive": {}}


def _require_columns(gdf, columns, label):
    missing = [c for c in columns if not c or c not in gdf.columns]
    if missing:
        raise SegsMappingError(f"'{label}': ontbrekende of niet-gekozen kolom(men) {missing}.")


def _resolve_single(gdf, entry: dict, label: str) -> npt.NDArray:
    kind = entry.get("soort")
    if kind == "direct":
        column = entry.get("kolom")
        _require_columns(gdf, [column], label)
        return gdf[column].to_numpy(dtype=float)
    if kind == "ratio":
        numerator = entry.get("teller_kolom")
        denominator = entry.get("noemer_kolom")
        scale = float(entry.get("schaal", 1.0) or 1.0)
        _require_columns(gdf, [numerator, denominator], label)
        num = gdf[numerator].to_numpy(dtype=float)
        den = gdf[denominator].to_numpy(dtype=float)
        safe_den = np.where(den != 0, den, 1.0)
        return np.where(den != 0, num / safe_den, 0.0) * scale
    raise SegsMappingError(f"'{label}': onbekend type mapping '{kind}'.")


def _resolve_columns(gdf, entry: dict, num_classes: int, label: str) -> npt.NDArray:
    kind = entry.get("soort")
    if kind == "direct":
        columns = entry.get("kolommen", [])
        if len(columns) != num_classes:
            raise SegsMappingError(f"'{label}': verwacht {num_classes} kolommen, maar {len(columns)} opgegeven.")
        _require_columns(gdf, columns, label)
        return gdf[columns].to_numpy(dtype=float)
    if kind == "shares":
        total_column = entry.get("totaal_kolom")
        share_columns = entry.get("aandeel_kolommen", [])
        if len(share_columns) != num_classes:
            raise SegsMappingError(
                f"'{label}': verwacht {num_classes} aandeel-kolommen, maar {len(share_columns)} opgegeven."
            )
        _require_columns(gdf, [total_column, *share_columns], label)
        total = gdf[total_column].to_numpy(dtype=float)
        shares = gdf[share_columns].to_numpy(dtype=float)
        # Shares may be given as fractions (0-1) or percentages (0-100);
        # normalise so each row sums to 1, working for either convention
        # without asking the user to pick one up front.
        row_sums = shares.sum(axis=1, keepdims=True)
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        normalised = np.where(row_sums > 0, shares / safe_sums, 0.0)
        return total[:, None] * normalised
    raise SegsMappingError(f"'{label}': onbekend type mapping '{kind}'.")


def _resolve_urbanization_lookup(entry: dict, class_labels: list[str], label: str) -> npt.NDArray:
    kind = entry.get("soort")
    if kind == "handmatig":
        rows = entry.get("waarden", [])
        if len(rows) != 5 or any(len(row) != len(class_labels) for row in rows):
            raise SegsMappingError(f"'{label}': verwacht 5 rijen van {len(class_labels)} waarden.")
        return np.asarray(rows, dtype=float)

    if kind == "laag":
        from ikob import geo_utils

        gpkg_path = entry.get("bestand")
        layer = entry.get("laag")
        key_column = entry.get("sleutel_kolom")
        value_columns = entry.get("waarde_kolommen", [])
        if len(value_columns) != len(class_labels):
            raise SegsMappingError(
                f"'{label}': verwacht {len(class_labels)} waarde-kolommen, maar {len(value_columns)} opgegeven."
            )
        df = geo_utils.read_layer(gpkg_path, layer)
        _require_columns(df, [key_column, *value_columns], label)
        keys = df[key_column].astype(int)
        if sorted(keys.tolist()) != [1, 2, 3, 4, 5]:
            raise SegsMappingError(
                f"'{label}': kolom '{key_column}' in laag '{layer}' moet exact de "
                f"waarden 1 t/m 5 bevatten (stedelijkheidsgraad); gevonden: {sorted(keys.unique().tolist())}."
            )
        ordered = df.set_index(keys).loc[[1, 2, 3, 4, 5]]
        return ordered[value_columns].to_numpy(dtype=float)

    raise SegsMappingError(f"'{label}': onbekend type mapping '{kind}'.")


# ── Public entry points, used by ikob.datasource._GpkgSegsInputBackend ──

def resolve_per_zone_variable(gdf, mapping: dict, variable_id: str) -> npt.NDArray:
    entry = mapping.get("per_zone", {}).get(variable_id)
    label = dict((v[0], v[1]) for v in PER_ZONE_VARIABLES).get(variable_id, variable_id)
    if entry is None:
        raise SegsMappingError(f"Geen kolom-mapping opgegeven voor '{label}'.")
    return _resolve_single(gdf, entry, label)


def resolve_motive_variable(gdf, mapping: dict, field_name: str) -> npt.NDArray:
    spec = dict((v[0], (v[1], v[2])) for v in MOTIVE_CLASS_VARIABLES).get(field_name)
    if spec is None:
        raise SegsMappingError(f"Onbekend motief-veld '{field_name}'.")
    label, class_labels = spec
    entry = mapping.get("motive", {}).get(field_name)
    if entry is None:
        raise SegsMappingError(f"Geen kolom-mapping opgegeven voor '{label}'.")
    return _resolve_columns(gdf, entry, len(class_labels), label)


def resolve_urbanization_lookup(mapping: dict, variable_id: str) -> npt.NDArray:
    spec = dict((v[0], (v[1], v[2])) for v in URBANIZATION_LOOKUP_VARIABLES).get(variable_id)
    if spec is None:
        raise SegsMappingError(f"Onbekende opzoektabel '{variable_id}'.")
    label, class_labels = spec
    entry = mapping.get("urbanization_lookup", {}).get(variable_id)
    if entry is None:
        raise SegsMappingError(f"Geen mapping opgegeven voor opzoektabel '{label}'.")
    return _resolve_urbanization_lookup(entry, class_labels, label)


def mapping_is_complete(mapping: dict, motive_fields: list[str] | None = None) -> tuple[bool, list[str]]:
    """Report every required variable still missing from *mapping*,
    without needing the actual gpkg file (a lightweight, wizard-friendly
    completeness check; full resolution -- which also validates that
    referenced columns still exist -- only happens once the pipeline
    actually reads the data)."""
    missing = []
    for var_id, label in PER_ZONE_VARIABLES:
        if var_id not in mapping.get("per_zone", {}):
            missing.append(label)
    for var_id, label, _ in URBANIZATION_LOOKUP_VARIABLES:
        if var_id not in mapping.get("urbanization_lookup", {}):
            missing.append(label)
    for field_name, label, _ in MOTIVE_CLASS_VARIABLES:
        if motive_fields and field_name not in motive_fields:
            continue
        if field_name not in mapping.get("motive", {}):
            missing.append(label)
    return (len(missing) == 0), missing