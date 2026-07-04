"""Attach curve-library ToleranceSpecs to the right D2 weight computation.

The curve-library (ikob.tolerance_curves.CurveRegistry) is keyed by
'group' strings of the form '<base_group>_<income>' (see
tolerance_curves.KNOWN_GROUPS), e.g. 'GeenAuto_vkOV_laag'. D2
(ikob.single_weights) instead computes one weight matrix per
(kind-of-mobility, preference, income[, fuel_kind]) cell, and several
base_groups can share the *same* cell: most notably, the legacy decay
curve parameters (ikob.constants.work_constants) have never depended on
car-possession category, only on modality and preference, so e.g. the
OV component for 'WelAuto_vkOV_laag' and 'GeenAuto_vkOV_laag' has always
been one and the same matrix.

This module bridges the two granularities using the *same*
`_weight_cache_key` logic that D4-D7 already rely on to decide which
weight matrices are shared: two base_groups share a D2 cell iff their
`_weight_cache_key` (for the modality in question) is identical. When a
curve-library attachment would split a shared cell into two different
curves, that is treated as a configuration error rather than resolved
silently one way or another -- the same fail-fast philosophy already
used by ikob.constants._validate and ikob.tolerance_curves.CurveRegistry.
"""

import logging

from ikob.competition import _weight_cache_key
from ikob.tolerance_curves import BASE_GROUPS, CurveRegistry, ToleranceSpec

logger = logging.getLogger(__name__)


class ConflictingCurveAttachment(ValueError):
    """Raised when base_groups sharing one D2 weight matrix carry
    different tolerance curves in the library."""


def sibling_base_groups(representative_group: str, modality: str, income: str) -> list[str]:
    """All base_groups (from tolerance_curves.BASE_GROUPS) whose D2 weight
    computation for *modality* is identical to *representative_group*'s,
    for the given *income* -- i.e. the same set D4-D7 would consider
    'the same weight matrix' via `_weight_cache_key`."""
    target = _weight_cache_key(f"{representative_group}_{income}", modality)
    return [
        bg for bg in BASE_GROUPS
        if _weight_cache_key(f"{bg}_{income}", modality) == target
    ]


def resolve_spec_for_computation(
    curve_registry: CurveRegistry | None,
    representative_group: str,
    modality: str,
    income: str,
) -> ToleranceSpec | None:
    """Return the ToleranceSpec attached to this D2 computation, if any.

    Looks up every base_group that shares *representative_group*'s exact
    weight matrix for (modality, income) (see `sibling_base_groups`), and
    requires that they carry either no attachment or all the *same* one.

    Returns None when curve_registry is None/empty or no sibling has an
    attachment (the caller should then fall back to the legacy curve).

    Raises ConflictingCurveAttachment when siblings disagree.
    """
    if curve_registry is None or len(curve_registry) == 0:
        return None

    siblings = sibling_base_groups(representative_group, modality, income)

    found: dict[str, ToleranceSpec] = {}
    for base_group in siblings:
        group = f"{base_group}_{income}"
        spec = curve_registry.spec_for(group)
        if spec is not None:
            found[group] = spec

    if not found:
        return None

    distinct: dict[ToleranceSpec, list[str]] = {}
    for group, spec in found.items():
        distinct.setdefault(spec, []).append(group)

    if len(distinct) > 1:
        description = "; ".join(f"{groups} -> {spec}" for spec, groups in distinct.items())
        raise ConflictingCurveAttachment(
            f"Groups {sorted(found)} all share one underlying weight matrix in "
            f"the current pipeline for modality {modality!r} (their travel-time "
            "decay curve has never depended on car-possession category), but "
            f"the curve library attaches different tolerance curves to them: "
            f"{description}. Attach the same curve to all of them (a single "
            "library entry can list several groups), or remove the "
            "conflicting attachment."
        )

    (spec,) = distinct.keys()
    logger.debug("Curve-library attachment resolved for %s (modality=%s, income=%s): %s",
                representative_group, modality, income, spec)
    return spec