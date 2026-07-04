import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

from ikob.configuration_definition import DecayCurveName
from ikob.constants import work_constants
from ikob.curve_attachment import resolve_spec_for_computation
from ikob.datasource import DataKey, DataSource, DataType
from ikob.tolerance_curves import CurveRegistry, calculate_weights_2d
from ikob.utils import DTYPE, maybe_to_sparse

logger = logging.getLogger(__name__)

ALL_PREFERENCES = ["Auto", "Neutraal", "Fiets", "OV"]


def calculate_weights(generalized_travel_time, modality, preference, decay_curve_name: DecayCurveName):
    """
    Vectorised decay-curve application (legacy path: constants.py tables).

    Replaces the original O(n²) Python loop with a single NumPy broadcast.
    """
    alpha, omega, scaling = work_constants(modality, preference, decay_curve_name)
    gtt = np.asarray(generalized_travel_time, dtype=DTYPE)

    mask = gtt < 180.0
    # Compute sigmoid only where needed to avoid overflow / wasted work
    weight_matrix = np.zeros_like(gtt, dtype=DTYPE)
    exponent = (-omega + gtt[mask]) * alpha
    weight_matrix[mask] = (1.0 / (1.0 + np.exp(exponent))) * scaling

    # Zero out negligible weights
    weight_matrix[weight_matrix < 0.001] = 0.0

    return maybe_to_sparse(weight_matrix)


def _resolve_and_compute(
    generalized_travel_time: DataSource,
    gtr_skim,
    modality: str,
    preference: str,
    decay_curve_name: DecayCurveName,
    curve_registry: CurveRegistry | None,
    representative_group: str,
    income: str,
    tijd_geld_key: DataKey | None,
):
    """Compute one D2 weight matrix, honoring a curve-library attachment
    if one applies to this (modality, income) computation.

    *tijd_geld_key* is the DataKey (subtopic="") whose 'tijd'/'geld'
    siblings hold the separate time/money skims this computation was
    collapsed from (see generalized_travel_time._maybe_store_time_money).
    They are only present when D1 was called with a non-empty
    curve_registry; if absent, a 'fixedVOT' attachment still works
    (applied directly to the already-collapsed gtr_skim, i.e. money=0,
    which is exact whenever the attachment's tau doesn't need to differ
    from the pipeline's own TVOM), but a 'copula' attachment cannot be
    honored and raises instead of silently falling back.
    """
    spec = resolve_spec_for_computation(curve_registry, representative_group, modality, income)
    if spec is None:
        return calculate_weights(gtr_skim, modality, preference, decay_curve_name)

    logger.info("Applying curve-library attachment for %s (modality=%s, income=%s).",
               representative_group, modality, income)

    if tijd_geld_key is not None:
        tijd_key = replace(tijd_geld_key, subtopic="tijd")
        geld_key = replace(tijd_geld_key, subtopic="geld")
        if tijd_key in generalized_travel_time.cache and geld_key in generalized_travel_time.cache:
            t_skim = generalized_travel_time.get(tijd_key)
            m_skim = generalized_travel_time.get(geld_key)
            return calculate_weights_2d(t_skim, m_skim, spec)

    if spec.mode != "fixedVOT":
        raise ValueError(
            f"{representative_group!r} (income={income!r}) has a 'copula' "
            "tolerance curve attached, but no decomposed time/money skim is "
            "available for it. Pass the same curve_registry into "
            "generalized_travel_time(config, curve_registry=...) so D1 also "
            "stores the decomposed skims."
        )
    logger.warning(
        "No decomposed time/money skim available for %s (income=%s); applying "
        "the attached fixedVOT curve directly to the already-collapsed travel "
        "time (equivalent unless its tau differs from the pipeline's own TVOM).",
        representative_group, income,
    )
    return calculate_weights_2d(gtr_skim, np.zeros_like(np.asarray(gtr_skim, dtype=DTYPE)), spec)


def calculate_single_weights(config, generalized_travel_time: DataSource,
                             curve_registry: CurveRegistry | None = None) -> DataSource:
    """
    Section D2: travel-time decay curves for car, PT, bike, and E-bike.

    Parallelises over (part_of_day × income) using threads.
    NumPy releases the GIL, so threads get real concurrency.

    *curve_registry*, if given, overrides the legacy constants-table
    decay curve for any group it attaches a ToleranceSpec to -- see
    ikob.curve_attachment for how base_groups map onto these
    computations, including the conflict-detection rules.
    """
    logger.info("Starting step: Weights (travel time decay curves) for car, PT, bike, and E-bike.")

    project_config = config["project"]
    skims_config = config["skims"]

    part_of_days = skims_config["dagsoort"]
    motive_name = project_config["motief"]["naam"]
    decay_curve = project_config["motief"]["reistijdvervalscurve"]
    regimes = project_config["beprijzingsregime"]

    incomes = ["hoog", "middelhoog", "middellaag", "laag"]

    weights = DataSource(config, DataType.WEIGHTS)

    def _process(part_of_day, income):
        add_bike_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, config, curve_registry)
        add_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, curve_registry)
        add_no_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, curve_registry)
        add_free_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, curve_registry)
        add_pt_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, curve_registry)
        add_free_pt_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, curve_registry)

    with ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(_process, pod, inc)
            for pod in part_of_days
            for inc in incomes
        ]
        for f in as_completed(futures):
            f.result()  # propagate exceptions

    return weights


# ── helpers ──────────────────────────────────────────────────────────

def _num_zones_from_weights(w):
    try:
        return w.shape[0]
    except AttributeError:
        return len(w)


def add_bike_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, config,
                     curve_registry=None):
    project_config = config["project"]
    modalities_bike = ["EFiets"] if project_config["fiets of E-fiets"]["E-fiets"] else ["Fiets"]
    for preference in ALL_PREFERENCES:
        for modality in modalities_bike:
            if preference == "Auto" or preference == "Fiets":
                key = DataKey("Fiets", part_of_day=part_of_day, regime=regimes, motive=motive_name, income=income)
                gtr_skim = generalized_travel_time.get(key)

                # The bike component only depends on whether the group's own
                # preference is Fiets, not on car-possession category; any
                # base_group with a matching preference is representative.
                representative_group = "WelAuto_vkFiets" if preference == "Fiets" else "WelAuto_vkAuto"
                weight_matrix = _resolve_and_compute(
                    generalized_travel_time, gtr_skim, modality, preference, decay_curve,
                    curve_registry, representative_group, income, key,
                )
                num_zones = _num_zones_from_weights(weight_matrix)

                if preference == "Auto":
                    key = DataKey(
                        "Fiets_vk",
                        part_of_day=part_of_day, regime=regimes, motive=motive_name, income=income,
                        header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
                    )
                else:
                    key = DataKey(
                        "Fiets_vk",
                        part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name,
                        preference=preference,
                        header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
                    )
                # No .copy() – the matrix is freshly created each iteration
                weights.set(key, weight_matrix)


def add_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights,
                    curve_registry=None):
    fuel_kinds = ["fossiel", "elektrisch"]
    for preference in ALL_PREFERENCES:
        for fuel_kind in fuel_kinds:
            key = DataKey(f"Auto_{fuel_kind}", part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name)
            gtr_skim = generalized_travel_time.get(key)

            representative_group = f"WelAuto_vk{preference}"
            weight_matrix = _resolve_and_compute(
                generalized_travel_time, gtr_skim, "Auto", preference, decay_curve,
                curve_registry, representative_group, income, key,
            )
            num_zones = _num_zones_from_weights(weight_matrix)
            key = DataKey(
                "Auto_vk",
                part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name,
                preference=preference, fuel_kind=fuel_kind,
                header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
            )
            weights.set(key, weight_matrix)


def add_no_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights,
                       curve_registry=None):
    no_car_kinds = ["GeenAuto", "GeenRijbewijs"]
    no_car_preferences = ["Neutraal", "OV", "Fiets"]
    for preference in no_car_preferences:
        for no_car_kind in no_car_kinds:
            key = DataKey(f"{no_car_kind}", part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name)
            gtr_skim = generalized_travel_time.get(key)

            representative_group = f"{no_car_kind}_vk{preference}"
            weight_matrix = _resolve_and_compute(
                generalized_travel_time, gtr_skim, "Auto", preference, decay_curve,
                curve_registry, representative_group, income, key,
            )
            num_zones = _num_zones_from_weights(weight_matrix)
            key = DataKey(
                f"{no_car_kind}_vk",
                part_of_day=part_of_day, income=income, regime=regimes, preference=preference, motive=motive_name,
                header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
            )
            weights.set(key, weight_matrix)


def add_pt_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights,
                   curve_registry=None):
    for preference in ALL_PREFERENCES:
        key = DataKey("OV", part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name)
        gtr_skim = generalized_travel_time.get(key)

        # The OV component only depends on preference, not car-possession
        # category (see module docstring in ikob.curve_attachment).
        representative_group = f"WelAuto_vk{preference}"
        weight_matrix = _resolve_and_compute(
            generalized_travel_time, gtr_skim, "OV", preference, decay_curve,
            curve_registry, representative_group, income, key,
        )
        num_zones = _num_zones_from_weights(weight_matrix)
        key = DataKey(
            "OV_vk",
            part_of_day=part_of_day, preference=preference, income=income, regime=regimes, motive=motive_name,
            header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
        )
        weights.set(key, weight_matrix)


def add_free_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights,
                         curve_registry=None):
    key = DataKey("GratisAuto", part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name)
    gtr_skim = generalized_travel_time.get(key)

    weight_matrix = _resolve_and_compute(
        generalized_travel_time, gtr_skim, "Auto", "Auto", decay_curve,
        curve_registry, "GratisAuto", income, key,
    )
    num_zones = _num_zones_from_weights(weight_matrix)
    free_car_preferences = ["Neutraal", "Auto"]
    for preference in free_car_preferences:
        key = DataKey(
            "GratisAuto_vk",
            part_of_day=part_of_day, preference=preference, income=income, regime=regimes, motive=motive_name,
            header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
        )
        # Same matrix for both preferences – no mutation downstream so aliasing is safe
        weights.set(key, weight_matrix)


def add_free_pt_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights,
                        curve_registry=None):
    key = DataKey("GratisOV", part_of_day=part_of_day, regime=regimes, motive=motive_name)
    gtr_skim = generalized_travel_time.get(key)

    weight_matrix = _resolve_and_compute(
        generalized_travel_time, gtr_skim, "OV", "OV", decay_curve,
        curve_registry, "WelAuto_GratisOV", income, key,
    )
    num_zones = _num_zones_from_weights(weight_matrix)
    special_pt_kinds = ["Neutraal", "OV"]
    for special_pt_kind in special_pt_kinds:
        key = DataKey(
            "GratisOV_vk",
            part_of_day=part_of_day, preference=special_pt_kind, income=income, regime=regimes, motive=motive_name,
            header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
        )
        weights.set(key, weight_matrix)