import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ikob.configuration_definition import DecayCurveName
from ikob.constants import work_constants
from ikob.datasource import DataKey, DataSource, DataType
from ikob.utils import DTYPE, maybe_to_sparse

logger = logging.getLogger(__name__)

ALL_PREFERENCES = ["Auto", "Neutraal", "Fiets", "OV"]


def calculate_weights(generalized_travel_time, modality, preference, decay_curve_name: DecayCurveName):
    """
    Vectorised decay-curve application.

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


def calculate_single_weights(config, generalized_travel_time: DataSource) -> DataSource:
    """
    Section D2: travel-time decay curves for car, PT, bike, and E-bike.

    Parallelises over (part_of_day × income) using threads.
    NumPy releases the GIL, so threads get real concurrency.
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
        add_bike_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, config)
        add_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights)
        add_no_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights)
        add_free_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights)
        add_pt_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights)
        add_free_pt_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights)

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


def add_bike_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights, config):
    project_config = config["project"]
    modalities_bike = ["EFiets"] if project_config["fiets of E-fiets"]["E-fiets"] else ["Fiets"]
    for preference in ALL_PREFERENCES:
        for modality in modalities_bike:
            if preference == "Auto" or preference == "Fiets":
                key = DataKey("Fiets", part_of_day=part_of_day, regime=regimes, motive=motive_name, income=income)
                gtr_skim = generalized_travel_time.get(key)
                weight_matrix = calculate_weights(gtr_skim, modality, preference, decay_curve_name=decay_curve)
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


def add_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights):
    fuel_kinds = ["fossiel", "elektrisch"]
    for preference in ALL_PREFERENCES:
        for fuel_kind in fuel_kinds:
            key = DataKey(f"Auto_{fuel_kind}", part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name)
            gtr_skim = generalized_travel_time.get(key)

            weight_matrix = calculate_weights(gtr_skim, "Auto", preference, decay_curve)
            num_zones = _num_zones_from_weights(weight_matrix)
            key = DataKey(
                "Auto_vk",
                part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name,
                preference=preference, fuel_kind=fuel_kind,
                header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
            )
            weights.set(key, weight_matrix)


def add_no_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights):
    no_car_kinds = ["GeenAuto", "GeenRijbewijs"]
    no_car_preferences = ["Neutraal", "OV", "Fiets"]
    for preference in no_car_preferences:
        for no_car_kind in no_car_kinds:
            key = DataKey(f"{no_car_kind}", part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name)
            gtr_skim = generalized_travel_time.get(key)

            weight_matrix = calculate_weights(gtr_skim, "Auto", preference, decay_curve)
            num_zones = _num_zones_from_weights(weight_matrix)
            key = DataKey(
                f"{no_car_kind}_vk",
                part_of_day=part_of_day, income=income, regime=regimes, preference=preference, motive=motive_name,
                header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
            )
            weights.set(key, weight_matrix)


def add_pt_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights):
    for preference in ALL_PREFERENCES:
        key = DataKey("OV", part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name)
        gtr_skim = generalized_travel_time.get(key)

        weight_matrix = calculate_weights(gtr_skim, "OV", preference, decay_curve)
        num_zones = _num_zones_from_weights(weight_matrix)
        key = DataKey(
            "OV_vk",
            part_of_day=part_of_day, preference=preference, income=income, regime=regimes, motive=motive_name,
            header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
        )
        weights.set(key, weight_matrix)


def add_free_car_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights):
    key = DataKey("GratisAuto", part_of_day=part_of_day, income=income, regime=regimes, motive=motive_name)
    gtr_skim = generalized_travel_time.get(key)

    weight_matrix = calculate_weights(gtr_skim, "Auto", "Auto", decay_curve)
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


def add_free_pt_weights(part_of_day, regimes, motive_name, decay_curve, income, generalized_travel_time, weights):
    key = DataKey("GratisOV", part_of_day=part_of_day, regime=regimes, motive=motive_name)
    gtr_skim = generalized_travel_time.get(key)

    weight_matrix = calculate_weights(gtr_skim, "OV", "OV", decay_curve)
    num_zones = _num_zones_from_weights(weight_matrix)
    special_pt_kinds = ["Neutraal", "OV"]
    for special_pt_kind in special_pt_kinds:
        key = DataKey(
            "GratisOV_vk",
            part_of_day=part_of_day, preference=special_pt_kind, income=income, regime=regimes, motive=motive_name,
            header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
        )
        weights.set(key, weight_matrix)