import logging

from ikob.datasource import DataKey, DataSource, DataType
from ikob.lazy_combined_weights import LazyCombinedDataSource
from ikob.utils import sparse_maximum

logger = logging.getLogger(__name__)


def has_preference(kind_car, kind_pt, preference):
    if kind_car == "GeenAuto" or kind_car == "GeenRijbewijs":
        if preference == "Auto":
            return False
        if kind_pt == "GratisOV":
            return preference == "OV"
        return True
    elif kind_car == "GratisAuto":
        if kind_pt == "GratisOV":
            return preference == "Neutraal"
        return preference == "Auto"
    elif kind_pt == "GratisOV":
        return preference == "OV"
    return True


def calculate_combined_weights(config, single_weights: DataSource) -> LazyCombinedDataSource:
    """
    Section D3 – register recipes for combined (max) weights.

    ROUND-2 CHANGE: no matrices are computed here.  Only lightweight
    ``register(key, [component_keys])`` calls are made.  The actual
    ``sparse_maximum`` is deferred to first access in D4–D7.
    """
    logger.info("Starting step: Register combined-weight recipes.")

    project_config = config["project"]
    skims_config = config["skims"]

    motive_name = project_config["motief"]["naam"]
    regimes = project_config["beprijzingsregime"]
    part_of_days = skims_config["dagsoort"]

    incomes = ["hoog", "middelhoog", "middellaag", "laag"]
    preferences = ["Auto", "Neutraal", "Fiets", "OV"]
    modalities_bike = ["Fiets"]
    car_kinds = ["Auto", "GeenAuto", "GeenRijbewijs", "GratisAuto"]
    pt_kinds = ["OV", "GratisOV"]
    fuel_kinds = ["fossiel", "elektrisch"]

    combined_weights = LazyCombinedDataSource(config, single_weights)

    # Grab num_zones from any already-cached single weight
    sample_key = DataKey("Fiets_vk", part_of_day=part_of_days[0], preference="",
                         income=incomes[0], regime=regimes, motive=motive_name)
    sample = single_weights.get(sample_key)
    try:
        num_zones = sample.shape[0]
    except AttributeError:
        num_zones = len(sample)

    def _hdr():
        return DataKey.zone_header(num_zones)

    def _idx():
        return DataKey.zone_index(num_zones)

    for part_of_day in part_of_days:
        for income in incomes:
            for preference in preferences:
                for modality_bike in modalities_bike:
                    preference_bike = "Fiets" if preference == "Fiets" else ""
                    bike_key = DataKey(f"{modality_bike}_vk", part_of_day=part_of_day,
                                       preference=preference_bike, income=income,
                                       regime=regimes, motive=motive_name)

                    # ── PT + Bike ────────────────────────────────
                    for pt_kind in pt_kinds:
                        if not has_preference("Auto", pt_kind, preference):
                            continue
                        pt_key = DataKey(f"{pt_kind}_vk", part_of_day=part_of_day,
                                         preference=preference, income=income,
                                         regime=regimes, motive=motive_name)
                        combined_key = DataKey(
                            f"{pt_kind}_{modality_bike}_vk", part_of_day=part_of_day,
                            income=income, regime=regimes, motive=motive_name,
                            preference=preference, subtopic="combinaties",
                            header=_hdr(), index=_idx())
                        combined_weights.register(combined_key, [bike_key, pt_key])

                    # ── Car + Bike ───────────────────────────────
                    for car_kind in car_kinds:
                        if not has_preference(car_kind, "OV", preference):
                            continue
                        if car_kind == "Auto":
                            for fuel_kind in fuel_kinds:
                                car_key = DataKey(f"{car_kind}_vk", part_of_day=part_of_day,
                                                  preference=preference, income=income,
                                                  regime=regimes, motive=motive_name,
                                                  fuel_kind=fuel_kind)
                                combined_key = DataKey(
                                    f"{car_kind}_{modality_bike}_vk", part_of_day=part_of_day,
                                    income=income, regime=regimes, motive=motive_name,
                                    preference=preference, subtopic="combinaties",
                                    fuel_kind=fuel_kind, header=_hdr(), index=_idx())
                                combined_weights.register(combined_key, [bike_key, car_key])
                        else:
                            car_key = DataKey(f"{car_kind}_vk", part_of_day=part_of_day,
                                              preference=preference, income=income,
                                              regime=regimes, motive=motive_name)
                            combined_key = DataKey(
                                f"{car_kind}_{modality_bike}_vk", part_of_day=part_of_day,
                                income=income, regime=regimes, motive=motive_name,
                                preference=preference, subtopic="combinaties",
                                header=_hdr(), index=_idx())
                            combined_weights.register(combined_key, [bike_key, car_key])

                # ── Car + PT ─────────────────────────────────
                for pt_kind in pt_kinds:
                    pt_key = DataKey(f"{pt_kind}_vk", part_of_day=part_of_day,
                                     preference=preference, income=income,
                                     regime=regimes, motive=motive_name)
                    for car_kind in car_kinds:
                        if not has_preference(car_kind, pt_kind, preference):
                            continue
                        if car_kind == "Auto":
                            for fuel_kind in fuel_kinds:
                                car_key = DataKey(f"{car_kind}_vk", part_of_day=part_of_day,
                                                  preference=preference, income=income,
                                                  regime=regimes, motive=motive_name,
                                                  fuel_kind=fuel_kind)
                                combined_key = DataKey(
                                    f"{car_kind}_{pt_kind}_vk", part_of_day=part_of_day,
                                    income=income, regime=regimes, motive=motive_name,
                                    preference=preference, subtopic="combinaties",
                                    fuel_kind=fuel_kind, header=_hdr(), index=_idx())
                                combined_weights.register(combined_key, [pt_key, car_key])
                        else:
                            car_key = DataKey(f"{car_kind}_vk", part_of_day=part_of_day,
                                              preference=preference, income=income,
                                              regime=regimes, motive=motive_name)
                            combined_key = DataKey(
                                f"{car_kind}_{pt_kind}_vk", part_of_day=part_of_day,
                                income=income, regime=regimes, motive=motive_name,
                                preference=preference, subtopic="combinaties",
                                header=_hdr(), index=_idx())
                            combined_weights.register(combined_key, [pt_key, car_key])

                # ── Car + PT + Bike ──────────────────────────
                for modality_bike in modalities_bike:
                    preference_bike = "Fiets" if preference == "Fiets" else ""
                    bike_key = DataKey(f"{modality_bike}_vk", part_of_day=part_of_day,
                                       preference=preference_bike, income=income,
                                       regime=regimes, motive=motive_name)
                    for pt_kind in pt_kinds:
                        pt_key = DataKey(f"{pt_kind}_vk", part_of_day=part_of_day,
                                         preference=preference, income=income,
                                         regime=regimes, motive=motive_name)
                        for car_kind in car_kinds:
                            if not has_preference(car_kind, pt_kind, preference):
                                continue
                            if car_kind == "Auto":
                                for fuel_kind in fuel_kinds:
                                    car_key = DataKey(f"{car_kind}_vk", part_of_day=part_of_day,
                                                      preference=preference, income=income,
                                                      regime=regimes, motive=motive_name,
                                                      fuel_kind=fuel_kind)
                                    combined_key = DataKey(
                                        f"{car_kind}_{pt_kind}_{modality_bike}_vk",
                                        part_of_day=part_of_day, income=income,
                                        regime=regimes, motive=motive_name,
                                        preference=preference, subtopic="combinaties",
                                        fuel_kind=fuel_kind, header=_hdr(), index=_idx())
                                    combined_weights.register(combined_key, [car_key, bike_key, pt_key])
                            else:
                                car_key = DataKey(f"{car_kind}_vk", part_of_day=part_of_day,
                                                  preference=preference, income=income,
                                                  regime=regimes, motive=motive_name)
                                combined_key = DataKey(
                                    f"{car_kind}_{pt_kind}_{modality_bike}_vk",
                                    part_of_day=part_of_day, income=income,
                                    regime=regimes, motive=motive_name,
                                    preference=preference, subtopic="combinaties",
                                    header=_hdr(), index=_idx())
                                combined_weights.register(combined_key, [car_key, bike_key, pt_key])

    logger.info("Registered %d combined-weight recipes (0 bytes allocated).",
                combined_weights.recipe_count())
    return combined_weights