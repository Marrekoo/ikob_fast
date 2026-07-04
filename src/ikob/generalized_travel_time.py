import logging
from dataclasses import replace

import numpy as np

import ikob.utils as utils
from ikob.chain_generator import chain_generator
from ikob.configuration_definition import TvomType
from ikob.datasource import (
    DataKey,
    DataSource,
    DataType,
    SegsSource,
    read_csv_from_config,
    read_parking_times,
)
from ikob.skims_provider import get_skims_provider
from ikob.tolerance_curves import CurveRegistry
from ikob.utils import DTYPE, IKOB_INFINITE

logger = logging.getLogger(__name__)


def _maybe_store_time_money(gtt_source: DataSource, curve_registry, base_key: DataKey, t, m):
    """Also cache the raw (time, money) components behind *base_key*,
    tagged via DataKey.subtopic ('tijd' / 'geld'), so a curve-library
    attachment (ikob.curve_attachment) can evaluate a custom -- possibly
    'copula' or custom-tau 'fixedVOT' -- tolerance curve on them in D2.

    Skipped entirely when no curve_registry is supplied, so a default run
    without a curve library has the exact same memory/CPU footprint as
    before this feature was added.
    """
    if curve_registry is None or len(curve_registry) == 0:
        return
    gtt_source.set(replace(base_key, subtopic="tijd", is_temporary=True), np.asarray(t, dtype=DTYPE))
    gtt_source.set(replace(base_key, subtopic="geld", is_temporary=True), np.asarray(m, dtype=DTYPE))


def generalized_travel_time(config, curve_registry: CurveRegistry | None = None) -> DataSource:
    """
    Compute generalized (experienced) travel time from time and costs.
    Corresponds to section D1 in IKOB-algorithm.pdf

    *curve_registry*, if given, additionally stores the un-collapsed
    (time, money) components for every skim below (except the
    chains-adjusted car skims, see the note there) so that D2
    (single_weights.py) can evaluate a curve-library attachment on them
    -- see ikob.curve_attachment.

    Skims themselves come from config["skims"]["skims_bron"] -- either
    pre-computed CSV files, or generated on the fly via r5py/OSRM -- see
    ikob.skims_provider.get_skims_provider().

    KEY CHANGES vs original:
    - "geen auto" double for-loop fully vectorised
    - "gratis auto" double for-loop fully vectorised
    - All skim matrices cast to float32 on load
    - .copy() removed where new arrays are created by vectorised ops
    - optional (time, money) component storage for curve-library support
    - pluggable skims source (files / r5py / OSRM)
    """
    logger.info("Starting step: Compute generalized travel time from time and costs.")

    project_config = config["project"]
    skims_config = config["skims"]
    tvom_config = config["TVOM"]
    advanced_config = config["geavanceerd"]
    ketens_config = config["ketens"]

    regime = project_config["beprijzingsregime"]
    motive_name = project_config["motief"]["naam"]
    motive_tvom = project_config["motief"]["TVOM"]
    chains = ketens_config["chains"]["gebruiken"]
    ketens_config["bestemmingslijst"]["gebruiken"]
    hub_name = ketens_config["chains"]["naam hub"]
    pt_cost_file = skims_config["OV kostenbestand"]["gebruiken"]
    tvom_dict = tvom_config[TvomType.WORK] if motive_tvom == TvomType.WORK else tvom_config[TvomType.OTHER]
    var_fossil = skims_config["Kosten auto fossiele brandstof"]["variabele kosten"] / 100
    road_pricing_fossil = skims_config["Kosten auto fossiele brandstof"]["kmheffing"] / 100
    var_electric = skims_config["Kosten elektrische auto"]["variabele kosten"] / 100
    road_pricing_electric = skims_config["Kosten elektrische auto"]["kmheffing"] / 100
    costs_no_car = skims_config["varkostenga"]
    time_costs_no_car = skims_config["tijdkostenga"]
    part_of_day = skims_config["dagsoort"]
    kind_no_car = ["GeenAuto", "GeenRijbewijs"]
    pt_km_price = skims_config["OV kosten"]["kmkosten"] / 100
    starting_rate = skims_config["OV kosten"]["starttarief"] / 100
    additional_costs = advanced_config["additionele_kosten"]["gebruiken"]
    parking_costs = advanced_config["parkeerkosten"]["gebruiken"]
    pricecap = skims_config["pricecap"]["gebruiken"]
    pricecap_value = skims_config["pricecap"]["getal"]
    bike_cost_euro_per_km = skims_config["bike_cost_ct_per_km"] / 100
    parking_times = read_parking_times(config)

    # Convert once to float32 array for vectorised ops
    parking_times_arr = np.asarray(parking_times, dtype=DTYPE)

    if parking_costs:
        parking_cost_array = np.asarray(
            read_csv_from_config(config, key="geavanceerd", id="parkeerkosten"), dtype=DTYPE
        )
    else:
        parking_cost_array = np.zeros(len(parking_times), dtype=DTYPE)

    if additional_costs:
        additional_cost_matrix = np.asarray(
            read_csv_from_config(config, key="geavanceerd", id="additionele_kosten"), dtype=DTYPE
        )
    else:
        additional_cost_matrix = np.zeros((len(parking_times), len(parking_times)), dtype=DTYPE)

    income_levels = ["laag", "middellaag", "middelhoog", "hoog"]
    fuel_kinds = ["fossiel", "elektrisch"]

    SegsSource(config)

    skims_reader = get_skims_provider(config)

    gtt_source = DataSource(config, DataType.GENERALIZED_TRAVEL_TIME)

    if chains:
        chain_generator(gtt_source, config)

    # Pre-compute the parking-time outer-sum (origin departure + destination arrival)
    parking_time_matrix = (
        parking_times_arr[:, 0][:, np.newaxis] + parking_times_arr[:, 1][np.newaxis, :]
    )

    for pod in part_of_day:
        car_time_matrix = np.asarray(skims_reader.read("Auto_Tijd", pod), dtype=DTYPE)
        car_distance_matrix = np.asarray(skims_reader.read("Auto_Afstand", pod), dtype=DTYPE)
        bike_time_matrix = np.asarray(skims_reader.read("Fiets_Tijd", pod), dtype=DTYPE)
        default_speed_km_p_minute = 15.0 / 60.0
        bike_distance_matrix = np.asarray(
            skims_reader.read("Fiets_Afstand", pod, default=(bike_time_matrix * default_speed_km_p_minute)),
            dtype=DTYPE,
        )
        pt_time_matrix = np.asarray(skims_reader.read("OV_Tijd", pod), dtype=DTYPE)

        num_zones = len(pt_time_matrix)

        if pt_cost_file:
            pt_cost_matrix = np.asarray(skims_reader.read("OV_Kosten", pod), dtype=DTYPE)
        else:
            pt_distance_matrix = np.asarray(skims_reader.read("OV_Afstand", pod), dtype=DTYPE)
            pt_cost_matrix = np.asarray(
                utils.costs_public_transport(pt_distance_matrix, pt_km_price, starting_rate, pricecap, pricecap_value),
                dtype=DTYPE,
            )

        for income_level in income_levels:
            tvom_factor = tvom_dict.get(income_level)

            # ── Bike GTT (already vectorised) ──
            gtr_skim = utils.compute_bike_gtt(bike_time_matrix, bike_distance_matrix, bike_cost_euro_per_km, tvom_factor)
            key = DataKey(
                id="Fiets", part_of_day=pod, regime=regime, motive=motive_name, income=income_level,
                header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
            )
            gtt_source.set(key, gtr_skim)
            t_bike, m_bike = utils.compute_bike_time_money(bike_time_matrix, bike_distance_matrix, bike_cost_euro_per_km)
            _maybe_store_time_money(gtt_source, curve_registry, key, t_bike, m_bike)

            # ── Car GTT (already vectorised via compute_car_gtt) ──
            for fuel_kind in fuel_kinds:
                if fuel_kind == "fossiel":
                    var_car_rate = var_fossil
                    road_pricing = road_pricing_fossil
                else:
                    var_car_rate = var_electric
                    road_pricing = road_pricing_electric

                gtr_skim = utils.compute_car_gtt(
                    car_time=car_time_matrix,
                    car_dist=car_distance_matrix,
                    var_rate=var_car_rate,
                    road_pricing=road_pricing,
                    tvom_factor=tvom_factor,
                    additional_costs_eurocent=additional_cost_matrix,
                    parking_times_array=parking_times_arr,
                    parking_costs_array_eurocent=parking_cost_array,
                )
                t_car, m_car = utils.compute_car_time_money(
                    car_time_matrix, car_distance_matrix, var_car_rate, road_pricing,
                    additional_cost_matrix, parking_times_arr, parking_cost_array,
                )

                if chains:
                    key = DataKey(
                        id=f"Pplusfiets_{fuel_kind}", part_of_day=pod, income=income_level,
                        hub_name=hub_name, motive=motive_name, regime=regime,
                    )
                    gtr_park_and_bike_skim = gtt_source.get(key)
                    bestskim = np.minimum(gtr_skim, gtr_park_and_bike_skim)
                    key = DataKey(
                        id=f"PplusR_{fuel_kind}", part_of_day=pod, income=income_level,
                        hub_name=hub_name, motive=motive_name, regime=regime,
                    )
                    gtr_park_and_ride_skim = gtt_source.get(key)
                    gtr_skim = np.minimum(bestskim, gtr_park_and_ride_skim)
                    if curve_registry is not None and len(curve_registry) > 0:
                        logger.warning(
                            "Chains (P+R/P+Bike) are enabled and a curve-library was "
                            "supplied: the decomposed time/money skim used for custom "
                            "curves ignores hub alternatives (direct-drive only)."
                        )

                key = DataKey(
                    id=f"Auto_{fuel_kind}", part_of_day=pod, income=income_level, regime=regime, motive=motive_name,
                    header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
                )
                gtt_source.set(key, gtr_skim)
                _maybe_store_time_money(gtt_source, curve_registry, key, t_car, m_car)

            # ── PT GTT (already vectorised) ──
            gtr_skim = utils.compute_pt_gtt(pt_time_matrix, pt_cost_matrix, tvom_factor)
            key = DataKey(
                id="OV", part_of_day=pod, income=income_level, motive=motive_name, regime=regime,
                header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
            )
            gtt_source.set(key, gtr_skim)
            t_pt, m_pt = utils.compute_pt_time_money(pt_time_matrix, pt_cost_matrix)
            _maybe_store_time_money(gtt_source, curve_registry, key, t_pt, m_pt)

            # ── Geen auto: VECTORISED (was double for-loop) ──
            for kind in kind_no_car:
                time_cost_factor = time_costs_no_car.get(kind)
                var_cost_factor = costs_no_car.get(kind) + road_pricing_electric
                total_cost = car_time_matrix * time_cost_factor + car_distance_matrix * var_cost_factor
                gtr_skim = (car_time_matrix + tvom_factor * total_cost).astype(DTYPE)

                key = DataKey(
                    id=f"{kind}", part_of_day=pod, income=income_level, motive=motive_name, regime=regime,
                    header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
                )
                gtt_source.set(key, gtr_skim)
                t_nc, m_nc = utils.compute_no_car_time_money(
                    car_time_matrix, car_distance_matrix, time_cost_factor, var_cost_factor
                )
                _maybe_store_time_money(gtt_source, curve_registry, key, t_nc, m_nc)

            # ── Gratis auto: VECTORISED (was double for-loop) ──
            total_time = car_time_matrix + parking_time_matrix
            monetary = car_distance_matrix * road_pricing_electric + parking_cost_array[np.newaxis, :] / 100
            if additional_costs:
                monetary = monetary + additional_cost_matrix / 100
            gtr_skim = (total_time + tvom_factor * monetary).astype(DTYPE)

            key = DataKey(
                id="GratisAuto", part_of_day=pod, income=income_level, motive=motive_name, regime=regime,
                header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
            )
            gtt_source.set(key, gtr_skim)
            t_free, m_free = utils.compute_free_car_time_money(
                car_time_matrix, car_distance_matrix, road_pricing_electric,
                parking_time_matrix, parking_cost_array,
                additional_cost_matrix if additional_costs else None,
            )
            _maybe_store_time_money(gtt_source, curve_registry, key, t_free, m_free)

            # ── Gratis OV (already vectorised) ──
            gtr_skim = np.where(pt_time_matrix > 0.5, pt_time_matrix, IKOB_INFINITE).astype(DTYPE)
            key = DataKey(
                id="GratisOV", part_of_day=pod, motive=motive_name, regime=regime,
                header=DataKey.zone_header(num_zones), index=DataKey.zone_index(num_zones),
            )
            gtt_source.set(key, gtr_skim)
            t_free_ov = np.where(pt_time_matrix > 0.5, pt_time_matrix, IKOB_INFINITE)
            m_free_ov = np.zeros_like(t_free_ov)  # free PT: no money component at all
            _maybe_store_time_money(gtt_source, curve_registry, key, t_free_ov, m_free_ov)

    return gtt_source