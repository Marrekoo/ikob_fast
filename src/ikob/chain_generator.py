import logging

import numpy as np
import numpy.typing as npt

from ikob.configuration_definition import TvomType
from ikob.datasource import DataKey, DataSource, SkimsSource, read_csv_from_config

logger = logging.getLogger(__name__)


class Hubs:
    def __init__(self, hubs: npt.NDArray):
        self.zones = hubs[:, 0]
        self.hub_costs_cents = hubs[:, 1]
        self.pt_transfer_times = hubs[:, 2]
        self.bike_transfer_times = hubs[:, 3]
        self.pay_for_pt = hubs[:, 4]
        self.num_hubs: int = len(hubs)


def _compute_chain_travel_time(
    hubs: Hubs,
    car_time: npt.NDArray,
    car_dist: npt.NDArray,
    bike_time: npt.NDArray,
    bike_dist: npt.NDArray,
    pt_time: npt.NDArray,
    pt_dist: npt.NDArray,
    factor: float,
    var_car_rate: float,
    road_pricing: float,
    pt_km_price: float,
    bike_cost_euro_per_km: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute Park+Bike and Park+Ride generalized travel time skims.

    For each hub, vectorized over all origin-destination pairs. Returns the
    element-wise minimum across all hubs.

    """
    num_zones = len(car_time)
    # Initialize with a high value
    result_bike = np.full((num_zones, num_zones), np.inf)
    result_ride = np.full((num_zones, num_zones), np.inf)

    for hub_idx in range(hubs.num_hubs):
        # ASSUMPTION! Zones are zero indexed. In earlier discussions and old code both zero and one based indexing has been used. We should discuss.
        zone_idx = int(hubs.zones[hub_idx])
        hub_cost = hubs.hub_costs_cents[hub_idx] / 100
        change_time_bike = hubs.bike_transfer_times[hub_idx]
        change_time_pt = hubs.pt_transfer_times[hub_idx]
        pay_for_pt = hubs.pay_for_pt[hub_idx]

        # Car leg: origin -> hub (shape: n)
        car_leg = car_time[:, zone_idx] + factor * (var_car_rate + road_pricing) * car_dist[:, zone_idx]

        # P+Bike: car_leg + bike from hub to destination + bike variable cost + transfer + hub cost
        p_bike = (
            car_leg[:, np.newaxis]
            + bike_time[zone_idx, :][np.newaxis, :]
            + change_time_bike
            + factor * bike_cost_euro_per_km * bike_dist[zone_idx, :][np.newaxis, :]
            + factor * hub_cost
        )

        # P+R: car_leg + PT from hub to destination + transfer + PT cost + hub cost
        p_ride = (
            car_leg[:, np.newaxis]
            + pt_time[zone_idx, :][np.newaxis, :]
            + change_time_pt
            + factor * pt_dist[zone_idx, :][np.newaxis, :] * pt_km_price * pay_for_pt
            + factor * hub_cost
        )

        result_bike = np.minimum(result_bike, p_bike)
        result_ride = np.minimum(result_ride, p_ride)

    if np.any(result_bike == np.inf) and hubs.num_hubs != 0:
        raise ValueError(
            f"A value in the park and bike travel time matrix is still infinite after considering travel via all {hubs.num_hubs} hubs."
        )
    if np.any(result_ride == np.inf) and hubs.num_hubs != 0:
        raise ValueError(
            f"A value in the park and ride travel time matrix is still infinite after considering travel via all {hubs.num_hubs} hubs."
        )

    return result_bike, result_ride


def chain_generator(generalized_travel_time: DataSource, config: dict):
    """Generate generalized travel time skims for chains (P+R and P+Bike).

    For each origin, computes the cost of driving to each hub and then
    continuing by bike or public transport. The result for each OD pair is
    the minimum across all hubs:

        P+Bike(i,j) = min_h( car(i,h) + bike(h,j) + transfer + hub_cost )
        P+R(i,j)    = min_h( car(i,h) + PT(h,j) + transfer + PT_cost + hub_cost )

    Results are stored in ``generalized_travel_time`` so they can be picked
    up by the main generalized travel time computation.
    """
    logger.info("Starting step: Compute chain (P+R / P+Bike) generalized travel times.")

    project_config = config["project"]
    skims_config = config["skims"]
    tvom_config = config["TVOM"]
    ketens_config = config["ketens"]

    regime = project_config["beprijzingsregime"]
    motive_name = project_config["motief"]["naam"]
    motive_tvom = project_config["motief"]["TVOM"]
    hub_name = ketens_config["chains"]["naam hub"]
    tvom = tvom_config[TvomType.WORK] if motive_tvom == TvomType.WORK else tvom_config[TvomType.OTHER]

    var_fossil = skims_config["Kosten auto fossiele brandstof"]["variabele kosten"] / 100
    road_pricing_fossil = skims_config["Kosten auto fossiele brandstof"]["kmheffing"] / 100
    var_electric = skims_config["Kosten elektrische auto"]["variabele kosten"] / 100
    road_pricing_electric = skims_config["Kosten elektrische auto"]["kmheffing"] / 100
    pt_km_price = skims_config["OV kosten"]["kmkosten"] / 100
    bike_cost_euro_per_km = skims_config["bike_cost_ct_per_km"] / 100
    part_of_day = skims_config["dagsoort"]

    income_levels = ["laag", "middellaag", "middelhoog", "hoog"]
    fuel_kinds = ["fossiel", "elektrisch"]

    hubs = Hubs(read_csv_from_config(config, key="ketens", id="chains"))

    skims_dir = config["project"]["paden"]["skims_directory"]
    skims_reader = SkimsSource(skims_dir)

    for pod in part_of_day:
        car_time = skims_reader.read("Auto_Tijd", pod)
        car_dist = skims_reader.read("Auto_Afstand", pod)
        bike_time = skims_reader.read("Fiets_Tijd", pod)
        default_speed_km_p_minute = 15 / 60
        bike_dist = skims_reader.read("Fiets_Afstand", pod, default=(bike_time * default_speed_km_p_minute))
        pt_time = skims_reader.read("OV_Tijd", pod)
        pt_dist = skims_reader.read("OV_Afstand", pod)

        for income_level in income_levels:
            factor = tvom.get(income_level)

            for fuel_kind in fuel_kinds:
                if fuel_kind == "fossiel":
                    var_car_rate = var_fossil
                    road_pricing = road_pricing_fossil
                else:
                    var_car_rate = var_electric
                    road_pricing = road_pricing_electric

                result_bike, result_ride = _compute_chain_travel_time(
                    hubs=hubs,
                    car_time=car_time,
                    car_dist=car_dist,
                    bike_time=bike_time,
                    bike_dist=bike_dist,
                    pt_time=pt_time,
                    pt_dist=pt_dist,
                    factor=factor,
                    var_car_rate=var_car_rate,
                    road_pricing=road_pricing,
                    pt_km_price=pt_km_price,
                    bike_cost_euro_per_km=bike_cost_euro_per_km,
                )

                key = DataKey(
                    id=f"Pplusfiets_{fuel_kind}",
                    part_of_day=pod,
                    income=income_level,
                    hub_name=hub_name,
                    motive=motive_name,
                    regime=regime,
                )
                generalized_travel_time.set(key, result_bike)

                key = DataKey(
                    id=f"PplusR_{fuel_kind}",
                    part_of_day=pod,
                    income=income_level,
                    hub_name=hub_name,
                    motive=motive_name,
                    regime=regime,
                )
                generalized_travel_time.set(key, result_ride)
