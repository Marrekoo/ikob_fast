import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

import ikob.utils as utils
from ikob.competition import _weight_cache_key, get_weight_matrix
from ikob.datasource import DataKey, DataSource, DataType, SegsSource
from ikob.utils import DTYPE

logger = logging.getLogger(__name__)


def create_citizens_file(distribution_matrix, working_population):
    """Vectorised: was a double Python loop."""
    return np.asarray(distribution_matrix, dtype=DTYPE) * np.asarray(working_population, dtype=DTYPE)[:, np.newaxis]


def reachable_population(config, single_weights: DataSource, combined_weights) -> DataSource:
    """
    Section D5: reachable population for destinations.

    ROUND-2 CHANGES:
    - Groups sharing the same weight matrix are batched: their citizen
      vectors are summed first, then a single matmul is performed.
      Reduces O(N²) matmuls from ~60 to ~12 per modality.
    """
    logger.info("Starting step: Reachable population for destinations.")

    project_config = config["project"]
    skims_config = config["skims"]
    distribution_config = config["verdeling"]
    part_of_days = skims_config["dagsoort"]
    advanced_config = config["geavanceerd"]

    scenario = project_config["verstedelijkingsscenario"]
    regimes = project_config["beprijzingsregime"]
    motive_name = project_config["motief"]["naam"]
    traveling_population_path = Path(project_config["motief"]["reizende populatie"])
    destinations_path = Path(project_config["motief"]["bestemmingsplaatsen"])
    car_possession_groups = advanced_config["welke_groepen"]
    income_groups = project_config["welke_inkomensgroepen"]
    fuel_kinds = ["fossiel", "elektrisch"]
    electric_percentage = distribution_config["Percelektrisch"]

    base_groups = [
        "GratisAuto", "GratisAuto_GratisOV",
        "WelAuto_GratisOV", "WelAuto_vkAuto", "WelAuto_vkNeutraal", "WelAuto_vkFiets", "WelAuto_vkOV",
        "GeenAuto_GratisOV", "GeenAuto_vkNeutraal", "GeenAuto_vkFiets", "GeenAuto_vkOV",
        "GeenRijbewijs_GratisOV", "GeenRijbewijs_vkNeutraal", "GeenRijbewijs_vkFiets", "GeenRijbewijs_vkOV",
    ]
    groups = [f"{bg}_{ig}" for ig in income_groups for bg in base_groups]

    modalities = ["Fiets", "Auto", "OV", "Auto_Fiets", "OV_Fiets", "Auto_OV", "Auto_OV_Fiets"]
    income_groups_out = ["laag", "middellaag", "middelhoog", "hoog"]
    headstring = modalities

    segs_source = SegsSource(config)

    traveling_population = np.asarray(
        segs_source.read(traveling_population_path.name, scenario=scenario), dtype=DTYPE
    )
    destinations = np.asarray(
        segs_source.read(destinations_path.name, scenario=scenario), dtype=DTYPE
    )

    num_zones = len(traveling_population)
    working_population = traveling_population.sum(axis=1)

    origins = DataSource(config, DataType.ORIGINS)

    for car_possession_group in car_possession_groups:
        distribution_matrix = np.asarray(segs_source.read(
            "Verdeling_over_groepen", type_caster=float, scenario=scenario, group=motive_name,
            modifier="alleen_autobezit" if car_possession_group == "alleen autobezit" else "",
            has_index_column=True,
        ), dtype=DTYPE)

        citizens = create_citizens_file(distribution_matrix, working_population)
        citizens_transpose = citizens.T  # shape (num_groups, num_zones)

        for part_of_day in part_of_days:
            for income_group in income_groups_out:
                general_possibility_totals = []
                for modality in modalities:
                    working_population_list = np.zeros(num_zones, dtype=DTYPE)

                    # ── Batch: accumulate citizen vectors by weight key ──
                    batches: dict[tuple, np.ndarray] = {}
                    batch_representative_group: dict[tuple, str] = {}

                    for igroup, group in enumerate(groups):
                        income = utils.group_income_level(group)
                        if income_group != income and income_group != "alle":
                            continue

                        wck = _weight_cache_key(group, modality)
                        citizen_vec = citizens_transpose[igroup]

                        if wck not in batches:
                            batches[wck] = np.zeros(num_zones, dtype=DTYPE)
                            batch_representative_group[wck] = group

                        # For fuel-blended modalities (WelAuto with Auto, or combined starting with A),
                        # we need to split into fossil/electric batches
                        if modality == "Fiets" or modality == "EFiets":
                            batches[wck] += citizen_vec
                        elif modality == "Auto":
                            if "WelAuto" in group:
                                K_e = electric_percentage.get(income_group) / 100
                                K_f = 1 - K_e
                                fk = wck + ("fossiel",)
                                ek = wck + ("elektrisch",)
                                if fk not in batches:
                                    batches[fk] = np.zeros(num_zones, dtype=DTYPE)
                                    batches[ek] = np.zeros(num_zones, dtype=DTYPE)
                                    batch_representative_group[fk] = group
                                    batch_representative_group[ek] = group
                                batches[fk] += K_f * citizen_vec
                                batches[ek] += K_e * citizen_vec
                                # remove the non-fuel key if it was just created
                                batches.pop(wck, None)
                                batch_representative_group.pop(wck, None)
                            else:
                                batches[wck] += citizen_vec
                        elif modality == "OV":
                            batches[wck] += citizen_vec
                        else:
                            # Combined modalities
                            cg = utils.combined_group(modality, group)
                            if cg and cg[0] == "A":
                                K_e = electric_percentage.get(income_group) / 100
                                K_f = 1 - K_e
                                fk = wck + ("fossiel",)
                                ek = wck + ("elektrisch",)
                                if fk not in batches:
                                    batches[fk] = np.zeros(num_zones, dtype=DTYPE)
                                    batches[ek] = np.zeros(num_zones, dtype=DTYPE)
                                    batch_representative_group[fk] = group
                                    batch_representative_group[ek] = group
                                batches[fk] += K_f * citizen_vec
                                batches[ek] += K_e * citizen_vec
                                batches.pop(wck, None)
                                batch_representative_group.pop(wck, None)
                            else:
                                batches[wck] += citizen_vec

                    # ── Execute batched matmuls ──
                    for bkey, total_citizens in batches.items():
                        rep_group = batch_representative_group[bkey]
                        income = utils.group_income_level(rep_group)

                        # Determine which weight matrix to fetch
                        if bkey[-1] in ("fossiel", "elektrisch"):
                            fuel_kind = bkey[-1]
                            base_wck = bkey[:-1]
                            # Fetch single fuel-kind matrix
                            preference = utils.find_preference(rep_group, modality)
                            if modality == "Auto":
                                sg = utils.single_group(modality, rep_group)
                                key = DataKey(f"{sg}_vk", part_of_day=part_of_day, preference=preference,
                                              income=income, regime=regimes, motive=motive_name, fuel_kind=fuel_kind)
                                matrix = single_weights.get(key)
                            else:
                                cg = utils.combined_group(modality, rep_group)
                                key = DataKey(f"{cg}_vk", part_of_day=part_of_day, preference=preference,
                                              income=income, regime=regimes, motive=motive_name,
                                              subtopic="combinaties", fuel_kind=fuel_kind)
                                matrix = combined_weights.get(key)
                        else:
                            K = electric_percentage.get(income_group) / 100
                            matrix = get_weight_matrix(
                                single_weights, combined_weights, rep_group, modality,
                                motive_name, regimes, part_of_day, income, K,
                            )

                        working_population_list += matrix.T @ total_citizens

                    key = DataKey(
                        id="Totaal", part_of_day=part_of_day, group=car_possession_group,
                        income=income_group, motive=motive_name, modality=modality, is_temporary=True,
                    )
                    origins.set(key, working_population_list)
                    general_possibility_totals.append(working_population_list)

                origins_total = np.round(np.column_stack(general_possibility_totals)).astype(int)
                key = DataKey(
                    id="Pot_totaal", part_of_day=part_of_day, group=car_possession_group,
                    income=income_group, motive=motive_name, index=DataKey.zone_index(num_zones),
                )
                origins.write_csv(origins_total, key, header=headstring)

            header = ["laag", "middellaag", "middelhoog", "hoog"]
            for modality in modalities:
                general_matrix = []
                for income_group in income_groups_out:
                    key = DataKey(
                        "Totaal", part_of_day=part_of_day, income=income_group, motive=motive_name,
                        group=car_possession_group, modality=modality, subtopic="",
                    )
                    general_matrix.append(origins.get(key))

                general_total_transpose = np.column_stack(general_matrix)

                general_matrix_product = np.where(
                    destinations > 0,
                    general_total_transpose * destinations,
                    0,
                )

                gtt_int = np.round(general_total_transpose).astype(int)
                key = DataKey(
                    id="Pot_totaal", part_of_day=part_of_day, group=car_possession_group,
                    motive=motive_name, modality=modality, index=DataKey.zone_index(num_zones),
                )
                origins.write_csv(gtt_int, key, header=header)

                key = DataKey(
                    id="Pot_totaalproduct", part_of_day=part_of_day, group=car_possession_group,
                    motive=motive_name, modality=modality, index=DataKey.zone_index(num_zones),
                )
                origins.write_csv(general_matrix_product, key, header=header)

    return origins