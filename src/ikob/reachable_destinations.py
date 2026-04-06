import logging
from pathlib import Path

import numpy as np

import ikob.utils as utils
from ikob.competition import get_weight_matrix, _weight_cache_key
from ikob.datasource import DataKey, DataSource, DataType, SegsSource
from ikob.utils import DTYPE

logger = logging.getLogger(__name__)


def reachable_destinations(config, single_weights: DataSource, combined_weights) -> DataSource:
    """
    Section D4: from weights to reachable destinations per zone.

    ROUND-2 CHANGES:
    - Matmul results cached by weight-matrix identity → ~5× fewer O(N²) ops
    - Weight matrices cached per (pod, income, modality) iteration
    """
    logger.info("Starting step: Reachable destinations for citizens.")

    project_config = config["project"]
    skims_config = config["skims"]
    distribution_config = config["verdeling"]
    part_of_days = skims_config["dagsoort"]
    advanced_config = config["geavanceerd"]

    scenario = project_config["verstedelijkingsscenario"]
    regime = project_config["beprijzingsregime"]
    motive_name = project_config["motief"]["naam"]
    traveling_population_path = Path(project_config["motief"]["reizende populatie"])
    destinations_path = Path(project_config["motief"]["bestemmingsplaatsen"])
    car_possession_groups = advanced_config["welke_groepen"]
    income_groups = project_config["welke_inkomensgroepen"]
    electric_percentage = distribution_config["Percelektrisch"]

    base_groups = [
        "GratisAuto", "GratisAuto_GratisOV",
        "WelAuto_GratisOV", "WelAuto_vkAuto", "WelAuto_vkNeutraal", "WelAuto_vkFiets", "WelAuto_vkOV",
        "GeenAuto_GratisOV", "GeenAuto_vkNeutraal", "GeenAuto_vkFiets", "GeenAuto_vkOV",
        "GeenRijbewijs_GratisOV", "GeenRijbewijs_vkNeutraal", "GeenRijbewijs_vkFiets", "GeenRijbewijs_vkOV",
    ]
    groups = [f"{bg}_{ig}" for ig in income_groups for bg in base_groups]

    modalities = ["Fiets", "Auto", "OV", "Auto_Fiets", "OV_Fiets", "Auto_OV", "Auto_OV_Fiets"]
    headstring = list(modalities)

    segs_source = SegsSource(config)

    traveling_population = np.asarray(segs_source.read(traveling_population_path.name, scenario=scenario), dtype=DTYPE)
    destinations_segs = np.asarray(segs_source.read(destinations_path.name, scenario=scenario), dtype=DTYPE)
    destinations = destinations_segs.T

    num_zones = len(destinations_segs)

    traveling_population_totals = traveling_population.sum(axis=1, keepdims=True)
    safe_totals = np.where(traveling_population_totals > 0, traveling_population_totals, 1.0)
    income_distributions = traveling_population / safe_totals
    income_distributions[traveling_population_totals.ravel() <= 0] = 0.0
    income_distributions_transposed = income_distributions.T

    potencies = DataSource(config, DataType.DESTINATIONS)

    for car_possession_group in car_possession_groups:
        distribution_matrix = np.asarray(segs_source.read(
            "Verdeling_over_groepen", type_caster=float, scenario=scenario, group=motive_name,
            modifier="alleen_autobezit" if car_possession_group == "alleen autobezit" else "",
            has_index_column=True,
        ), dtype=DTYPE)
        distribution_matrix_transpose = distribution_matrix.T

        for part_of_day in part_of_days:
            for i_income_group, income_group in enumerate(income_groups):
                destinations_for_income = destinations[i_income_group]
                incomes = income_distributions_transposed[i_income_group]
                safe_incomes = np.where(incomes > 0, incomes, 1.0)
                general_possibility_totals = []

                K = electric_percentage.get(income_group) / 100

                for modality in modalities:
                    possibility_sum = np.zeros(num_zones, dtype=DTYPE)

                    # ── Matmul + matrix caches for this modality ──
                    _matrix_cache = {}
                    _matmul_cache = {}

                    for i_group, group in enumerate(groups):
                        distribution = distribution_matrix_transpose[i_group]
                        income = utils.group_income_level(group)
                        if income_group != income and income_group != "alle":
                            continue

                        wck = _weight_cache_key(group, modality)

                        if wck not in _matmul_cache:
                            matrix = get_weight_matrix(
                                single_weights, combined_weights, group, modality,
                                motive_name, regime, part_of_day, income, K,
                                _matrix_cache=_matrix_cache,
                            )
                            _matmul_cache[wck] = matrix @ destinations_for_income

                        possibility = _matmul_cache[wck] * distribution
                        possibility = possibility / safe_incomes
                        possibility[incomes <= 0] = 0
                        possibility_sum += possibility

                    key = DataKey("Totaal", part_of_day=part_of_day, income=income_group,
                                  group=car_possession_group, motive=motive_name, modality=modality, is_temporary=True)
                    potencies.set(key, possibility_sum)
                    general_possibility_totals.append(possibility_sum)

                gpt_arr = np.round(np.column_stack(general_possibility_totals)).astype(int)
                key = DataKey("Ontpl_totaal", part_of_day=part_of_day, group=car_possession_group,
                              income=income_group, motive=motive_name, index=DataKey.zone_index(num_zones))
                potencies.write_csv(gpt_arr, key, header=headstring)

            header = ["laag", "middellaag", "middelhoog", "hoog"]
            for modality in modalities:
                general_matrix = []
                for income_group in income_groups:
                    key = DataKey("Totaal", part_of_day=part_of_day, income=income_group,
                                  group=car_possession_group, motive=motive_name, modality=modality)
                    general_matrix.append(potencies.get(key))

                if len(income_groups) > 1:
                    general_possibility_totals_transposed = np.column_stack(general_matrix)
                else:
                    general_possibility_totals_transposed = np.asarray(general_matrix)

                general_matrix_product = np.where(
                    traveling_population > 0,
                    general_possibility_totals_transposed * traveling_population,
                    0,
                )

                gpt_int = np.round(general_possibility_totals_transposed).astype(int)
                key = DataKey("Ontpl_totaal", part_of_day=part_of_day, group=car_possession_group,
                              motive=motive_name, modality=modality, index=DataKey.zone_index(num_zones))
                potencies.write_csv(gpt_int, key, header=header)

                gmp_int = np.round(general_matrix_product).astype(int)
                key = DataKey("Ontpl_totaalproduct", part_of_day=part_of_day, group=car_possession_group,
                              motive=motive_name, modality=modality, index=DataKey.zone_index(num_zones))
                potencies.write_csv(gmp_int, key, header=header)

    return potencies