import logging
from pathlib import Path

import numpy as np

import ikob.utils as utils
from ikob.datasource import DataKey, DataSource, DataType, SegsSource
from ikob.utils import DTYPE

logger = logging.getLogger(__name__)


def compute_income_distributions(citizens_or_destinations):
    """Vectorised income distribution."""
    arr = np.asarray(citizens_or_destinations, dtype=DTYPE)
    totals = arr.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals > 0, totals, 1.0)
    result = arr / safe_totals
    result[totals.ravel() <= 0] = 0.0
    return result


def _weight_cache_key(group, modality):
    """Deterministic key that uniquely identifies a weight matrix
    within a fixed (part_of_day, income, regime, motive, K) context."""
    preference = utils.find_preference(group, modality)
    if modality in ("Fiets", "EFiets"):
        return (modality, "Fiets" if preference == "Fiets" else "")
    sg = utils.single_group(modality, group) if modality in ("Auto", "OV") else None
    cg = utils.combined_group(modality, group)
    if modality in ("Auto", "OV"):
        return (modality, preference, sg)
    return (modality, preference, cg)


def get_weight_matrix(
    single_weights, combined_weights,
    group, modality, motive, regime, part_of_day, income,
    ratio_electric: float,
    *,
    _matrix_cache: dict | None = None,
):
    """Fetch the weight matrix for a (group, modality) combination.

    If ``_matrix_cache`` is provided, the matrix is cached by a
    deterministic key to avoid redundant lookups/blending.
    """
    cache_key = _weight_cache_key(group, modality)

    if _matrix_cache is not None and cache_key in _matrix_cache:
        return _matrix_cache[cache_key]

    preference = utils.find_preference(group, modality)

    if modality in ("Fiets", "EFiets"):
        preference_bike = "Fiets" if preference == "Fiets" else ""
        key = DataKey(f"{modality}_vk", part_of_day=part_of_day, regime=regime, motive=motive,
                      preference=preference_bike, income=income)
        matrix = single_weights.get(key)

    else:
        sg = utils.single_group(modality, group)
        cg = utils.combined_group(modality, group)

        if (modality == "Auto" and "WelAuto" in group) or (cg and cg[0] == "A"):
            subtopic = "" if modality == "Auto" else "combinaties"
            weights = single_weights if modality == "Auto" else combined_weights
            string = sg if modality == "Auto" else cg
            key_f = DataKey(f"{string}_vk", part_of_day=part_of_day, regime=regime, motive=motive,
                            preference=preference, income=income, subtopic=subtopic, fuel_kind="fossiel")
            key_e = DataKey(f"{string}_vk", part_of_day=part_of_day, regime=regime, motive=motive,
                            preference=preference, income=income, subtopic=subtopic, fuel_kind="elektrisch")
            matrix_fossil = weights.get(key_f)
            matrix_electric = weights.get(key_e)
            matrix = ratio_electric * matrix_electric + (1 - ratio_electric) * matrix_fossil

        elif modality in ("Auto", "OV"):
            key = DataKey(f"{sg}_vk", part_of_day=part_of_day, regime=regime, motive=motive,
                          preference=preference, income=income)
            matrix = single_weights.get(key)

        else:
            key = DataKey(f"{cg}_vk", part_of_day=part_of_day, regime=regime, motive=motive,
                          preference=preference, income=income, subtopic="combinaties")
            matrix = combined_weights.get(key)

    if _matrix_cache is not None:
        _matrix_cache[cache_key] = matrix
    return matrix


# ── Legacy step functions (kept for selective-skip fallback) ─────────

def competition_on_destinations(config, single_weights, combined_weights, origins):
    """Section D6."""
    logger.info("Starting step: Compute competition on destinations")
    return competition(config, single_weights, combined_weights, origins, citizens=False)


def competition_on_citizens(config, single_weights, combined_weights, origins):
    """Section D7."""
    logger.info("Starting step: Compute competition on citizens")
    return competition(config, single_weights, combined_weights, origins, citizens=True)


def competition(config, single_weights, combined_weights, origins, citizens=True):
    if citizens:
        logger.info("Competition for citizens.")
    else:
        logger.info("Competition for destinations.")

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
    electric_percentage = distribution_config["Percelektrisch"]

    groups = [
        "GratisAuto_laag", "GratisAuto_GratisOV_laag",
        "WelAuto_GratisOV_laag", "WelAuto_vkAuto_laag", "WelAuto_vkNeutraal_laag",
        "WelAuto_vkFiets_laag", "WelAuto_vkOV_laag",
        "GeenAuto_GratisOV_laag", "GeenAuto_vkNeutraal_laag", "GeenAuto_vkFiets_laag", "GeenAuto_vkOV_laag",
        "GeenRijbewijs_GratisOV_laag", "GeenRijbewijs_vkNeutraal_laag",
        "GeenRijbewijs_vkFiets_laag", "GeenRijbewijs_vkOV_laag",
        "GratisAuto_middellaag", "GratisAuto_GratisOV_middellaag",
        "WelAuto_GratisOV_middellaag", "WelAuto_vkAuto_middellaag", "WelAuto_vkNeutraal_middellaag",
        "WelAuto_vkFiets_middellaag", "WelAuto_vkOV_middellaag",
        "GeenAuto_GratisOV_middellaag", "GeenAuto_vkNeutraal_middellaag",
        "GeenAuto_vkFiets_middellaag", "GeenAuto_vkOV_middellaag",
        "GeenRijbewijs_GratisOV_middellaag", "GeenRijbewijs_vkNeutraal_middellaag",
        "GeenRijbewijs_vkFiets_middellaag", "GeenRijbewijs_vkOV_middellaag",
        "GratisAuto_middelhoog", "GratisAuto_GratisOV_middelhoog",
        "WelAuto_GratisOV_middelhoog", "WelAuto_vkAuto_middelhoog", "WelAuto_vkNeutraal_middelhoog",
        "WelAuto_vkFiets_middelhoog", "WelAuto_vkOV_middelhoog",
        "GeenAuto_GratisOV_middelhoog", "GeenAuto_vkNeutraal_middelhoog",
        "GeenAuto_vkFiets_middelhoog", "GeenAuto_vkOV_middelhoog",
        "GeenRijbewijs_GratisOV_middelhoog", "GeenRijbewijs_vkNeutraal_middelhoog",
        "GeenRijbewijs_vkFiets_middelhoog", "GeenRijbewijs_vkOV_middelhoog",
        "GratisAuto_hoog", "GratisAuto_GratisOV_hoog",
        "WelAuto_GratisOV_hoog", "WelAuto_vkAuto_hoog", "WelAuto_vkNeutraal_hoog",
        "WelAuto_vkFiets_hoog", "WelAuto_vkOV_hoog",
        "GeenAuto_GratisOV_hoog", "GeenAuto_vkNeutraal_hoog",
        "GeenAuto_vkFiets_hoog", "GeenAuto_vkOV_hoog",
        "GeenRijbewijs_GratisOV_hoog", "GeenRijbewijs_vkNeutraal_hoog",
        "GeenRijbewijs_vkFiets_hoog", "GeenRijbewijs_vkOV_hoog",
    ]

    modalities = ["Fiets", "Auto", "OV", "Auto_Fiets", "OV_Fiets", "Auto_OV", "Auto_OV_Fiets"]
    income_groups = ["laag", "middellaag", "middelhoog", "hoog"]
    headstring = list(modalities)

    segs_source = SegsSource(config)

    traveling_population = np.asarray(
        segs_source.read(traveling_population_path.name, scenario=scenario), dtype=DTYPE
    )
    destinations = np.asarray(
        segs_source.read(destinations_path.name, scenario=scenario), dtype=DTYPE
    )

    income_distributions = compute_income_distributions(traveling_population if citizens else destinations)
    subtopic_competition = "inwoners" if citizens else "bestemmingen"
    competition_filename_suffix = "Pot" if citizens else "Ontpl"
    competitions = DataSource(config, DataType.COMPETITION)

    citizens_or_destinations = traveling_population if citizens else destinations
    num_zones = len(citizens_or_destinations)

    for car_possession_group in car_possession_groups:
        distribution_matrix = np.asarray(segs_source.read(
            "Verdeling_over_groepen", type_caster=float, scenario=scenario, group=motive_name,
            modifier="alleen_autobezit" if car_possession_group == "alleen autobezit" else "",
            has_index_column=True,
        ), dtype=DTYPE)

        for part_of_day in part_of_days:
            for i_income_group, income_group in enumerate(income_groups):
                general_possibility_totals = []

                for modality in modalities:
                    key = DataKey("Totaal", part_of_day=part_of_day, motive=motive_name, modality=modality,
                                  income=income_group, group=car_possession_group)
                    reach = origins.get(key)
                    safe_reach = np.where(reach > 0, reach, 1.0)

                    income_distribution = income_distributions[:, i_income_group]
                    safe_income_dist = np.where(income_distribution > 0, income_distribution, 1.0)

                    rhs = citizens_or_destinations.T[i_income_group] / safe_reach

                    K = electric_percentage.get(income_group) / 100

                    _matrix_cache = {}
                    _matmul_cache = {}
                    competition_total = np.zeros(num_zones, dtype=DTYPE)

                    for i_group, group in enumerate(groups):
                        distribution = distribution_matrix[:, i_group]
                        income = utils.group_income_level(group)
                        if income_group != income and income_group != "alle":
                            continue

                        wck = _weight_cache_key(group, modality)

                        if wck not in _matmul_cache:
                            matrix = get_weight_matrix(
                                single_weights, combined_weights, group, modality,
                                motive_name, regimes, part_of_day, income, K,
                                _matrix_cache=_matrix_cache,
                            )
                            _matmul_cache[wck] = matrix @ rhs

                        competition_val = _matmul_cache[wck]
                        competition_total += competition_val * distribution / safe_income_dist

                    key = DataKey(
                        id="Totaal", part_of_day=part_of_day, subtopic=subtopic_competition,
                        income=income_group, motive=motive_name, modality=modality,
                        group=car_possession_group, is_temporary=True,
                    )
                    competitions.set(key, competition_total)

                    general_possibility_totals.append(competition_total)
                    general_totals_transpose = np.column_stack(general_possibility_totals)
                    key = DataKey(
                        id=f"{competition_filename_suffix}_conc", part_of_day=part_of_day,
                        subtopic=subtopic_competition, income=income_group, motive=motive_name,
                        group=car_possession_group, index=DataKey.zone_index(num_zones),
                    )
                    competitions.write_csv(general_totals_transpose, key, header=headstring)

            header = ["laag", "middellaag", "middelhoog", "hoog"]
            for modality in modalities:
                general_matrix = []
                for income_group in income_groups:
                    key = DataKey("Totaal", part_of_day=part_of_day, motive=motive_name, modality=modality,
                                  income=income_group, subtopic=subtopic_competition, group=car_possession_group)
                    general_matrix.append(competitions.get(key))
                general_totals_transpose = np.column_stack(general_matrix)

                if citizens:
                    mask = destinations > 0
                else:
                    mask = traveling_population > 0
                general_matrix_product = np.where(
                    mask,
                    general_totals_transpose * citizens_or_destinations,
                    0,
                )

                key = DataKey(
                    id=f"{competition_filename_suffix}_conc", part_of_day=part_of_day,
                    subtopic=subtopic_competition, motive=motive_name, modality=modality,
                    group=car_possession_group, index=DataKey.zone_index(num_zones),
                )
                competitions.write_csv(general_totals_transpose, key, header=header)

                key = DataKey(
                    id=f"{competition_filename_suffix}_concproduct", part_of_day=part_of_day,
                    subtopic=subtopic_competition, motive=motive_name, modality=modality,
                    group=car_possession_group, index=DataKey.zone_index(num_zones),
                )
                competitions.write_csv(general_matrix_product, key, header=header)

    return competitions