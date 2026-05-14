"""
Competition-adjusted accessibility — Shen (1998) two-step floating catchment.

Reference
---------
Shen, Q. (1998). "Location characteristics of inner-city neighborhoods and
employment accessibility of low-wage workers." Environment and Planning B:
Planning and Design, 25(3), 345-365.

Formulation (with IKOB's group structure)
-----------------------------------------
Let W^{g,m}_{ij} be the travel-time decay weight for group g under modality m,
O^{ig}_j the opportunities for income class ig at destination j, P^g_i =
s^g_i * workpop_i the absolute count of group g at origin i, and
inc^{ig}_i = P^{ig}_i / P_i the income share per zone.  Groups share a weight
matrix iff they share `_weight_cache_key(group, modality)`.

Pass 1 (naïve potentials, already computed in D4 and D5):
    U^{g,m}_i = sum_j  W^{g,m}_ij  ·  O^{ig(g)}_j
    V^{ig,m}_j = sum_{g∈ig} sum_i  W^{g,m}_ij  ·  P^g_i

Pass 2 (Shen competition-adjusted accessibility):

    Citizen-side A (this is D6, a.k.a. `competition_on_destinations`):
        A^{g,m}_i  = sum_j  W^{g,m}_ij  ·  O^{ig(g)}_j / V^{ig,m}_j
        A^{ig,m}_i = sum_{g∈ig}  s^{g|ig}_i  ·  A^{g,m}_i          (weighted AVG)

    Destination-side B (this is D7, a.k.a. `competition_on_citizens`):
        B^{g,m}_j  = sum_i  W^{g,m}_ij  ·  P^g_i / U^{g,m}_i
                   = ( (W^{g,m})^T @ (P^g / U^{g,m}) )_j
        B^{ig,m}_j = sum_{g∈ig}  B^{g,m}_j                         (plain SUM)

Note on fuel-kind blending: for Auto-based groups the weight matrix returned by
`get_weight_matrix` is already a fuel-share-blended W, so the Shen aggregation
happens on the blended matrix.  This matches how D4 computes U.
"""

import logging
from pathlib import Path

import numpy as np

import ikob.utils as utils
from ikob.datasource import DataKey, DataSource, DataType, SegsSource
from ikob.utils import DTYPE

logger = logging.getLogger(__name__)


# ── Group / modality definitions (shared with D4, D5) ────────────────

_BASE_GROUPS = [
    "GratisAuto", "GratisAuto_GratisOV",
    "WelAuto_GratisOV", "WelAuto_vkAuto", "WelAuto_vkNeutraal",
    "WelAuto_vkFiets", "WelAuto_vkOV",
    "GeenAuto_GratisOV", "GeenAuto_vkNeutraal",
    "GeenAuto_vkFiets", "GeenAuto_vkOV",
    "GeenRijbewijs_GratisOV", "GeenRijbewijs_vkNeutraal",
    "GeenRijbewijs_vkFiets", "GeenRijbewijs_vkOV",
]

_INCOME_LEVELS = ["laag", "middellaag", "middelhoog", "hoog"]

_MODALITIES = [
    "Fiets", "Auto", "OV", "Auto_Fiets",
    "OV_Fiets", "Auto_OV", "Auto_OV_Fiets",
]

_GROUPS = [f"{bg}_{ig}" for ig in _INCOME_LEVELS for bg in _BASE_GROUPS]


# ── Public helpers (kept for import compatibility with D4, D5) ───────

def compute_income_distributions(citizens_or_destinations):
    """Per-zone income shares: result[i, ig] = X^{ig}_i / X_i."""
    arr = np.asarray(citizens_or_destinations, dtype=DTYPE)
    totals = arr.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals > 0, totals, 1.0)
    result = arr / safe_totals
    result[totals.ravel() <= 0] = 0.0
    return result


def _weight_cache_key(group, modality):
    """Canonical key identifying a weight matrix within a fixed
    (part_of_day, income, regime, motive, K) context."""
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
    """Fetch the (possibly fuel-blended) weight matrix for (group, modality)."""
    cache_key = _weight_cache_key(group, modality)

    if _matrix_cache is not None and cache_key in _matrix_cache:
        return _matrix_cache[cache_key]

    preference = utils.find_preference(group, modality)

    if modality in ("Fiets", "EFiets"):
        preference_bike = "Fiets" if preference == "Fiets" else ""
        key = DataKey(f"{modality}_vk", part_of_day=part_of_day, regime=regime,
                      motive=motive, preference=preference_bike, income=income)
        matrix = single_weights.get(key)
    else:
        sg = utils.single_group(modality, group)
        cg = utils.combined_group(modality, group)

        if (modality == "Auto" and "WelAuto" in group) or (cg and cg[0] == "A"):
            subtopic = "" if modality == "Auto" else "combinaties"
            weights = single_weights if modality == "Auto" else combined_weights
            string = sg if modality == "Auto" else cg
            key_f = DataKey(f"{string}_vk", part_of_day=part_of_day, regime=regime,
                            motive=motive, preference=preference, income=income,
                            subtopic=subtopic, fuel_kind="fossiel")
            key_e = DataKey(f"{string}_vk", part_of_day=part_of_day, regime=regime,
                            motive=motive, preference=preference, income=income,
                            subtopic=subtopic, fuel_kind="elektrisch")
            m_fossil = weights.get(key_f)
            m_electric = weights.get(key_e)
            matrix = ratio_electric * m_electric + (1 - ratio_electric) * m_fossil
        elif modality in ("Auto", "OV"):
            key = DataKey(f"{sg}_vk", part_of_day=part_of_day, regime=regime,
                          motive=motive, preference=preference, income=income)
            matrix = single_weights.get(key)
        else:
            key = DataKey(f"{cg}_vk", part_of_day=part_of_day, regime=regime,
                          motive=motive, preference=preference, income=income,
                          subtopic="combinaties")
            matrix = combined_weights.get(key)

    if _matrix_cache is not None:
        _matrix_cache[cache_key] = matrix
    return matrix


# ── Sparse/dense safe matvec helpers ─────────────────────────────────

def _matvec(W, v):
    """Compute W @ v, returning a 1-D float32 ndarray regardless of W's type."""
    out = W @ v
    return np.asarray(out, dtype=DTYPE).ravel()


def _matvec_T(W, v):
    """Compute W.T @ v, returning a 1-D float32 ndarray."""
    out = W.T @ v
    return np.asarray(out, dtype=DTYPE).ravel()


# ── Public steps D6 and D7 ───────────────────────────────────────────

def competition_on_destinations(config, single_weights, combined_weights,
                                naive_origins: DataSource):
    """
    Section D6 — Shen citizen-side accessibility A_i.

    Uses D5's output V^{ig,m}_j = naive_origins["Totaal", pod, ig, car_group, m]
    as the competition denominator.  Result is indexed by origin.
    """
    logger.info("Starting step: Shen citizen-side accessibility A_i (D6).")
    return _run_shen(config, single_weights, combined_weights,
                     naive_origins=naive_origins, side="citizen")


def competition_on_citizens(config, single_weights, combined_weights,
                            naive_destinations: DataSource):
    """
    Section D7 — Shen destination-side accessibility B_j.

    Per-group U^{g,m}_i is recomputed internally (one matmul per unique
    weight key) to satisfy Shen's per-group normalisation.  The
    ``naive_destinations`` argument is accepted for API compatibility but
    only its config/context is used.  Result is indexed by destination.
    """
    logger.info("Starting step: Shen destination-side accessibility B_j (D7).")
    return _run_shen(config, single_weights, combined_weights,
                     naive_destinations=naive_destinations, side="destination")


# ── Core Shen pass ───────────────────────────────────────────────────

def _run_shen(config, single_weights, combined_weights,
              naive_origins=None, naive_destinations=None, side=""):
    assert side in ("citizen", "destination")
    is_citizen_side = (side == "citizen")

    project_config = config["project"]
    skims_config = config["skims"]
    distribution_config = config["verdeling"]
    advanced_config = config["geavanceerd"]

    part_of_days = skims_config["dagsoort"]
    scenario = project_config["verstedelijkingsscenario"]
    regime = project_config["beprijzingsregime"]
    motive_name = project_config["motief"]["naam"]
    traveling_population_path = Path(project_config["motief"]["reizende populatie"])
    destinations_path = Path(project_config["motief"]["bestemmingsplaatsen"])
    car_possession_groups = advanced_config["welke_groepen"]
    electric_percentage = distribution_config["Percelektrisch"]

    segs_source = SegsSource(config)
    traveling_population = np.asarray(
        segs_source.read(traveling_population_path.name, scenario=scenario), dtype=DTYPE)
    destinations = np.asarray(
        segs_source.read(destinations_path.name, scenario=scenario), dtype=DTYPE)

    num_zones = len(traveling_population)
    working_population = traveling_population.sum(axis=1)

    # inc^{ig}_i — citizen income share per zone (for the A weighted average)
    income_dist_citizens = compute_income_distributions(traveling_population)

    competitions = DataSource(config, DataType.COMPETITION)

    subtopic = "bestemmingen" if is_citizen_side else "inwoners"
    conc_prefix = "Ontpl" if is_citizen_side else "Pot"

    for car_group in car_possession_groups:
        dist_matrix = np.asarray(segs_source.read(
            "Verdeling_over_groepen", type_caster=float, scenario=scenario,
            group=motive_name,
            modifier="alleen_autobezit" if car_group == "alleen autobezit" else "",
            has_index_column=True,
        ), dtype=DTYPE)

        # P^g_i = s^g_i · workpop_i  (absolute citizens per group per zone)
        citizens_matrix = dist_matrix * working_population[:, np.newaxis]

        for pod in part_of_days:
            # per-modality list of (N,) arrays – one entry per income level,
            # for the across-income CSV write at the end of each pod
            income_stacks: dict[str, list] = {m: [] for m in _MODALITIES}

            for i_ig, ig in enumerate(_INCOME_LEVELS):
                K = electric_percentage.get(ig) / 100
                O_ig = destinations[:, i_ig].astype(DTYPE)        # O^{ig}_j
                inc_ig = income_dist_citizens[:, i_ig]             # inc^{ig}_i
                safe_inc_ig = np.where(inc_ig > 0, inc_ig, 1.0)

                per_modality_results = []

                for modality in _MODALITIES:
                    result = np.zeros(num_zones, dtype=DTYPE)

                    # Per-modality caches (shared between groups with same wck)
                    _matrix_cache: dict = {}
                    _U_cache: dict = {}      # wck -> W @ O_ig  (= U^{g,m}_i)
                    _WOV_cache: dict = {}    # wck -> W @ (O/V) (for A)
                    B_batches: dict = {}     # wck -> Σ P^g over groups sharing wck

                    if is_citizen_side:
                        # Fetch V^{ig,m}_j from D5 and build O/V once per (ig, m)
                        key_V = DataKey(
                            "Totaal", part_of_day=pod, income=ig,
                            group=car_group, motive=motive_name, modality=modality,
                        )
                        V_igm = np.asarray(naive_origins.get(key_V), dtype=DTYPE)
                        OV_ratio = np.where(
                            V_igm > 0, O_ig / np.where(V_igm > 0, V_igm, 1.0), 0
                        ).astype(DTYPE)

                    for i_grp, grp in enumerate(_GROUPS):
                        grp_income = utils.group_income_level(grp)
                        if ig != grp_income and ig != "alle":
                            continue

                        wck = _weight_cache_key(grp, modality)

                        if wck not in _matrix_cache:
                            get_weight_matrix(
                                single_weights, combined_weights,
                                grp, modality, motive_name, regime,
                                pod, grp_income, K,
                                _matrix_cache=_matrix_cache,
                            )
                        W = _matrix_cache[wck]

                        if is_citizen_side:
                            # A^{g,m}_i  = (W @ (O/V))_i,  shared across g sharing wck
                            if wck not in _WOV_cache:
                                _WOV_cache[wck] = _matvec(W, OV_ratio)

                            # Within-income share  s^{g|ig}_i = s^g_i / inc^{ig}_i
                            s_g = dist_matrix[:, i_grp]
                            s_g_given_ig = np.where(
                                inc_ig > 0, s_g / safe_inc_ig, 0
                            ).astype(DTYPE)
                            result += s_g_given_ig * _WOV_cache[wck]
                        else:
                            # Destination side: batch P^g by wck for one matvec_T later
                            if wck not in _U_cache:
                                _U_cache[wck] = _matvec(W, O_ig)   # U^{g,m}_i
                            B_batches.setdefault(
                                wck, np.zeros(num_zones, dtype=DTYPE)
                            )
                            B_batches[wck] += citizens_matrix[:, i_grp]

                    if not is_citizen_side:
                        # B^{ig,m}_j = Σ_wck W_wck^T @ (Σ_g P^g / U_wck)
                        for wck, P_batch in B_batches.items():
                            U_wck = _U_cache[wck]
                            safe_U = np.where(U_wck > 0, U_wck, 1.0)
                            ratio = np.where(
                                U_wck > 0, P_batch / safe_U, 0
                            ).astype(DTYPE)
                            result += _matvec_T(_matrix_cache[wck], ratio)

                    # Cache per-income per-modality result (temporary)
                    temp_key = DataKey(
                        id="Totaal", part_of_day=pod, subtopic=subtopic,
                        income=ig, motive=motive_name, modality=modality,
                        group=car_group, is_temporary=True,
                    )
                    competitions.set(temp_key, result)
                    per_modality_results.append(result)
                    income_stacks[modality].append(result)

                # ── Per-income CSV: rows = zones, cols = modalities ──
                per_income_matrix = np.column_stack(per_modality_results)
                key = DataKey(
                    id=f"{conc_prefix}_conc", part_of_day=pod, subtopic=subtopic,
                    income=ig, motive=motive_name, group=car_group,
                    index=DataKey.zone_index(num_zones),
                )
                competitions.write_csv(per_income_matrix, key,
                                       header=list(_MODALITIES))

            # ── Per-modality CSV across incomes: rows = zones, cols = incomes ──
            income_header = list(_INCOME_LEVELS)
            for modality in _MODALITIES:
                stacked = np.column_stack(income_stacks[modality])   # (N, 4)

                key = DataKey(
                    id=f"{conc_prefix}_conc", part_of_day=pod, subtopic=subtopic,
                    motive=motive_name, modality=modality, group=car_group,
                    index=DataKey.zone_index(num_zones),
                )
                competitions.write_csv(stacked, key, header=income_header)

                # "Product" file: accessibility × counterpart mass per zone.
                #   citizen side:     A^{ig,m}_i · P^{ig}_i        (mask P > 0)
                #   destination side: B^{ig,m}_j · O^{ig}_j        (mask O > 0)
                if is_citizen_side:
                    mass = traveling_population
                    mask = traveling_population > 0
                else:
                    mass = destinations
                    mask = destinations > 0
                product = np.where(mask, stacked * mass, 0)

                key = DataKey(
                    id=f"{conc_prefix}_concproduct", part_of_day=pod,
                    subtopic=subtopic, motive=motive_name, modality=modality,
                    group=car_group, index=DataKey.zone_index(num_zones),
                )
                competitions.write_csv(product, key, header=income_header)

    return competitions