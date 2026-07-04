"""
Fused D4–D7 kernel.

Replaces the four separate steps – reachable destinations (D4), reachable
population (D5), competition on destinations (D6), and competition on
citizens (D7) – with **one** combined pass that shares weight matrices
across all four computations.

Key benefits
~~~~~~~~~~~~
* Each unique weight matrix is fetched / materialised **once** per
  (part-of-day, income, modality) instead of up to four times.
* Lazy combined weights (``sparse_maximum``) are evaluated once per unique
  key rather than being recomputed in every step.
* CSV writes are offloaded to background threads via ``AsyncCsvWriter``.
* Income groups within a (car-group, part-of-day) pair are processed in
  parallel using a ``ThreadPoolExecutor``.

Correctness notes
~~~~~~~~~~~~~~~~~
* D5 fuel blending: the reference splits WelAuto/A-combined citizens into
  fossil/electric batches and multiplies each with the unblended matrix.
  Here we use the blended matrix directly, which is algebraically
  identical:  W_f^T (K_f c) + W_e^T (K_e c) = (K_f W_f + K_e W_e)^T c.
  Only float32 rounding order differs; validate with np.allclose.
* Output formatting mirrors the reference steps exactly:
  - D4 "Ontpl_totaal" (per income and across income): rounded int.
  - D4 "Ontpl_totaalproduct": rounded int (as in reachable_destinations).
  - D5 "Pot_totaal": rounded int; "Pot_totaalproduct": raw float
    (as in reachable_population – the asymmetry is inherited on purpose).
  - D6/D7 "conc" and "concproduct": raw float (as in competition).
* Shen products:
  - D6 (citizen side):     A · traveling_population, masked on P > 0.
  - D7 (destination side): B · destinations,         masked on O > 0.
* This kernel processes all four income levels; it does NOT honor a
  subset in config["project"]["welke_inkomensgroepen"] (the reference
  steps misalign columns for subsets anyway – fix that first if needed).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import ikob.utils as utils
from ikob.async_writer import AsyncCsvWriter
from ikob.competition import _weight_cache_key, get_weight_matrix
from ikob.datasource import DataKey, DataSource, DataType, SegsSource
from ikob.utils import DTYPE

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

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


# ── Helpers ──────────────────────────────────────────────────────────

def _create_citizens(distribution_matrix, working_population):
    """(num_zones, num_groups) citizen matrix."""
    return (
        np.asarray(distribution_matrix, dtype=DTYPE)
        * np.asarray(working_population, dtype=DTYPE)[:, np.newaxis]
    )


def _async_write(writer, ds, data, key, header):
    """Submit a CSV write to the background writer."""
    if key.is_temporary:
        return
    path = ds.get_write_path(key)
    writer.submit(data, path, header=header, index=key.index)


# ── Main entry point ─────────────────────────────────────────────────

def run_fused_d4_d5_d6_d7(config, single_weights, combined_weights):
    """Compute D4, D5, D6, D7 in a single fused pass.

    Returns
    -------
    potencies : DataSource   (D4 – reachable destinations)
    origins   : DataSource   (D5 – reachable population)
    comp_dest : DataSource   (D6 – competition on destinations)
    comp_cit  : DataSource   (D7 – competition on citizens)
    """
    logger.info("Starting fused D4+D5+D6+D7 kernel.")

    # ── Config ───────────────────────────────────────────────────────
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

    # ── Shared input data ────────────────────────────────────────────
    segs_source = SegsSource(config)

    traveling_population = np.asarray(
        segs_source.read(traveling_population_path.name, scenario=scenario),
        dtype=DTYPE,
    )
    destinations_segs = np.asarray(
        segs_source.read(destinations_path.name, scenario=scenario),
        dtype=DTYPE,
    )

    num_zones = len(traveling_population)
    working_population = traveling_population.sum(axis=1)

    # Citizen-side income distributions (from travelling population).
    # Used for: D4 normalisation AND the D6 Shen-A weighted average.
    # NOTE: destination-based income shares must NOT be used for D6 –
    # the reference (competition.py) uses citizen shares there.
    trav_totals = traveling_population.sum(axis=1, keepdims=True)
    safe_trav_totals = np.where(trav_totals > 0, trav_totals, 1.0)
    income_dist_citizens = traveling_population / safe_trav_totals
    income_dist_citizens[trav_totals.ravel() <= 0] = 0.0
    income_dist_citizens_T = income_dist_citizens.T            # (4, N)

    destinations_T = destinations_segs.T                       # (4, N)

    # ── Output containers ────────────────────────────────────────────
    potencies = DataSource(config, DataType.DESTINATIONS)      # D4
    origins = DataSource(config, DataType.ORIGINS)             # D5
    comp_dest = DataSource(config, DataType.COMPETITION)       # D6
    comp_cit = DataSource(config, DataType.COMPETITION)        # D7

    writer = AsyncCsvWriter(num_workers=2)

    zone_idx = DataKey.zone_index(num_zones)
    modality_header = list(_MODALITIES)
    income_header = list(_INCOME_LEVELS)

    # ── Main loop ────────────────────────────────────────────────────
    for car_group in car_possession_groups:
        dist_matrix = np.asarray(
            segs_source.read(
                "Verdeling_over_groepen",
                type_caster=float,
                scenario=scenario,
                group=motive_name,
                modifier=(
                    "alleen_autobezit"
                    if car_group == "alleen autobezit"
                    else ""
                ),
                has_index_column=True,
            ),
            dtype=DTYPE,
        )

        citizens_T = _create_citizens(dist_matrix, working_population).T  # (G, N)

        for pod in part_of_days:
            # Accumulators filled by the per-income-group workers.
            # Keys: (income_group, modality) → 1-D float32 array.
            # Threads write disjoint keys, so plain dicts are safe.
            d4_res: dict[tuple, np.ndarray] = {}
            d5_res: dict[tuple, np.ndarray] = {}
            d6_res: dict[tuple, np.ndarray] = {}
            d7_res: dict[tuple, np.ndarray] = {}

            # ── Per-income cell (parallelisable) ─────────────────────

            def _cell(i_ig: int, income_group: str):
                K = electric_percentage.get(income_group) / 100

                # D4 / D6 vectors
                dest_vec = destinations_T[i_ig]                # O^{ig}_j
                inc_ig = income_dist_citizens_T[i_ig]          # inc^{ig}_i
                safe_inc_ig = np.where(inc_ig > 0, inc_ig, 1.0)

                d4_mod_cols = []
                d5_mod_cols = []
                d6_mod_cols = []
                d7_mod_cols = []

                for modality in _MODALITIES:
                    # Shared weight-matrix cache for D4+D5+D6+D7
                    _mcache: dict = {}

                    # ── Phase 1: D4 + D5 ─────────────────────────
                    d4_sum = np.zeros(num_zones, dtype=DTYPE)
                    d5_sum = np.zeros(num_zones, dtype=DTYPE)
                    _d4_mv: dict[tuple, np.ndarray] = {}
                    _d5_batch: dict[tuple, np.ndarray] = {}

                    for i_grp, grp in enumerate(_GROUPS):
                        inc = utils.group_income_level(grp)
                        if income_group != inc and income_group != "alle":
                            continue

                        wck = _weight_cache_key(grp, modality)

                        # Ensure W is cached (blended for Auto-based groups)
                        if wck not in _mcache:
                            get_weight_matrix(
                                single_weights, combined_weights,
                                grp, modality, motive_name, regime,
                                pod, inc, K,
                                _matrix_cache=_mcache,
                            )

                        # D4: accumulate weighted matmul
                        if wck not in _d4_mv:
                            _d4_mv[wck] = np.asarray(
                                _mcache[wck] @ dest_vec, dtype=DTYPE
                            ).ravel()

                        poss = _d4_mv[wck] * dist_matrix[:, i_grp]
                        poss = poss / safe_inc_ig
                        poss[inc_ig <= 0] = 0
                        d4_sum += poss

                        # D5: batch citizens by weight key
                        if wck not in _d5_batch:
                            _d5_batch[wck] = np.zeros(num_zones, dtype=DTYPE)
                        _d5_batch[wck] += citizens_T[i_grp]

                    # D5 batched matmuls (blended W – see module docstring)
                    for wck, cit_batch in _d5_batch.items():
                        d5_sum += np.asarray(
                            _mcache[wck].T @ cit_batch, dtype=DTYPE
                        ).ravel()

                    # Store totals for across-income aggregation
                    d4_res[(income_group, modality)] = d4_sum
                    d5_res[(income_group, modality)] = d5_sum

                    # Store in DataSource caches (for potential external readers)
                    potencies.set(
                        DataKey(
                            "Totaal", part_of_day=pod, income=income_group,
                            group=car_group, motive=motive_name,
                            modality=modality, is_temporary=True,
                        ),
                        d4_sum,
                    )
                    origins.set(
                        DataKey(
                            "Totaal", part_of_day=pod, income=income_group,
                            group=car_group, motive=motive_name,
                            modality=modality, is_temporary=True,
                        ),
                        d5_sum,
                    )

                    # ── Phase 2: Shen A (D6) and Shen B (D7) ─────
                    # Shen A:   A^{ig,m}_i = Σ_{g∈ig} s^{g|ig}_i · (W_wck @ (O/V))_i
                    # Shen B:   B^{ig,m}_j = Σ_wck W_wck^T @ (Σ_{g with wck} P^g / U_wck)
                    #
                    # Per-wck U is already available as _d4_mv[wck] (= W_wck @ O_ig).
                    # Per-wck P-batch is already available as _d5_batch[wck] (absolute).

                    safe_V = np.where(d5_sum > 0, d5_sum, 1.0)
                    OV_ratio = np.where(
                        d5_sum > 0, dest_vec / safe_V, 0
                    ).astype(DTYPE)

                    d6_tot = np.zeros(num_zones, dtype=DTYPE)
                    d7_tot = np.zeros(num_zones, dtype=DTYPE)
                    _d6_WOV: dict[tuple, np.ndarray] = {}

                    # --- Shen A (citizen-side, D6) ---
                    for i_grp, grp in enumerate(_GROUPS):
                        inc = utils.group_income_level(grp)
                        if income_group != inc and income_group != "alle":
                            continue

                        wck = _weight_cache_key(grp, modality)

                        if wck not in _d6_WOV:
                            W = _mcache[wck]
                            _d6_WOV[wck] = np.asarray(
                                W @ OV_ratio, dtype=DTYPE
                            ).ravel()

                        s_g = dist_matrix[:, i_grp]
                        s_g_given_ig = np.where(
                            inc_ig > 0, s_g / safe_inc_ig, 0
                        ).astype(DTYPE)
                        d6_tot += s_g_given_ig * _d6_WOV[wck]

                    # --- Shen B (destination-side, D7) ---
                    # Batch P^g by wck so only one W_wck^T @ (·) per unique W.
                    for wck, P_batch in _d5_batch.items():
                        U_wck = _d4_mv[wck]
                        safe_U = np.where(U_wck > 0, U_wck, 1.0)
                        ratio = np.where(
                            U_wck > 0, P_batch / safe_U, 0
                        ).astype(DTYPE)
                        W = _mcache[wck]
                        d7_tot += np.asarray(W.T @ ratio, dtype=DTYPE).ravel()

                    d6_res[(income_group, modality)] = d6_tot
                    d7_res[(income_group, modality)] = d7_tot

                    comp_dest.set(
                        DataKey(
                            "Totaal", part_of_day=pod,
                            subtopic="bestemmingen",
                            income=income_group, motive=motive_name,
                            modality=modality, group=car_group,
                            is_temporary=True,
                        ),
                        d6_tot,
                    )
                    comp_cit.set(
                        DataKey(
                            "Totaal", part_of_day=pod,
                            subtopic="inwoners",
                            income=income_group, motive=motive_name,
                            modality=modality, group=car_group,
                            is_temporary=True,
                        ),
                        d7_tot,
                    )

                    d4_mod_cols.append(d4_sum)
                    d5_mod_cols.append(d5_sum)
                    d6_mod_cols.append(d6_tot)
                    d7_mod_cols.append(d7_tot)

                # ── Per-income CSV writes ────────────────────────

                # D4 per-income (all modalities) – rounded int (reference)
                _async_write(
                    writer, potencies,
                    np.round(np.column_stack(d4_mod_cols)).astype(int),
                    DataKey(
                        "Ontpl_totaal", part_of_day=pod,
                        group=car_group, income=income_group,
                        motive=motive_name, index=zone_idx,
                    ),
                    header=modality_header,
                )

                # D5 per-income – rounded int (reference)
                _async_write(
                    writer, origins,
                    np.round(np.column_stack(d5_mod_cols)).astype(int),
                    DataKey(
                        "Pot_totaal", part_of_day=pod,
                        group=car_group, income=income_group,
                        motive=motive_name, index=zone_idx,
                    ),
                    header=modality_header,
                )

                # D6 per-income – raw float (reference)
                _async_write(
                    writer, comp_dest,
                    np.column_stack(d6_mod_cols),
                    DataKey(
                        "Ontpl_conc", part_of_day=pod,
                        subtopic="bestemmingen",
                        income=income_group, motive=motive_name,
                        group=car_group, index=zone_idx,
                    ),
                    header=modality_header,
                )

                # D7 per-income – raw float (reference)
                _async_write(
                    writer, comp_cit,
                    np.column_stack(d7_mod_cols),
                    DataKey(
                        "Pot_conc", part_of_day=pod,
                        subtopic="inwoners",
                        income=income_group, motive=motive_name,
                        group=car_group, index=zone_idx,
                    ),
                    header=modality_header,
                )

            # ── Run income cells in parallel ─────────────────────
            with ThreadPoolExecutor(
                max_workers=min(4, len(_INCOME_LEVELS))
            ) as pool:
                futs = [
                    pool.submit(_cell, i, ig)
                    for i, ig in enumerate(_INCOME_LEVELS)
                ]
                for f in as_completed(futs):
                    f.result()          # propagate exceptions

            # ── Across-income aggregation ────────────────────────
            for modality in _MODALITIES:

                # -- D4 across income --
                d4_cols = [
                    d4_res[(ig, modality)] for ig in _INCOME_LEVELS
                ]
                d4_stack = np.column_stack(d4_cols)

                _async_write(
                    writer, potencies,
                    np.round(d4_stack).astype(int),
                    DataKey(
                        "Ontpl_totaal", part_of_day=pod,
                        group=car_group, motive=motive_name,
                        modality=modality, index=zone_idx,
                    ),
                    header=income_header,
                )
                # Reference (reachable_destinations) rounds this product.
                _async_write(
                    writer, potencies,
                    np.round(
                        np.where(
                            traveling_population > 0,
                            d4_stack * traveling_population, 0,
                        )
                    ).astype(int),
                    DataKey(
                        "Ontpl_totaalproduct", part_of_day=pod,
                        group=car_group, motive=motive_name,
                        modality=modality, index=zone_idx,
                    ),
                    header=income_header,
                )

                # -- D5 across income --
                d5_cols = [
                    d5_res[(ig, modality)] for ig in _INCOME_LEVELS
                ]
                d5_stack = np.column_stack(d5_cols)

                _async_write(
                    writer, origins,
                    np.round(d5_stack).astype(int),
                    DataKey(
                        "Pot_totaal", part_of_day=pod,
                        group=car_group, motive=motive_name,
                        modality=modality, index=zone_idx,
                    ),
                    header=income_header,
                )
                # Reference (reachable_population) does NOT round this one.
                _async_write(
                    writer, origins,
                    np.where(
                        destinations_segs > 0,
                        d5_stack * destinations_segs, 0,
                    ),
                    DataKey(
                        "Pot_totaalproduct", part_of_day=pod,
                        group=car_group, motive=motive_name,
                        modality=modality, index=zone_idx,
                    ),
                    header=income_header,
                )

                # -- D6 across income --
                d6_cols = [
                    d6_res[(ig, modality)] for ig in _INCOME_LEVELS
                ]
                d6_stack = np.column_stack(d6_cols)

                _async_write(
                    writer, comp_dest,
                    d6_stack,
                    DataKey(
                        "Ontpl_conc", part_of_day=pod,
                        subtopic="bestemmingen",
                        motive=motive_name, modality=modality,
                        group=car_group, index=zone_idx,
                    ),
                    header=income_header,
                )
                # Citizen side: A · traveling_population, masked on P > 0.
                _async_write(
                    writer, comp_dest,
                    np.where(
                        traveling_population > 0,
                        d6_stack * traveling_population, 0,
                    ),
                    DataKey(
                        "Ontpl_concproduct", part_of_day=pod,
                        subtopic="bestemmingen",
                        motive=motive_name, modality=modality,
                        group=car_group, index=zone_idx,
                    ),
                    header=income_header,
                )

                # -- D7 across income --
                d7_cols = [
                    d7_res[(ig, modality)] for ig in _INCOME_LEVELS
                ]
                d7_stack = np.column_stack(d7_cols)

                _async_write(
                    writer, comp_cit,
                    d7_stack,
                    DataKey(
                        "Pot_conc", part_of_day=pod,
                        subtopic="inwoners",
                        motive=motive_name, modality=modality,
                        group=car_group, index=zone_idx,
                    ),
                    header=income_header,
                )
                # Destination side: B · destinations, masked on O > 0.
                _async_write(
                    writer, comp_cit,
                    np.where(
                        destinations_segs > 0,
                        d7_stack * destinations_segs, 0,
                    ),
                    DataKey(
                        "Pot_concproduct", part_of_day=pod,
                        subtopic="inwoners",
                        motive=motive_name, modality=modality,
                        group=car_group, index=zone_idx,
                    ),
                    header=income_header,
                )

    # Drain background writes before returning
    writer.shutdown()
    logger.info("Fused D4+D5+D6+D7 kernel complete.")
    return potencies, origins, comp_dest, comp_cit