import numpy as np

from tests.unit.conftest import SegsCapture


def _minimal_config():
    return {
        "__filename__": "pytest",
        "project": {
            "verstedelijkingsscenario": "2023",
            "motieven": ["werk"],
            "paden": {
                "segs_directory": "segs",
                "output_directory": "out",
                "skims_directory": "skims",
            },
        },
        "verdeling": {
            "GratisOVpercentage": 0.03,
        },
        "geavanceerd": {
            "kunstmab": {"gebruiken": False, "bestand": "unused.csv"},
        },
    }


def _expected_laag_header_prefix() -> list[str]:
    return [
        "GratisAuto_laag",
        "GratisAuto_GratisOV_laag",
        "WelAuto_GratisOV_laag",
        "WelAuto_vkAuto_laag",
        "WelAuto_vkNeutraal_laag",
        "WelAuto_vkFiets_laag",
        "WelAuto_vkOV_laag",
        "GeenAuto_GratisOV_laag",
        "GeenAuto_vkNeutraal_laag",
        "GeenAuto_vkFiets_laag",
        "GeenAuto_vkOV_laag",
        "GeenRijbewijs_GratisOV_laag",
        "GeenRijbewijs_vkNeutraal_laag",
        "GeenRijbewijs_vkFiets_laag",
        "GeenRijbewijs_vkOV_laag",
    ]


def _expected_laag_block_for_zone(
    *,
    zone_index: int,
    segs: dict[tuple[str, str], np.ndarray],
    free_pt_percentage: float,
    scenario: str = "2023",
) -> np.ndarray:
    """Compute expected 15 output values for income class 'laag' for a zone.

    This mirrors the calculations in `ikob.distribute_over_groups.distribute_over_groups`.
    It's not ideal that the test in part reimplements the same logic as the code under test, but still the tested code is significantly more obtuse.
    """

    income_index = 0  # 'laag'

    # Fixed values from the implementation.
    free_car_per_income = [0]  # only 'laag' is relevant here.
    preferences = ["Auto", "Neutraal", "Fiets", "OV"]
    preferences_no_car = ["Neutraal", "Fiets", "OV"]

    car_possessions_per_household = np.array(segs[("CBS_autos_per_huishouden", "")], dtype=float)
    urbanization_grade = np.array(segs[("Stedelijkheidsgraad", "")], dtype=int)
    urbanization = [int(sgg) - 1 for sgg in urbanization_grade]

    no_license_segs = np.array(segs[("GeenRijbewijs", "")], dtype=float)
    no_car_segs = np.array(segs[("GeenAuto", "")], dtype=float)
    with_car_segs = np.array(segs[("WelAuto", "")], dtype=float)
    preferences_segs = np.array(segs[("Voorkeuren", "")], dtype=float)
    preferences_no_car_segs = np.array(segs[("VoorkeurenGeenAuto", "")], dtype=float)
    citizens_per_class = np.array(segs[("Beroepsbevolking_inkomensklasse", scenario)], dtype=float)

    income_distribution_over_population = citizens_per_class[zone_index, :] / np.sum(citizens_per_class[zone_index, :])

    income_share_of_population = income_distribution_over_population[income_index]

    urb = urbanization[zone_index]
    # theoretical_car_possession = sum over income classes of (percentage_with_car_per_income_class * income_distribution_over_population)
    share_of_population_with_car = np.sum((with_car_segs[urb] / 100) * income_distribution_over_population)
    car_possessions_per_household = car_possessions_per_household[zone_index]

    # car_possession_per_income_class is computed from the theoretical car possession using a correction factor
    #   car_possession_per_income_class = theoretical_car_possession_per_income_class * (car_possession / theoretical_car_possession)

    # From the docs: if car_possessions_per_household / 100 < share_of_population_with_car:
    #   Then few households will have multiple cars and the car_possession is equal to the share of population with a car.
    # Otherwise:
    #   The car_possessions is equal to the theoretical share of the population with a car.
    car_possession_correction = 1.0
    if car_possessions_per_household / 100 < share_of_population_with_car:
        car_possession_correction = (car_possessions_per_household / 100) / share_of_population_with_car

    with_car_share_theoretical = with_car_segs[urb][income_index] / 100
    with_car = with_car_share_theoretical * car_possession_correction

    if car_possession_correction != 1:
        no_car_correction = (1 - with_car) / (1 - with_car_share_theoretical)
    else:
        no_car_correction = 1.0

    no_car_with_license = no_car_segs[urb][income_index] / 100 * no_car_correction
    no_license = no_license_segs[urb][income_index] / 100 * no_car_correction

    free_car = with_car * free_car_per_income[income_index]
    no_free_car = with_car - free_car

    out: list[float] = []
    # GratisAuto_laag
    out.append(free_car * (1 - free_pt_percentage) * income_share_of_population)
    # GratisAuto_GratisOV_laag
    out.append(free_car * free_pt_percentage * income_share_of_population)
    # WelAuto_GratisOV_laag
    out.append(no_free_car * free_pt_percentage * income_share_of_population)
    # WelAuto_vk* columns
    for i_pref in range(len(preferences)):
        share_pref = no_free_car * (1 - free_pt_percentage) * (preferences_segs[urb][i_pref] / 100)
        out.append(share_pref * income_share_of_population)
    # GeenAuto_GratisOV + GeenAuto_vk*
    out.append(no_car_with_license * free_pt_percentage * income_share_of_population)
    for i_pref in range(len(preferences_no_car)):
        share_pref = no_car_with_license * (1 - free_pt_percentage) * (preferences_no_car_segs[urb][i_pref] / 100)
        out.append(share_pref * income_share_of_population)
    # GeenRijbewijs_GratisOV + GeenRijbewijs_vk*
    out.append(no_license * free_pt_percentage * income_share_of_population)
    for i_pref in range(len(preferences_no_car)):
        share_pref = no_license * (1 - free_pt_percentage) * (preferences_no_car_segs[urb][i_pref] / 100)
        out.append(share_pref * income_share_of_population)

    assert len(out) == 15
    return np.array(out, dtype=float)


def test_distribute_over_groups_computation(segs_capture):
    from ikob.distribute_over_groups import distribute_population_over_groups

    # Prepare
    # The `segs` dict represents the in-memory contents of the SEGS datasource.
    # Keys are `(dataset_id, scenario)`.
    segs = {
        # Per-zone average cars per household (used to derive free-car share).
        ("CBS_autos_per_huishouden", ""): np.array([0.5, 1.5]) * 100,
        # Urbanization grade per zone.
        ("Stedelijkheidsgraad", ""): np.array([1, 2]),
        # Urbanization grades x 4 income classes (low..high): counts/shares by car/license status.
        ("GeenRijbewijs", ""): np.array([[10, 10, 10, 10], [5, 5, 5, 5]]),
        ("GeenAuto", ""): np.array([[10, 10, 10, 10], [5, 5, 5, 5]]),
        ("WelAuto", ""): np.array([[80, 80, 80, 80], [90, 90, 90, 90]]),
        # Urbanization grades x 4 preference categories (model-specific preference bins).
        ("Voorkeuren", ""): np.array([[25, 25, 25, 25], [25, 25, 25, 25]]),
        # Urbanization grades x 3 preferences (only for the "GeenAuto" segment).
        ("VoorkeurenGeenAuto", ""): np.array([[34, 33, 33], [34, 33, 33]]),
        # Per-zone x 4 income classes: population counts per income class.
        ("Beroepsbevolking_inkomensklasse", "2023"): np.array([[10, 10, 10, 10], [20, 20, 20, 20]]),
    }
    capture: SegsCapture = segs_capture(segs)

    config = _minimal_config()
    # Act
    distribute_population_over_groups(config)

    # Assert
    # distribute_over_groups should write a CSV for the computed distribution.
    writes = [w for w in capture.writes_csv if w["id"] == "Verdeling_over_groepen"]
    assert writes, "Expected distribute_over_groups to call SegsSource.write_csv"

    # The main output is written for group=Beroepsbevolking, modifier="".
    main = [w for w in writes if w["group"] == "Beroepsbevolking" and w["modifier"] == ""]
    assert len(main) == 1
    data = main[0]["data"]
    header = main[0]["header"]

    assert data.shape == (2, 60)  # 2 zones, 60 group categories.

    # The total distribution of the population over all groups sums to 1 per zone.
    np.testing.assert_allclose(data.sum(axis=1), np.ones(2), rtol=1e-7, atol=1e-7)

    # Each income block is a partition of that income's population share.
    # In this fixture, each income class has equal weight: 0.25.
    for income_block in range(4):
        start = income_block * 15
        end = start + 15
        np.testing.assert_allclose(data[0, start:end].sum(), 0.25, rtol=1e-12, atol=1e-12)

    # Assert every output value for income class 'laag' (the first 15 columns)
    # using the same calculations as the implementation.
    assert header[:15] == _expected_laag_header_prefix()
    expected_laag_zone0 = _expected_laag_block_for_zone(
        zone_index=0, segs=segs, free_pt_percentage=config["verdeling"]["GratisOVpercentage"]
    )
    np.testing.assert_allclose(data[0, :15], expected_laag_zone0, rtol=1e-12, atol=1e-12)


def test_distribution_of_income_group_is_independent_of_population_distribution(segs_capture):
    """
    The distribution of an income group over the groups is not related to how the total population is distributed among income groups.
    """
    from ikob.distribute_over_groups import distribute_population_over_groups

    # Two zones identical in every way except income distribution.
    segs = {
        ("CBS_autos_per_huishouden", ""): np.array([0.8, 0.8]) * 100,
        ("Stedelijkheidsgraad", ""): np.array([1, 1]),
        ("GeenRijbewijs", ""): np.array([[10, 10, 10, 10], [10, 10, 10, 10]]),
        ("GeenAuto", ""): np.array([[10, 10, 10, 10], [10, 10, 10, 10]]),
        ("WelAuto", ""): np.array([[10, 80, 80, 80], [10, 80, 80, 80]]),
        ("Voorkeuren", ""): np.array([[25, 25, 25, 25], [25, 25, 25, 25]]),
        ("VoorkeurenGeenAuto", ""): np.array([[34, 33, 33], [34, 33, 33]]),
        ("Beroepsbevolking_inkomensklasse", "2023"): np.array([[209, 209, 209, 208], [2, 209, 209, 208]]),
    }
    capture: SegsCapture = segs_capture(segs)
    config = _minimal_config()
    distribute_population_over_groups(config)

    writes = [w for w in capture.writes_csv if w["id"] == "Verdeling_over_groepen"]
    main = [w for w in writes if w["group"] == "Beroepsbevolking" and w["modifier"] == ""]
    data = main[0]["data"]

    # Normalize the 'laag' income group distribution (columns 0-15) for each zone.
    laag_dist_zone0 = data[0, 0:15] / data[0, 0:15].sum()
    hoog_dist_zone0 = data[0, 45:60] / data[0, 45:60].sum()
    laag_dist_zone1 = data[1, 0:15] / data[1, 0:15].sum()
    hoog_dist_zone1 = data[1, 45:60] / data[1, 45:60].sum()

    # The normalized distribution within the income group should be identical
    # regardless of how the total population is distributed across income classes.
    np.testing.assert_allclose(laag_dist_zone0, laag_dist_zone1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hoog_dist_zone0, hoog_dist_zone1, rtol=1e-12, atol=1e-12)


def test_distribution_of_groups_dependent_on_car_possession_correction_factor(segs_capture):
    """
    The distribution of an income group over the groups is not related to how the total population is distributed among income groups
    EXCEPT when the car possession correction factor is relevant.
    The correction factor is dependent on the % of households with a car in a zone.
    This is computed from the the "WelAuto" segs combined with "Beroepsbevolking_inkomensklasse",
    and therefore dependent on the the distribution of the population over income groups.

    The car possession factor only comes in to play when the the % of households with a car is *more* than the average number of cars per household.
    So a small 'CBS_autos_per_huishouden' achieves this state.

    """
    from ikob.distribute_over_groups import distribute_population_over_groups

    # Two zones identical in every way except income distribution.
    segs = {
        ("CBS_autos_per_huishouden", ""): np.array([0.05, 0.05]) * 100,
        ("Stedelijkheidsgraad", ""): np.array([1, 1]),
        ("GeenRijbewijs", ""): np.array([[10, 10, 10, 10], [10, 10, 10, 10]]),
        ("GeenAuto", ""): np.array([[10, 10, 10, 10], [10, 10, 10, 10]]),
        ("WelAuto", ""): np.array([[10, 80, 80, 80], [10, 80, 80, 80]]),
        ("Voorkeuren", ""): np.array([[25, 25, 25, 25], [25, 25, 25, 25]]),
        ("VoorkeurenGeenAuto", ""): np.array([[34, 33, 33], [34, 33, 33]]),
        ("Beroepsbevolking_inkomensklasse", "2023"): np.array([[209, 209, 209, 208], [2, 209, 209, 208]]),
    }
    capture: SegsCapture = segs_capture(segs)
    config = _minimal_config()
    distribute_population_over_groups(config)

    writes = [w for w in capture.writes_csv if w["id"] == "Verdeling_over_groepen"]
    main = [w for w in writes if w["group"] == "Beroepsbevolking" and w["modifier"] == ""]
    data = main[0]["data"]

    # Normalize the 'laag' income group distribution (columns 0-15) for each zone.
    laag_dist_zone0 = data[0, 0:15] / data[0, 0:15].sum()
    hoog_dist_zone0 = data[0, 45:60] / data[0, 45:60].sum()
    laag_dist_zone1 = data[1, 0:15] / data[1, 0:15].sum()
    hoog_dist_zone1 = data[1, 45:60] / data[1, 45:60].sum()

    # The normalized distribution within the income group should be identical
    # regardless of how the total population is distributed across income classes.
    assert np.abs(laag_dist_zone0 - laag_dist_zone1).sum() > 1e-6
    assert np.abs(hoog_dist_zone0 - hoog_dist_zone1).sum() > 1e-6
