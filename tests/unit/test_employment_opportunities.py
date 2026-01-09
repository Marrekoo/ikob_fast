import numpy as np
import pytest


@pytest.fixture
def employment_opportunities_setup(monkeypatch, segs_capture):
    """Common setup for employment opportunities tests."""
    import ikob.employment_opportunities as employment_opportunities

    pod = "Restdag"
    motive = "werk"
    regime = "Basis"

    # Working population size should not matter for total employment opportunities.
    working_pop_income = np.array(
        [
            [50.0, 50.0, 0.0, 0.0],
            [100.0, 100.0, 0.0, 0.0],
        ]
    )

    # All zones have the same job opportunities for low income.
    jobs_income = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0, 0.0],
        ]
    )

    # Distribution matrix for target group: 2 zones × 60 columns (15 per income class).
    # The distribution over the groups should be inline with the actual working population.
    distribution_per_income = np.zeros((2, 15), dtype=float)
    distribution_per_income[0, 0] = 1.0
    distribution_per_income[1, 0] = 0.5
    distribution_per_income[1, 1] = 0.5

    distribution = np.zeros((2, 60), dtype=float)
    distribution[:, 0:15] = distribution_per_income * 0.5
    distribution[:, 15:30] = distribution_per_income * 0.5

    segs_capture(
        {
            ("Beroepsbevolking_inkomensklasse", "2023"): working_pop_income,
            ("Arbeidsplaatsen_inkomensklasse", "2023"): jobs_income,
            ("Verdeling_over_groepen_Beroepsbevolking", "2023"): distribution,
        }
    )

    # Identity weight matrix: each zone only reaches its own jobs.
    identity = np.eye(2)
    monkeypatch.setattr(employment_opportunities, "get_weight_matrix", lambda *args, **kwargs: identity)

    # Capture xlsx writes
    xlsx_writes = []

    def capture_write_xlsx(self, data, key, header=None):
        xlsx_writes.append({"data": data, "key": key, "header": header})

    monkeypatch.setattr(employment_opportunities.DataSource, "write_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(employment_opportunities.DataSource, "write_xlsx", capture_write_xlsx)

    class _Weights:
        def get(self, _key):
            return identity

    config = {
        "__filename__": "pytest",
        "project": {
            "verstedelijkingsscenario": "2023",
            "beprijzingsregime": regime,
            "motieven": [motive],
            "welke_inkomensgroepen": ["laag", "middellaag", "middelhoog", "hoog"],
            "paden": {
                "output_directory": "out",
                "skims_directory": "skims",
                "segs_directory": "segs",
            },
        },
        "skims": {"dagsoort": [pod]},
        "verdeling": {"Percelektrisch": {"laag": 0.0, "middellaag": 0.0, "middelhoog": 0.0, "hoog": 0.0}},
        "geavanceerd": {"welke_groepen": ["alle groepen"]},
    }

    potencies = employment_opportunities.employment_opportunities(config, _Weights(), _Weights())

    return {
        "potencies": potencies,
        "xlsx_writes": xlsx_writes,
        "pod": pod,
        "motive": motive,
        "working_pop_income": working_pop_income,
        "jobs_income": jobs_income,
    }


def test_employment_opportunities_totals(employment_opportunities_setup):
    """Reachable employment opportunities totals are independent of working population size and distribution over groups."""
    from ikob.datasource import DataKey

    setup = employment_opportunities_setup
    potencies = setup["potencies"]
    pod = setup["pod"]
    motive = setup["motive"]

    key = DataKey(
        "Totaal",
        part_of_day=pod,
        income="laag",
        group="alle groepen",
        motive=motive,
        modality="Auto",
    )
    totals = potencies.get(key)

    # Intended behavior (identity reach): each zone reaches its own jobs.
    expected_totaal = np.array([100.0, 100.0])
    np.testing.assert_allclose(totals, expected_totaal)


def test_employment_opportunities_ontpl_totaal(employment_opportunities_setup):
    """Ontpl_totaal xlsx output shows reachability by income group, independent of distribution over groups."""

    setup = employment_opportunities_setup
    xlsx_writes = setup["xlsx_writes"]
    working_pop_income = setup["working_pop_income"]
    jobs_income = setup["jobs_income"]

    # Test Ontpl_totaal xlsx write (per modality, showing reachability by income group)
    ontpl_totaal_writes = [w for w in xlsx_writes if w["key"].id == "Ontpl_totaal" and w["key"].modality == "Auto"]
    assert len(ontpl_totaal_writes) == 1, "Expected exactly one Ontpl_totaal write for Auto modality"

    ontpl_totaal_data = ontpl_totaal_writes[0]["data"]
    # Since all zones can only reach their own jobs, the reachability is equal to the jobs_income values.
    np.testing.assert_array_equal(ontpl_totaal_data, jobs_income)

    # Test Ontpl_totaalproduct xlsx write (product of reachability and population)
    ontpl_product_writes = [
        w for w in xlsx_writes if w["key"].id == "Ontpl_totaalproduct" and w["key"].modality == "Auto"
    ]
    assert len(ontpl_product_writes) == 1, "Expected exactly one Ontpl_totaalproduct write for Auto modality"

    ontpl_product_data = ontpl_product_writes[0]["data"]
    # Product should be reachability * working_population per zone and income
    np.testing.assert_array_equal(ontpl_product_data, jobs_income * working_pop_income)
