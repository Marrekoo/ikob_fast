import numpy as np
import pytest


@pytest.fixture
def potential_companies_setup(monkeypatch, segs_capture):
    """Common setup for potential companies tests."""
    import ikob.potential_companies as pc

    pod = "Restdag"
    motive = "werk"
    regime = "Basis"

    # Job counts should not matter for total potential companies (citizens reaching destinations).
    jobs_income = np.array(
        [
            [50.0, 50.0, 0.0, 0.0],
            [100.0, 100.0, 0.0, 0.0],
        ]
    )

    # All zones have the same working population for low income.
    working_pop_income = np.array(
        [
            [100.0, 0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0, 0.0],
        ]
    )

    # Distribution matrix for target group: 2 zones × 60 columns.
    # Total results should not be dependent on this distribution.
    distribution = np.zeros((2, 60), dtype=float)
    distribution[0, 0] = 1.0
    distribution[1, 0] = 0.5
    distribution[1, 1] = 0.5

    segs_capture(
        {
            ("Beroepsbevolking_inkomensklasse", "2023"): working_pop_income,
            ("Arbeidsplaatsen_inkomensklasse", "2023"): jobs_income,
            ("Verdeling_over_groepen_Beroepsbevolking", "2023"): distribution,
        }
    )

    # Identity weight matrix: each destination only receives from its own zone.
    identity = np.eye(2)

    class _Weights:
        def get(self, _key):
            return identity

    # Capture xlsx writes
    xlsx_writes = []

    def capture_write_xlsx(self, data, key, header=None):
        xlsx_writes.append({"data": data, "key": key, "header": header})

    monkeypatch.setattr(pc.DataSource, "write_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(pc.DataSource, "write_xlsx", capture_write_xlsx)

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

    origins = pc.potential_companies(config, _Weights(), _Weights())

    return {
        "origins": origins,
        "xlsx_writes": xlsx_writes,
        "pod": pod,
        "motive": motive,
        "working_pop_income": working_pop_income,
        "jobs_income": jobs_income,
    }


def test_potential_companies_totals(potential_companies_setup):
    """Reachable citizens (potential workforce) totals are independent of job counts and distribution over groups."""
    from ikob.datasource import DataKey

    origins = potential_companies_setup["origins"]
    pod = potential_companies_setup["pod"]
    motive = potential_companies_setup["motive"]

    key = DataKey(
        "Totaal",
        part_of_day=pod,
        income="laag",
        group="alle groepen",
        motive=motive,
        modality="Auto",
    )
    totals = origins.get(key)

    # Intended behavior (identity reach): each destination receives its own working population.
    expected_totaal = np.array([100.0, 100.0])
    np.testing.assert_allclose(totals, expected_totaal)


def test_potential_companies_pot_totaal(potential_companies_setup):
    """Pot_totaal xlsx output shows reachability by income group, independent of distribution over groups."""

    xlsx_writes = potential_companies_setup["xlsx_writes"]
    working_pop_income = potential_companies_setup["working_pop_income"]
    jobs_income = potential_companies_setup["jobs_income"]

    # Test Pot_totaal xlsx write (per modality, showing reachability by income group)
    pot_totaal_writes = [w for w in xlsx_writes if w["key"].id == "Pot_totaal" and w["key"].modality == "Auto"]
    assert len(pot_totaal_writes) == 1, "Expected exactly one Pot_totaal write for Auto modality"

    pot_totaal_data = pot_totaal_writes[0]["data"]
    # Since each destination only receives from its own zone, the reachability equals working_pop_income values.
    np.testing.assert_array_equal(pot_totaal_data, working_pop_income)

    # Test Pot_totaalproduct xlsx write (product of reachability and jobs)
    pot_product_writes = [w for w in xlsx_writes if w["key"].id == "Pot_totaalproduct" and w["key"].modality == "Auto"]
    assert len(pot_product_writes) == 1, "Expected exactly one Pot_totaalproduct write for Auto modality"

    pot_product_data = pot_product_writes[0]["data"]
    # Product should be reachability * jobs per zone and income
    np.testing.assert_array_equal(pot_product_data, working_pop_income * jobs_income)
