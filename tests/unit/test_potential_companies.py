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
            [100.0, 0.0, 0.0, 40.0],
            [100.0, 0.0, 0.0, 20.0],
        ]
    )

    # All zones have the same working population for low income.
    working_pop_income = np.array(
        [
            [50.0, 50.0, 0, 100.0],
            [100.0, 100.0, 0, 200.0],
        ]
    )

    # Distribution matrix for target group: 2 zones × 60 columns.
    # Total results should not be dependent on this distribution.
    distribution_per_income = np.zeros((2, 15), dtype=float)
    distribution_per_income[0, 0] = 1.0
    distribution_per_income[1, 0] = 0.5
    distribution_per_income[1, 1] = 0.5

    distribution = np.zeros((2, 60), dtype=float)
    distribution[:, 0:15] = distribution_per_income * (1 / 4)
    distribution[:, 15:30] = distribution_per_income * (1 / 4)
    distribution[:, 45:60] = distribution_per_income * (1 / 2)

    segs_capture(
        {
            ("Beroepsbevolking_inkomensklasse", "2023"): working_pop_income,
            ("Arbeidsplaatsen_inkomensklasse", "2023"): jobs_income,
            ("Verdeling_over_groepen_Beroepsbevolking", "2023"): distribution,
        }
    )

    # Diagonal weight matrix: each zone only reaches its own jobs.
    weight = 0.8
    weight_matrix = np.eye(2) * weight

    class _Weights:
        def get(self, _key):
            return weight_matrix

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

    origins = pc.potential_companies(config, _Weights(), _Weights())  # type: ignore

    return {
        "origins": origins,
        "xlsx_writes": xlsx_writes,
        "pod": pod,
        "motive": motive,
        "working_pop_income": working_pop_income,
        "jobs_income": jobs_income,
        "weight": weight,
    }


# As defined in potential_companies
modalities = ["Fiets", "Auto", "OV", "Auto_Fiets", "OV_Fiets", "Auto_OV", "Auto_OV_Fiets"]


@pytest.mark.parametrize("modality", modalities)
@pytest.mark.parametrize(
    ("income_group", "income_index"), (("laag", 0), ("middellaag", 1), ("middelhoog", 2), ("hoog", 3))
)
def test_potential_companies_totals(modality, income_group, income_index, potential_companies_setup):
    """Reachable citizens (potential workforce) totals are independent of job counts and distribution over groups."""
    from ikob.datasource import DataKey

    origins = potential_companies_setup["origins"]
    pod = potential_companies_setup["pod"]
    motive = potential_companies_setup["motive"]
    working_pop_income = potential_companies_setup["working_pop_income"]
    weight = potential_companies_setup["weight"]

    key = DataKey(
        "Totaal",
        part_of_day=pod,
        income=income_group,
        group="alle groepen",
        motive=motive,
        modality=modality,
    )
    totals = origins.get(key)

    # Intended behavior (diagonal reach): each destination receives its own working population.
    expected_totaal = np.array(working_pop_income[:, income_index] * weight)
    np.testing.assert_allclose(totals, expected_totaal)


@pytest.mark.parametrize("modality", modalities)
def test_potential_companies_pot_totaal(modality, potential_companies_setup):
    """Pot_totaal xlsx output shows reachability by income group, independent of distribution over groups."""

    xlsx_writes = potential_companies_setup["xlsx_writes"]
    working_pop_income = potential_companies_setup["working_pop_income"]
    jobs_income = potential_companies_setup["jobs_income"]
    weight = potential_companies_setup["weight"]

    # Test Pot_totaal xlsx write (per modality, showing reachability by income group)
    pot_totaal_writes = [w for w in xlsx_writes if w["key"].id == "Pot_totaal" and w["key"].modality == modality]
    assert len(pot_totaal_writes) == 1, f"Expected exactly one Pot_totaal write for modality {modality}"

    pot_totaal_data = pot_totaal_writes[0]["data"]
    # Since each destination only receives from its own zone, the reachability equals working_pop_income values.
    np.testing.assert_array_equal(pot_totaal_data, working_pop_income * weight)

    # Test Pot_totaalproduct xlsx write (product of reachability and jobs)
    pot_product_writes = [
        w for w in xlsx_writes if w["key"].id == "Pot_totaalproduct" and w["key"].modality == modality
    ]
    assert len(pot_product_writes) == 1, f"Expected exactly one Pot_totaalproduct write for modality {modality}"

    pot_product_data = pot_product_writes[0]["data"]
    # Product should be reachability * jobs per zone and income
    np.testing.assert_array_equal(pot_product_data, working_pop_income * weight * jobs_income)
