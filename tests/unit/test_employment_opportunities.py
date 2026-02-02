import numpy as np
import pytest


@pytest.fixture(
    params=[
        np.array([[0.8, 0.15, 0.05], [0.2, 0.7, 0.1], [0.1, 0.2, 0.5]]),
        np.eye(3) * 0.8,
    ],
    ids=["complicated_matrix", "diagonal_matrix"],
)
def employment_opportunities_setup(request, monkeypatch, segs_capture):
    """Common setup for employment opportunities tests."""
    import ikob.employment_opportunities as employment_opportunities

    pod = "Restdag"
    motive = "werk"
    regime = "Basis"

    # Working population size should not matter for total employment opportunities.
    working_pop_income = np.array(
        [
            [50.0, 50.0, 0, 100.0],
            [100.0, 100.0, 0, 200.0],
            [75.0, 75.0, 0, 150.0],
        ]
    )

    # All zones have the same job opportunities for low income.
    jobs_income = np.array(
        [
            [100.0, 0.0, 0.0, 40.0],
            [100.0, 0.0, 0.0, 20.0],
            [80.0, 0.0, 0.0, 30.0],
        ]
    )

    # Distribution matrix for target group: 3 zones × 60 columns (15 per income class).
    # The distribution over the groups should be inline with the actual working population.
    distribution_per_income = np.zeros((3, 15), dtype=float)
    distribution_per_income[0, 0] = 1.0
    distribution_per_income[1, 0] = 0.5
    distribution_per_income[1, 1] = 0.5
    distribution_per_income[2, 0] = 0.3
    distribution_per_income[2, 1] = 0.4
    distribution_per_income[2, 2] = 0.3

    distribution = np.zeros((3, 60), dtype=float)
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

    # Use the parametrized weight matrix
    monkeypatch.setattr(employment_opportunities, "get_weight_matrix", lambda *args, **kwargs: request.param)

    # Capture xlsx writes
    xlsx_writes = []

    def capture_write_xlsx(self, data, key, header=None):
        xlsx_writes.append({"data": data, "key": key, "header": header})

    monkeypatch.setattr(employment_opportunities.DataSource, "write_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(employment_opportunities.DataSource, "write_xlsx", capture_write_xlsx)

    class _Weights:
        def get(self, _key):
            return request.param

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

    potencies = employment_opportunities.employment_opportunities(config, _Weights(), _Weights())  # type: ignore

    return {
        "potencies": potencies,
        "xlsx_writes": xlsx_writes,
        "pod": pod,
        "motive": motive,
        "working_pop_income": working_pop_income,
        "jobs_income": jobs_income,
        "weight_matrix": request.param,
    }


# As defined in employment_opportunities
modalities = ["Fiets", "Auto", "OV", "Auto_Fiets", "OV_Fiets", "Auto_OV", "Auto_OV_Fiets"]


@pytest.mark.parametrize("modality", modalities)
@pytest.mark.parametrize(
    ("income_group", "income_index"), (("laag", 0), ("middellaag", 1), ("middelhoog", 2), ("hoog", 3))
)
def test_employment_opportunities_totals(modality, income_group, income_index, employment_opportunities_setup):
    """Reachable employment opportunities totals are independent of working population size and distribution over groups."""
    from ikob.datasource import DataKey

    potencies = employment_opportunities_setup["potencies"]
    pod = employment_opportunities_setup["pod"]
    motive = employment_opportunities_setup["motive"]
    jobs_income = employment_opportunities_setup["jobs_income"]
    weight_matrix = employment_opportunities_setup["weight_matrix"]

    key = DataKey(
        "Totaal",
        part_of_day=pod,
        income=income_group,
        group="alle groepen",
        motive=motive,
        modality=modality,
    )
    totals = potencies.get(key)

    # Intended behavior: each zone reaches its own jobs, multiplied by the weight
    expected_totaal = weight_matrix @ jobs_income[:, income_index]
    np.testing.assert_allclose(totals, expected_totaal)


@pytest.mark.parametrize("modality", modalities)
def test_employment_opportunities_ontpl_totaal(modality, employment_opportunities_setup):
    """Ontpl_totaal xlsx output shows reachability by income group, independent of distribution over groups."""

    xlsx_writes = employment_opportunities_setup["xlsx_writes"]
    working_pop_income = employment_opportunities_setup["working_pop_income"]
    jobs_income = employment_opportunities_setup["jobs_income"]
    weight_matrix = employment_opportunities_setup["weight_matrix"]

    # Test Ontpl_totaal xlsx write (per modality, showing reachability by income group)
    ontpl_totaal_writes = [w for w in xlsx_writes if w["key"].id == "Ontpl_totaal" and w["key"].modality == modality]
    assert len(ontpl_totaal_writes) == 1, f"Expected exactly one Ontpl_totaal write for modality {modality}"

    ontpl_totaal_data = ontpl_totaal_writes[0]["data"]
    # Ontpl_totaal is the Totaal values for each income group stacked by zone
    # Each Totaal[income_group] = weight_matrix @ jobs_income[:, income_group]
    ontpl_totaal_expected = np.array([weight_matrix @ jobs_income[:, i] for i in range(jobs_income.shape[1])]).T
    # Output files get rounded to integers for presentation
    np.testing.assert_allclose(ontpl_totaal_data, np.round(ontpl_totaal_expected))

    # Test Ontpl_totaalproduct xlsx write (product of reachability and population)
    ontpl_product_writes = [
        w for w in xlsx_writes if w["key"].id == "Ontpl_totaalproduct" and w["key"].modality == modality
    ]
    assert len(ontpl_product_writes) == 1, f"Expected exactly one Ontpl_totaalproduct write for modality {modality}"

    ontpl_product_data = ontpl_product_writes[0]["data"]
    # Product should be ontpl_totaal * working_population per zone and income
    # Output files get rounded to integers for presentation
    np.testing.assert_allclose(ontpl_product_data, np.round(ontpl_totaal_expected * working_pop_income))
