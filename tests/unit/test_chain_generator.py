import numpy as np

from ikob.chain_generator import Hubs, chain_generator, compute_chain_travel_time
from ikob.datasource import DataKey, DataSource, DataType


def _make_hubs(zones, hub_costs, pt_transfer, bike_transfer, pay_for_pt):
    """Build a Hubs object from plain lists."""
    data = np.column_stack([zones, hub_costs, pt_transfer, bike_transfer, pay_for_pt])
    return Hubs(data)


def _simple_matrices(n=4):
    """Return small deterministic skim matrices for *n* zones."""
    rng = np.random.default_rng(42)
    car_time = rng.random((n, n)) * 30
    car_dist = rng.random((n, n)) * 50
    bike_time = rng.random((n, n)) * 20
    bike_dist = rng.random((n, n)) * 15
    pt_time = rng.random((n, n)) * 25
    pt_dist = rng.random((n, n)) * 40
    return car_time, car_dist, bike_time, bike_dist, pt_time, pt_dist


def test_single_hub_basic():
    """With one hub the result equals that hub's contribution (where closer)."""
    n = 4
    car_time, car_dist, bike_time, bike_dist, pt_time, pt_dist = _simple_matrices(n)

    # Hub at zone 2 (1-indexed), modest costs
    hubs = _make_hubs(zones=[2], hub_costs=[100], pt_transfer=[5], bike_transfer=[3], pay_for_pt=[1])
    factor = 2.0
    var_car_rate = 0.05
    road_pricing = 0.01
    bike_cost_euro_per_km = 0.02

    # Pre-compute PT cost matrix (simple: no starting rate, no pricecap)
    pt_km_price = 0.10
    pt_cost = pt_dist * pt_km_price

    result_bike, result_ride = compute_chain_travel_time(
        hubs,
        car_time,
        car_dist,
        bike_time,
        bike_dist,
        pt_time,
        pt_cost,
        factor,
        var_car_rate,
        road_pricing,
        bike_cost_euro_per_km,
        additional_costs=np.zeros((n, n)),
        parking_times=np.zeros((n, 3)),
    )

    # With just a single hub, the resulting times should be equal to taking the car to the hub, and then either the bike or pt
    zone_with_hub = 2
    hub_cost = 100 / 100
    car_leg = car_time[:, zone_with_hub] + factor * (var_car_rate + road_pricing) * car_dist[:, zone_with_hub]

    expected_bike = (
        car_leg[:, np.newaxis]
        + bike_time[zone_with_hub, :][np.newaxis, :]
        + 3  # bike_transfer
        + factor * bike_cost_euro_per_km * bike_dist[zone_with_hub, :][np.newaxis, :]
        + factor * hub_cost
    )
    expected_ride = (
        car_leg[:, np.newaxis]
        + np.where(
            pt_time[zone_with_hub, :] > 0.5,
            pt_time[zone_with_hub, :] + factor * pt_cost[zone_with_hub, :] * 1,  # pay_for_pt=1
            9999,
        )[np.newaxis, :]
        + 5  # pt_transfer
        + factor * hub_cost
    )

    np.testing.assert_allclose(result_bike, expected_bike)
    np.testing.assert_allclose(result_ride, expected_ride)


def test_minimum_across_hubs():
    """With two hubs, element-wise minimum is taken correctly."""
    n = 4
    car_time, car_dist, bike_time, bike_dist, pt_time, pt_dist = _simple_matrices(n)

    hubs = _make_hubs(
        zones=[1, 3],
        hub_costs=[50, 80],
        pt_transfer=[4, 6],
        bike_transfer=[2, 3],
        pay_for_pt=[1, 1],
    )

    pt_cost = pt_dist * 0.08

    result_bike, result_ride = compute_chain_travel_time(
        hubs,
        car_time,
        car_dist,
        bike_time,
        bike_dist,
        pt_time,
        pt_cost,
        factor=1.5,
        var_car_rate=0.04,
        road_pricing=0.02,
        bike_cost_euro_per_km=0.02,
        additional_costs=np.zeros((n, n)),
        parking_times=np.zeros((n, 3)),
    )

    # Result should be <= single-hub results
    for zone_col in [0, 2]:  # hub zones 1 and 3 (0-indexed)
        single = _make_hubs(
            zones=[zone_col + 1],
            hub_costs=[50 if zone_col == 0 else 80],
            pt_transfer=[4 if zone_col == 0 else 6],
            bike_transfer=[2 if zone_col == 0 else 3],
            pay_for_pt=[1],
        )
        sb, sr = compute_chain_travel_time(
            single,
            car_time,
            car_dist,
            bike_time,
            bike_dist,
            pt_time,
            pt_cost,
            factor=1.5,
            var_car_rate=0.04,
            road_pricing=0.02,
            bike_cost_euro_per_km=0.02,
            additional_costs=np.zeros((n, n)),
            parking_times=np.zeros((n, 3)),
        )
        assert np.all(result_bike <= sb + 1e-10)
        assert np.all(result_ride <= sr + 1e-10)


def _make_config():
    return {
        "project": {
            "beprijzingsregime": "basis",
            "motief": {"naam": "woon-werk", "TVOM": "werk"},
            "paden": {
                "output_directory": "out",
                "skims_directory": "skims",
                "segs_directory": "segs",
            },
        },
        "skims": {
            "Kosten auto fossiele brandstof": {"variabele kosten": 5, "kmheffing": 1},
            "Kosten elektrische auto": {"variabele kosten": 3, "kmheffing": 1},
            "OV kosten": {"kmkosten": 10, "starttarief": 100},
            "OV kostenbestand": {"gebruiken": False},
            "pricecap": {"gebruiken": False, "getal": 0},
            "bike_cost_ct_per_km": 2,
            "dagsoort": ["restdag"],
            "parkeerzoektijden_bestand": "unused.csv",
        },
        "TVOM": {
            "werk": {"laag": 0.9, "middellaag": 1.5, "middelhoog": 2.0, "hoog": 2.5},
            "overig": {"laag": 0.8, "middellaag": 1.2, "middelhoog": 1.6, "hoog": 2.0},
        },
        "ketens": {
            "chains": {
                "gebruiken": True,
                "naam hub": "test_hubset",
            },
            "bestemmingslijst": {
                "gebruiken": False,
            },
        },
        "geavanceerd": {
            "additionele_kosten": {"gebruiken": False, "bestand": ""},
            "parkeerkosten": {"gebruiken": False, "bestand": ""},
        },
        "__filename__": "tmp",
    }


def test_chain_generator(monkeypatch):
    """chain_generator populates the DataSource with P+Bike and P+R keys."""
    import ikob.chain_generator as cg

    num_zones = 5
    rng = np.random.default_rng(99)

    skims_data = {
        ("Auto_Tijd", "restdag"): rng.random((num_zones, num_zones)) * 30,
        ("Auto_Afstand", "restdag"): rng.random((num_zones, num_zones)) * 30,
        ("Fiets_Tijd", "restdag"): rng.random((num_zones, num_zones)) * 30,
        ("OV_Tijd", "restdag"): rng.random((num_zones, num_zones)) * 30,
        ("OV_Afstand", "restdag"): rng.random((num_zones, num_zones)) * 30,
    }

    def fake_skims_source(_skims_dir):
        class _Reader:
            def read(self, id, dagdeel, type_caster=float, default=None, has_index_column=False):
                key = (id, dagdeel)
                if key in skims_data:
                    return np.array(skims_data[key], dtype=type_caster)
                if default is not None:
                    return default
                raise FileNotFoundError(f"Skim {id}/{dagdeel} not found, with no default.")

        return _Reader()

    hub_data = np.array([[2, 50, 5, 3, 1], [4, 80, 6, 4, 1]], dtype=float)

    def fake_read_csv_from_config(config, key, id, type_caster=float, has_index_column=False):
        if key == "ketens" and id == "chains":
            return hub_data
        raise AssertionError(f"Unexpected read_csv_from_config call: key={key!r}, id={id!r}")

    monkeypatch.setattr(cg, "SkimsSource", fake_skims_source)
    monkeypatch.setattr(cg, "read_csv_from_config", fake_read_csv_from_config)
    monkeypatch.setattr(cg, "read_parking_times", lambda _config: np.zeros((num_zones, 3)))

    config = _make_config()

    datasource = DataSource(config, DataType.GENERALIZED_TRAVEL_TIME)
    datasource.cache = {}

    chain_generator(datasource, config)

    # Expect keys for 4 income levels x 2 fuel kinds x {Pplusfiets, PplusR} = 16 keys
    assert len(datasource.cache) == 16
    for income in ["laag", "middellaag", "middelhoog", "hoog"]:
        for fuel in ["fossiel", "elektrisch"]:
            for prefix in ["Pplusfiets", "PplusR"]:
                key = DataKey(
                    id=f"{prefix}_{fuel}",
                    part_of_day="restdag",
                    income=income,
                    hub_name="test_hubset",
                    motive="woon-werk",
                    regime="basis",
                )
                assert key in datasource.cache, f"Missing key: {key}"
                matrix = datasource.cache[key]
                assert matrix.shape == (num_zones, num_zones)
