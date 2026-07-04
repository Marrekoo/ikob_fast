import numpy as np
import pandas as pd
import pytest

from ikob.geo_utils import GeoDataError, canonical_zone_order, reindex_to_zone_order


def test_canonical_zone_order_is_sorted_and_one_based():
    order = canonical_zone_order(["BU0003", "BU0001", "BU0002", "BU0001"])
    assert order == {"BU0001": 1, "BU0002": 2, "BU0003": 3}


def test_reindex_to_zone_order_reorders_rows():
    order = canonical_zone_order(["B", "A", "C"])
    df = pd.DataFrame({"buurtcode": ["C", "A", "B"], "waarde": [30, 10, 20]})
    result = reindex_to_zone_order(df, order, "buurtcode", value_columns=["waarde"])
    np.testing.assert_array_equal(result, [10, 20, 30])


def test_reindex_to_zone_order_squeezes_single_column():
    order = canonical_zone_order(["A", "B"])
    df = pd.DataFrame({"buurtcode": ["A", "B"], "waarde": [1, 2]})
    result = reindex_to_zone_order(df, order, "buurtcode")
    assert result.ndim == 1


def test_reindex_to_zone_order_rejects_missing_buurtcode():
    order = canonical_zone_order(["A", "B", "C"])
    df = pd.DataFrame({"buurtcode": ["A", "B"], "waarde": [1, 2]})
    with pytest.raises(GeoDataError, match="missing"):
        reindex_to_zone_order(df, order, "buurtcode", value_columns=["waarde"])


def test_reindex_to_zone_order_rejects_unknown_buurtcode():
    order = canonical_zone_order(["A", "B"])
    df = pd.DataFrame({"buurtcode": ["A", "B", "X"], "waarde": [1, 2, 3]})
    with pytest.raises(GeoDataError, match="not present"):
        reindex_to_zone_order(df, order, "buurtcode", value_columns=["waarde"])


def test_reindex_to_zone_order_rejects_duplicate_buurtcode():
    order = canonical_zone_order(["A"])
    df = pd.DataFrame({"buurtcode": ["A", "A"], "waarde": [1, 2]})
    with pytest.raises(GeoDataError, match="duplicate"):
        reindex_to_zone_order(df, order, "buurtcode", value_columns=["waarde"])


def test_validate_cbs_buurt_layer_gpkg_roundtrip(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    from ikob.geo_utils import validate_cbs_buurt_layer

    gdf = gpd.GeoDataFrame(
        {"buurtcode": ["BU0001", "BU0002"], "inwoners": [100, 200]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:28992",
    )
    path = tmp_path / "test.gpkg"
    gdf.to_file(path, layer="buurten", driver="GPKG")

    result = validate_cbs_buurt_layer(path, "buurten", required_columns=("inwoners",))
    assert list(result["buurtcode"]) == ["BU0001", "BU0002"]


def test_validate_cbs_buurt_layer_rejects_duplicate_buurtcode(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    from ikob.geo_utils import GeoDataError, validate_cbs_buurt_layer

    gdf = gpd.GeoDataFrame(
        {"buurtcode": ["BU0001", "BU0001"]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:28992",
    )
    path = tmp_path / "test.gpkg"
    gdf.to_file(path, layer="buurten", driver="GPKG")

    with pytest.raises(GeoDataError, match="duplicate"):
        validate_cbs_buurt_layer(path, "buurten")