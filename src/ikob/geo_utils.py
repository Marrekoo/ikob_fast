"""GeoPackage support for buurt-level (CBS opendata) input.

CBS ("Centraal Bureau voor de Statistiek") distributes neighbourhood-level
("buurt") statistics and geometry as GeoPackages. This module lets IKOB
read SEGS input tables from such a GeoPackage instead of a directory of
index-column CSVs.

Design: rather than relying on IKOB's historical assumption that every
input CSV lists zones in the exact same, otherwise-unchecked order (see
utils._check_index_column), every table in the GeoPackage is joined on an
explicit buurtcode column. A canonical zone order is derived once, by
sorting the buurtcodes found in the reference geometry layer, and every
other layer is reindexed onto that same order -- with a hard failure if a
layer's buurtcodes don't exactly match the reference set.
"""

import logging
import pathlib

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the dependency
    gpd = None
    GEOPANDAS_AVAILABLE = False


class GeoDataError(ValueError):
    """Raised for malformed or inconsistent GeoPackage input."""


def _require_geopandas():
    if not GEOPANDAS_AVAILABLE:
        raise GeoDataError(
            "GeoPackage support requires the 'geopandas' package (and a "
            "working pyogrio/GDAL install). Install it, e.g. via "
            "'pip install geopandas', or switch project.paden.segs_format "
            "back to 'csv'."
        )


def list_layers(gpkg_path) -> list[str]:
    """List every layer name present in a GeoPackage."""
    _require_geopandas()
    gpkg_path = pathlib.Path(gpkg_path)
    if not gpkg_path.exists():
        raise GeoDataError(f"GeoPackage not found: '{gpkg_path}'.")
    try:
        return list(gpd.list_layers(gpkg_path)["name"])
    except AttributeError:
        import fiona
        return list(fiona.listlayers(gpkg_path))


def read_layer(gpkg_path, layer: str):
    """Read one layer of a GeoPackage, failing fast with a clear error if
    it's missing. Falls back to a plain-table reader for non-spatial
    (attribute-only) GeoPackage layers, e.g. IKOB's own derived SEGS
    tables (WelAuto, GeenAuto, Voorkeuren, ...) that carry no geometry."""
    _require_geopandas()
    gpkg_path = pathlib.Path(gpkg_path)
    if not gpkg_path.exists():
        raise GeoDataError(f"GeoPackage not found: '{gpkg_path}'.")
    available = list_layers(gpkg_path)
    if layer not in available:
        raise GeoDataError(
            f"Layer '{layer}' not found in {gpkg_path}. Available layers: {available}."
        )
    try:
        return gpd.read_file(gpkg_path, layer=layer)
    except Exception:
        import pyogrio
        return pyogrio.read_dataframe(gpkg_path, layer=layer)


def layer_columns(gpkg_path, layer: str, sample_rows: int = 5):
    """Return (column_names, dtype_by_column, sample_dataframe) for a
    layer, reading only a handful of rows where the backend supports it
    -- useful for populating a column-mapping UI on large layers without
    waiting for a full read."""
    _require_geopandas()
    try:
        sample = gpd.read_file(gpkg_path, layer=layer, rows=sample_rows)
    except TypeError:
        # Older geopandas/pyogrio without 'rows' support: fall back to a
        # full read (still fine for buurt-sized layers).
        sample = read_layer(gpkg_path, layer)
    columns = [c for c in sample.columns if c != "geometry"]
    dtypes = {c: str(sample[c].dtype) for c in columns}
    return columns, dtypes, sample


def validate_cbs_buurt_layer(
    gpkg_path,
    layer: str,
    buurtcode_column: str = "buurtcode",
    required_columns: tuple[str, ...] = (),
):
    """Read and validate a CBS-style buurt geometry layer.

    Checks (all fail fast with GeoDataError):
    - the layer exists and is non-empty,
    - *buurtcode_column* and every column in *required_columns* is present,
    - buurtcode values are unique (no buurt listed twice),
    - every feature has valid, non-empty geometry,
    - the layer has a defined CRS (needed for correct centroids).

    Returns the GeoDataFrame on success.
    """
    gdf = read_layer(gpkg_path, layer)

    if len(gdf) == 0:
        raise GeoDataError(f"Layer '{layer}' in {gpkg_path} is empty.")

    missing = [c for c in (buurtcode_column, *required_columns) if c not in gdf.columns]
    if missing:
        raise GeoDataError(
            f"Layer '{layer}' in {gpkg_path} is missing required column(s) {missing}. "
            f"Available columns: {list(gdf.columns)}."
        )

    codes = gdf[buurtcode_column].astype(str)
    duplicates = sorted(codes[codes.duplicated()].unique())
    if duplicates:
        raise GeoDataError(
            f"Layer '{layer}' in {gpkg_path} has duplicate {buurtcode_column} "
            f"value(s): {duplicates}."
        )

    if "geometry" not in gdf.columns:
        raise GeoDataError(f"Layer '{layer}' in {gpkg_path} has no geometry column.")
    if gdf.geometry.isna().any() or (~gdf.geometry.is_valid).any():
        bad = sorted(codes[gdf.geometry.isna() | ~gdf.geometry.is_valid].tolist())
        raise GeoDataError(
            f"Layer '{layer}' in {gpkg_path} has missing/invalid geometry for "
            f"{buurtcode_column}(s): {bad}."
        )

    if gdf.crs is None:
        raise GeoDataError(
            f"Layer '{layer}' in {gpkg_path} has no coordinate reference system "
            "(CRS) set; cannot reliably compute centroids or reproject it."
        )

    return gdf


def canonical_zone_order(buurtcodes) -> dict[str, int]:
    """Assign a stable, sequential 1-based zone index to a set of buurtcodes.

    Sorting lexicographically on the buurtcode string makes the mapping
    fully deterministic and reproducible across runs and across every
    layer in the GeoPackage, replacing IKOB's old implicit "just list
    everything in matching row order" convention.
    """
    unique_codes = sorted({str(c) for c in buurtcodes})
    return {code: i + 1 for i, code in enumerate(unique_codes)}


def effective_zone_order(config) -> dict[str, int]:
    """The canonical zone order this project actually models: every
    buurtcode in the reference buurten layer, optionally restricted to a
    selected study area (project.paden.studiegebied), if one is set.

    Shared by ikob.datasource._GpkgSegsInputBackend and
    ikob.skims_provider so both always agree on which zones exist and in
    which order -- a mismatch there would silently misalign SEGS data
    against skims.
    """
    paden = config["project"]["paden"]
    gpkg_path = paden["segs_bestand"]
    buurten_layer = paden.get("segs_buurten_laag", "buurten")
    buurtcode_column = paden.get("segs_buurtcode_kolom", "buurtcode")
    reference = validate_cbs_buurt_layer(gpkg_path, buurten_layer, buurtcode_column)

    studiegebied = paden.get("studiegebied", {}) or {}
    zone_codes = studiegebied.get("zone_codes", [])
    if zone_codes:
        available = set(reference[buurtcode_column].astype(str))
        missing = sorted(set(zone_codes) - available)
        if missing:
            raise GeoDataError(
                f"Studiegebied bevat buurtcode(s) die niet (meer) in de "
                f"referentielaag voorkomen: {missing}."
            )
        return canonical_zone_order(zone_codes)
    return canonical_zone_order(reference[buurtcode_column])


def reindex_to_zone_order(
    df,
    zone_order: dict[str, int],
    buurtcode_column: str,
    value_columns: list[str] | None = None,
) -> npt.NDArray:
    """Reorder *df* onto *zone_order* and return a plain numpy array.

    Every buurtcode in *zone_order* must appear in *df* exactly once, and
    *df* must not contain any buurtcode outside *zone_order* -- both are
    treated as configuration errors (mismatched input files) rather than
    silently ignored, mirroring utils._check_index_column's fail-fast
    philosophy for the legacy CSV format. Use reindex_gdf_to_zone_order
    instead when the source layer legitimately contains zones outside
    the modelled set (e.g. the full-country buurten reference layer with
    a narrower studiegebied selected).
    """
    codes = df[buurtcode_column].astype(str)

    extra = sorted(set(codes) - set(zone_order))
    if extra:
        raise GeoDataError(
            f"Column '{buurtcode_column}' contains buurtcode(s) not present in "
            f"the reference buurten layer: {extra}."
        )
    missing = sorted(set(zone_order) - set(codes))
    if missing:
        raise GeoDataError(
            f"Column '{buurtcode_column}' is missing buurtcode(s) that are "
            f"present in the reference buurten layer: {missing}."
        )
    duplicates = sorted(codes[codes.duplicated()].unique())
    if duplicates:
        raise GeoDataError(
            f"Column '{buurtcode_column}' has duplicate buurtcode(s): {duplicates}."
        )

    ordered = df.set_index(codes.values)
    ordered = ordered.loc[sorted(zone_order, key=zone_order.get)]

    if value_columns is None:
        value_columns = [c for c in df.columns if c not in (buurtcode_column, "geometry")]

    result = ordered[value_columns].to_numpy()
    if result.shape[1] == 1:
        return result[:, 0]
    return result


def reindex_gdf_to_zone_order(gdf, zone_order: dict[str, int], buurtcode_column: str):
    """Like reindex_to_zone_order, but returns the filtered/reordered
    GeoDataFrame itself (every original column, including geometry)
    rather than a single value array, and *filters out* rows outside
    zone_order instead of raising on them.

    Used when a single layer (e.g. the full-country CBS buurten
    reference layer) legitimately contains more zones than the modelled
    set, and multiple different columns of it need to be resolved
    against the same canonical, filtered zone order -- see
    ikob.segs_mapping.
    """
    codes = gdf[buurtcode_column].astype(str)
    mask = codes.isin(zone_order)
    filtered = gdf[mask].copy()
    filtered_codes = codes[mask]

    missing = sorted(set(zone_order) - set(filtered_codes))
    if missing:
        raise GeoDataError(
            f"Kolom '{buurtcode_column}' mist buurtcode(s) die in het "
            f"gemodelleerde zonegebied voorkomen: {missing}."
        )
    duplicates = sorted(filtered_codes[filtered_codes.duplicated()].unique())
    if duplicates:
        raise GeoDataError(f"Kolom '{buurtcode_column}' heeft dubbele buurtcode(s): {duplicates}.")

    ordered = filtered.set_index(filtered_codes.values)
    return ordered.loc[sorted(zone_order, key=zone_order.get)]


def zone_centroids_lonlat(
    gdf,
    zone_order: dict[str, int],
    buurtcode_column: str = "buurtcode",
) -> npt.NDArray:
    """Return an (N, 2) array of (lon, lat) centroids, ordered by
    *zone_order*, reprojected to EPSG:4326 (WGS84) -- the format most
    routing engines (e.g. r5py) expect for origin/destination points."""
    _require_geopandas()
    projected = gdf.to_crs(gdf.estimate_utm_crs()) if gdf.crs is not None else gdf
    centroids_4326 = projected.geometry.centroid.to_crs(epsg=4326)
    codes = gdf[buurtcode_column].astype(str)
    order = sorted(zone_order, key=zone_order.get)
    lookup = dict(zip(codes, zip(centroids_4326.x, centroids_4326.y)))
    missing = [c for c in order if c not in lookup]
    if missing:
        raise GeoDataError(f"No centroid could be computed for buurtcode(s): {missing}.")
    return np.array([lookup[c] for c in order], dtype=float)


def dissolve_and_select(gdf, selected_mask, buffer_km: float, buurtcode_column: str = "buurtcode"):
    """Given a boolean mask selecting the 'core' study-area zones in
    *gdf*, return (core_codes, buffer_codes): the buurtcode of every
    selected zone, and of every *additional* zone whose centroid falls
    within buffer_km of the dissolved union of the selected zones'
    geometries.

    Membership is decided using zone centroids rather than full polygon
    overlap: a small, deliberate simplification that trades a bit of
    edge-case precision (a zone whose centroid falls just outside the
    buffer but whose polygon still clips it will be excluded, and vice
    versa) for a rule that is simple to state and to reproduce.
    """
    _require_geopandas()
    if not selected_mask.any():
        raise GeoDataError("Geen zones geselecteerd voor het studiegebied.")

    projected = gdf.to_crs(gdf.estimate_utm_crs())
    core = projected[selected_mask]
    try:
        dissolved = core.geometry.union_all()
    except AttributeError:
        dissolved = core.geometry.unary_union
    buffered = dissolved.buffer(buffer_km * 1000.0)

    centroids = projected.geometry.centroid
    within_buffer = centroids.within(buffered)

    codes = gdf[buurtcode_column].astype(str)
    core_codes = sorted(codes[selected_mask].tolist())
    buffer_codes = sorted(codes[within_buffer & ~selected_mask].tolist())
    return core_codes, buffer_codes