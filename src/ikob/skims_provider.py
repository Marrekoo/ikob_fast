"""Pluggable sources for the base skims IKOB needs per part-of-day:
Auto_Tijd, Auto_Afstand, Fiets_Tijd, Fiets_Afstand, OV_Tijd, and either
OV_Kosten or OV_Afstand.

Every provider implements the same read(id, dagdeel, type_caster=float,
default=None, has_index_column=True) signature as
ikob.datasource.SkimsSource, so generalized_travel_time.py and
chain_generator.py work unchanged with whichever one the project
configuration selects (config["skims"]["skims_bron"]):

  - "bestanden": the original behaviour -- pre-computed CSV skims on disk.
  - "r5py":      compute time/distance matrices with the r5py routing
    engine (R5 on OpenStreetMap + GTFS). Requires the optional
    dependencies 'r5py', 'geopandas', and a Java 21 runtime. r5py's public
    API has shifted across releases; this adapter targets its documented
    core classes and fails loudly with an actionable message rather than
    silently producing wrong numbers if your installed version's return
    columns/enums differ.
  - "osrm":      query one or more running OSRM server(s) for car/bike
    skims (OSRM has no transit routing, so OV skims must still come from
    files or r5py).

Zone locations for route-based providers default to buurt centroids
derived from the *effective* (possibly study-area-filtered) zone set --
see ikob.geo_utils.effective_zone_order -- so a route-generation run
always uses exactly the same zone universe as ikob.datasource.SegsSource.
"""

import datetime as dt
import logging
import pathlib

import numpy as np
import numpy.typing as npt

from ikob.datasource import DataSourceError, SkimsSource
from ikob.utils import DTYPE

logger = logging.getLogger(__name__)


class SkimsProviderError(DataSourceError):
    pass


def get_skims_provider(config):
    """Factory: build whichever skims provider project.skims.skims_bron selects."""
    skims_config = config["skims"]
    bron = skims_config.get("skims_bron", "bestanden")
    if bron == "bestanden":
        return SkimsSource(config["project"]["paden"]["skims_directory"])
    if bron == "r5py":
        return R5PySkimsProvider(config)
    if bron == "osrm":
        return OsrmSkimsProvider(config)
    raise SkimsProviderError(
        f"Unknown skims_bron {bron!r}; expected 'bestanden', 'r5py', or 'osrm'."
    )


# ── shared helpers ───────────────────────────────────────────────────

_DAGDEEL_VELD = {
    "Ochtendspits": "vertrektijd_ochtendspits",
    "Restdag": "vertrektijd_restdag",
    "Avondspits": "vertrektijd_avondspits",
}


def _zone_locations_lonlat(config) -> npt.NDArray:
    """(N, 2) array of (lon, lat) per zone, in the project's *effective*
    zone order (ikob.geo_utils.effective_zone_order), for use as routing
    origins/destinations.

    Prefers an explicit skims.zone_locaties_bestand (csv with columns
    zone,lon,lat, or a gpkg layer named 'zones' with point geometry and a
    'buurtcode' column); falls back to buurt centroids derived from the
    SEGS GeoPackage's *effective* (possibly study-area-restricted) zone
    order when project.paden.segs_format is 'gpkg' -- this must match
    ikob.datasource._GpkgSegsInputBackend's zone order exactly, or SEGS
    data and route-generated skims would silently misalign.
    """
    from ikob import geo_utils

    override = config["skims"].get("zone_locaties_bestand", "")
    if override:
        override_path = pathlib.Path(override)
        if override_path.suffix.lower() == ".gpkg":
            gdf = geo_utils.read_layer(override_path, "zones")
            zone_order = geo_utils.canonical_zone_order(gdf["buurtcode"])
            return geo_utils.zone_centroids_lonlat(gdf, zone_order, "buurtcode")
        import pandas as pd
        df = pd.read_csv(override_path).sort_values("zone")
        return df[["lon", "lat"]].to_numpy(dtype=float)

    paden = config["project"]["paden"]
    if paden.get("segs_format", "csv") != "gpkg":
        raise SkimsProviderError(
            "Route-based skims need zone coordinates: either set "
            "skims.zone_locaties_bestand, or set project.paden.segs_format "
            "to 'gpkg' so buurt centroids can be used automatically."
        )
    if not paden.get("segs_bestand", ""):
        raise SkimsProviderError(
            "project.paden.segs_bestand is empty; cannot derive zone "
            "locations from a GeoPackage."
        )

    gpkg_path = pathlib.Path(paden["segs_bestand"])
    buurten_layer = paden.get("segs_buurten_laag", "buurten")
    buurtcode_column = paden.get("segs_buurtcode_kolom", "buurtcode")

    zone_order = geo_utils.effective_zone_order(config)
    raw = geo_utils.read_layer(gpkg_path, buurten_layer)
    gdf = geo_utils.reindex_gdf_to_zone_order(raw, zone_order, buurtcode_column)
    return geo_utils.zone_centroids_lonlat(gdf, zone_order, buurtcode_column)


# ── r5py ─────────────────────────────────────────────────────────────

class R5PySkimsProvider:
    """Generate time/distance skims on the fly with r5py."""

    def __init__(self, config):
        try:
            import r5py
        except ImportError as err:
            raise SkimsProviderError(
                "skims_bron is 'r5py' but the 'r5py' package (and a Java 21 "
                "runtime) is not available in this environment."
            ) from err
        self._r5py = r5py

        r5_config = config["skims"]["r5py"]
        osm_pbf = r5_config.get("osm_pbf", "")
        if not osm_pbf or not pathlib.Path(osm_pbf).exists():
            raise SkimsProviderError(f"skims.r5py.osm_pbf not found: '{osm_pbf}'.")

        gtfs_dir = r5_config.get("gtfs_directory", "")
        gtfs_files = sorted(pathlib.Path(gtfs_dir).glob("*.zip")) if gtfs_dir else []
        if gtfs_dir and not gtfs_files:
            logger.warning("skims.r5py.gtfs_directory '%s' contains no .zip GTFS feeds; "
                           "OV routing will have no transit component.", gtfs_dir)

        logger.info("Building r5py TransportNetwork from %s (+%d GTFS feed(s))…",
                   osm_pbf, len(gtfs_files))
        self._network = r5py.TransportNetwork(osm_pbf, [str(f) for f in gtfs_files])

        self._max_time = dt.timedelta(minutes=r5_config.get("max_reistijd_minuten", 180))
        self._vertrekdatum = r5_config.get("vertrekdatum", "2024-04-10")
        self._config = config
        self._zone_locations = None
        self._cache: dict[tuple[str, str], npt.NDArray] = {}

    def _points_gdf(self):
        if self._zone_locations is None:
            import geopandas as gpd
            from shapely.geometry import Point

            lonlat = _zone_locations_lonlat(self._config)
            self._zone_locations = gpd.GeoDataFrame(
                {"id": [str(i + 1) for i in range(len(lonlat))]},
                geometry=[Point(xy) for xy in lonlat],
                crs="EPSG:4326",
            )
        return self._zone_locations

    def _departure(self, dagdeel: str) -> dt.datetime:
        r5_config = self._config["skims"]["r5py"]
        field = _DAGDEEL_VELD.get(dagdeel)
        time_str = r5_config.get(field, "08:00") if field else "08:00"
        date = dt.date.fromisoformat(self._vertrekdatum)
        hour, minute = (int(part) for part in time_str.split(":"))
        return dt.datetime.combine(date, dt.time(hour=hour, minute=minute))

    def _transport_modes(self, mode: str):
        r5py = self._r5py
        try:  # recent r5py: one unified TransportMode enum
            modes = {
                "Auto": [r5py.TransportMode.CAR],
                "Fiets": [r5py.TransportMode.BICYCLE],
                "OV": [r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            }
        except AttributeError:  # older r5py: separate LegMode/TransitMode
            modes = {
                "Auto": [r5py.LegMode.CAR],
                "Fiets": [r5py.LegMode.BICYCLE],
                "OV": [r5py.TransitMode.TRANSIT, r5py.LegMode.WALK],
            }
        if mode not in modes:
            raise SkimsProviderError(f"Unknown mode {mode!r} requested from r5py provider.")
        return modes[mode]

    def _travel_times(self, mode: str, dagdeel: str) -> npt.NDArray:
        points = self._points_gdf()
        computer = self._r5py.TravelTimeMatrixComputer(
            self._network,
            origins=points,
            destinations=points,
            departure=self._departure(dagdeel),
            transport_modes=self._transport_modes(mode),
            max_time=self._max_time,
        )
        travel_times = computer.compute_travel_times()
        n = len(points)
        matrix = np.full((n, n), np.nan, dtype=DTYPE)
        id_to_idx = {row.id: i for i, row in points.iterrows()}
        for row in travel_times.itertuples(index=False):
            matrix[id_to_idx[row.from_id], id_to_idx[row.to_id]] = row.travel_time
        return matrix

    def _distances(self, mode: str, dagdeel: str) -> npt.NDArray:
        points = self._points_gdf()
        try:
            computer = self._r5py.DetailedItinerariesComputer(
                self._network,
                origins=points,
                destinations=points,
                departure=self._departure(dagdeel),
                transport_modes=self._transport_modes(mode),
                max_time=self._max_time,
            )
            details = computer.compute_travel_details()
        except Exception as err:
            raise SkimsProviderError(
                f"Failed to compute {mode} distances via r5py's "
                "DetailedItinerariesComputer; this API is version-sensitive. "
                "Consider supplying a pre-computed *_Afstand.csv skim instead."
            ) from err

        if "distance" not in details.columns:
            raise SkimsProviderError(
                "r5py's DetailedItinerariesComputer did not return a "
                "'distance' column with your installed r5py version; got "
                f"columns {list(details.columns)}. Consider supplying a "
                "pre-computed *_Afstand.csv skim instead."
            )

        n = len(points)
        matrix = np.zeros((n, n), dtype=DTYPE)
        id_to_idx = {row.id: i for i, row in points.iterrows()}
        grouped = details.groupby(["from_id", "to_id"])["distance"].sum()
        for (from_id, to_id), distance_m in grouped.items():
            matrix[id_to_idx[from_id], id_to_idx[to_id]] = distance_m / 1000.0  # m -> km
        return matrix

    def read(self, id: str, dagdeel: str, type_caster=float, default=None, has_index_column=True):
        cache_key = (id, dagdeel)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if id == "Auto_Tijd":
            result = self._travel_times("Auto", dagdeel)
        elif id == "Auto_Afstand":
            result = self._distances("Auto", dagdeel)
        elif id == "Fiets_Tijd":
            result = self._travel_times("Fiets", dagdeel)
        elif id == "Fiets_Afstand":
            result = self._distances("Fiets", dagdeel)
        elif id == "OV_Tijd":
            result = self._travel_times("OV", dagdeel)
        elif id == "OV_Afstand":
            result = self._distances("OV", dagdeel)
        elif id == "OV_Kosten":
            if default is None:
                raise SkimsProviderError(
                    "r5py has no notion of PT fares; provide 'OV kostenbestand' "
                    "as a CSV, or rely on OV kosten (starttarief/kmkosten) "
                    "applied to OV_Afstand instead."
                )
            return default
        else:
            raise SkimsProviderError(f"Unknown skim id {id!r} requested from r5py provider.")

        self._cache[cache_key] = result.astype(type_caster)
        return self._cache[cache_key]


# ── OSRM ─────────────────────────────────────────────────────────────

class OsrmSkimsProvider:
    """Query a running OSRM server's /table service for time+distance
    matrices. Car and bike only; OSRM does not support transit routing."""

    _PROFILE_SERVER_FIELD = {"Auto": ("auto_server", "driving"), "Fiets": ("fiets_server", "cycling")}

    def __init__(self, config):
        try:
            import requests
        except ImportError as err:
            raise SkimsProviderError(
                "skims_bron is 'osrm' but the 'requests' package is not available."
            ) from err
        self._requests = requests
        self._osrm_config = config["skims"]["osrm"]
        self._config = config
        self._cache: dict[str, tuple[npt.NDArray, npt.NDArray]] = {}

    def _table(self, mode: str):
        field, profile = self._PROFILE_SERVER_FIELD[mode]
        base_url = self._osrm_config.get(field, "")
        if not base_url:
            raise SkimsProviderError(f"skims.osrm.{field} is not configured.")

        lonlat = _zone_locations_lonlat(self._config)
        coords = ";".join(f"{lon},{lat}" for lon, lat in lonlat)
        url = f"{base_url.rstrip('/')}/table/v1/{profile}/{coords}"
        response = self._requests.get(url, params={"annotations": "duration,distance"}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "Ok":
            raise SkimsProviderError(f"OSRM /table request failed: {payload}")

        n = len(lonlat)
        durations = np.asarray(payload["durations"], dtype=DTYPE) / 60.0    # s -> min
        distances = np.asarray(payload["distances"], dtype=DTYPE) / 1000.0  # m -> km
        assert durations.shape == (n, n)
        return durations, distances

    def read(self, id: str, dagdeel: str, type_caster=float, default=None, has_index_column=True):
        mode = "Auto" if id.startswith("Auto") else "Fiets" if id.startswith("Fiets") else None
        if mode is None:
            if default is not None:
                return default
            raise SkimsProviderError(
                f"OSRM cannot provide skim {id!r} (OSRM has no transit routing); "
                "supply it as a file, or use skims_bron='r5py' for OV."
            )

        if mode not in self._cache:
            self._cache[mode] = self._table(mode)  # static road network: no dagdeel dependency
        durations, distances = self._cache[mode]

        if id.endswith("_Tijd"):
            return durations.astype(type_caster)
        return distances.astype(type_caster)