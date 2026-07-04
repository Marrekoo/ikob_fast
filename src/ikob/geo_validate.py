"""Additive input-file validation for GeoPackage-based SEGS input and for
the r5py/OSRM skim-generation backends.

Deliberately self-contained -- it does not modify
ikob.config.validate.FileValidator -- but is designed to be called from
it. Wire it in with, e.g.:

    # inside FileValidator.validate_input_files():
    from ikob.config.geo_validate import validate_geo_inputs
    valid = valid and validate_geo_inputs(self.config)

validate_geo_inputs() returns True/False and only *logs* problems, since
FileValidator.validate_input_files() is expected to collect every problem
before the caller decides whether to abort (mirroring the warn-and-report
convention used elsewhere, e.g.
distribute_over_groups._validate_car_possession_segs).
"""

import logging
import pathlib

logger = logging.getLogger(__name__)


def _validate_segs_gpkg(config) -> bool:
    from ikob import geo_utils

    paden = config["project"]["paden"]
    gpkg_path = paden.get("segs_bestand", "")
    if not gpkg_path:
        logger.error("project.paden.segs_format is 'gpkg' but segs_bestand is empty.")
        return False

    buurten_layer = paden.get("segs_buurten_laag", "buurten")
    buurtcode_column = paden.get("segs_buurtcode_kolom", "buurtcode")

    try:
        geo_utils.validate_cbs_buurt_layer(gpkg_path, buurten_layer, buurtcode_column)
    except geo_utils.GeoDataError as err:
        logger.error("SEGS GeoPackage validation failed: %s", err)
        return False

    logger.info("SEGS GeoPackage %s: reference layer '%s' OK.", gpkg_path, buurten_layer)
    return True


def _validate_r5py_inputs(config) -> bool:
    r5_config = config["skims"].get("r5py", {})
    ok = True

    osm_pbf = r5_config.get("osm_pbf", "")
    if not osm_pbf or not pathlib.Path(osm_pbf).exists():
        logger.error("skims.r5py.osm_pbf not found: '%s'.", osm_pbf)
        ok = False

    gtfs_dir = r5_config.get("gtfs_directory", "")
    if gtfs_dir:
        gtfs_files = list(pathlib.Path(gtfs_dir).glob("*.zip"))
        if not gtfs_files:
            logger.warning("skims.r5py.gtfs_directory '%s' has no .zip GTFS feeds; "
                           "OV routing will be walk-only.", gtfs_dir)

    try:
        import datetime as dt
        dt.date.fromisoformat(r5_config.get("vertrekdatum", ""))
    except ValueError:
        logger.error("skims.r5py.vertrekdatum %r is not a valid ISO date (JJJJ-MM-DD).",
                     r5_config.get("vertrekdatum"))
        ok = False

    try:
        import r5py  # noqa: F401
    except ImportError:
        logger.error("skims_bron is 'r5py' but the 'r5py' package is not installed.")
        ok = False

    return ok


def _validate_osrm_inputs(config) -> bool:
    osrm_config = config["skims"].get("osrm", {})
    for field in ("auto_server", "fiets_server"):
        url = osrm_config.get(field, "")
        if not url:
            logger.warning("skims.osrm.%s is empty; that mode's skims must come from a file.", field)
            continue
        try:
            import requests
            requests.get(url, timeout=3)
        except Exception as err:
            # Soft warning only: the OSRM server may simply not be running
            # yet at config-validation time.
            logger.warning("Could not reach OSRM server %s (%s): %s", field, url, err)
    return True


def validate_geo_inputs(config) -> bool:
    """Validate whichever GeoPackage/skim-generation inputs the project
    configuration selects. Returns True iff no hard errors were found."""
    ok = True

    if config["project"]["paden"].get("segs_format", "csv") == "gpkg":
        ok = _validate_segs_gpkg(config) and ok

    skims_bron = config["skims"].get("skims_bron", "bestanden")
    if skims_bron == "r5py":
        ok = _validate_r5py_inputs(config) and ok
    elif skims_bron == "osrm":
        ok = _validate_osrm_inputs(config) and ok

    return ok