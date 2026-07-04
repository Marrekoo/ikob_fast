import enum
import logging
import os
import pathlib
import shutil
from dataclasses import dataclass, field
from typing import Type

import numpy.typing as npt
from numpy.typing import NDArray

import ikob.utils as utils
from ikob.urbanization_grade_to_parking_times import urbanization_grade_to_parking_times

logger = logging.getLogger(__name__)


class DataSourceError(Exception):
    pass


def get_project_name(config) -> str:
    return config["__filename__"]


def get_project_directory(config) -> pathlib.Path:
    paths = config["project"]["paden"]
    output_dir = pathlib.Path(paths["output_directory"])
    return output_dir / get_project_name(config)


def get_temporary_directory(config) -> pathlib.Path:
    project_dir = get_project_directory(config)
    return project_dir / "tussenresultaten"


def read_csv_from_config(config, key: str, id: str, type_caster=float, has_index_column=True):
    csv_path = config[key][id]
    if isinstance(csv_path, dict):
        csv_path = csv_path["bestand"]

    if csv_path == "":
        raise DataSourceError(
            f"Problem occurred while reading path from config with key: '{key}' and id '{id}'. Path is empty"
        )

    csv_path = pathlib.Path(csv_path)
    try:
        return utils.read_csv(csv_path, type_caster, has_index_column=has_index_column)
    except Exception:
        raise DataSourceError(
            f"Problem occurred while reading path from config with key: '{key}' and id '{id}'. Path is '{csv_path}'"
        )


def read_parking_times(config):
    config_skims = config["skims"]
    segs_dir = pathlib.Path(config["project"]["paden"]["segs_directory"])

    parking_time_path = pathlib.Path(config_skims.get("parkeerzoektijden_bestand", segs_dir / "Parkeerzoektijd.csv"))

    if parking_time_path.exists():
        logging.info("Reading parking times from: '%s'", parking_time_path)
        return utils.read_csv_int(parking_time_path)

    urbanization_path = segs_dir / "Stedelijkheidsgraad.csv"
    assert urbanization_path.exists(), (
        "Missing both Parkeerzoektijden, Stedelijkheidsgraad files."
    )

    logger.info("Generating parking times from '%s'", urbanization_path)
    urbanization_grade = utils.read_csv_int(urbanization_path)
    return urbanization_grade_to_parking_times(urbanization_grade)


class SkimsSource:
    def __init__(self, skims_dir: pathlib.Path | str):
        if skims_dir == "":
            raise DataSourceError("Skims source initialized with empty skims dir")
        self.skims_dir = pathlib.Path(skims_dir)

    def read(self, id: str, dagdeel: str, type_caster=float, default: npt.NDArray | None = None,
             has_index_column=True) -> npt.NDArray:
        path = (self.skims_dir / dagdeel / id).with_suffix(".csv")
        if os.path.exists(path):
            return utils.read_csv(path, type_caster=type_caster, has_index_column=has_index_column)
        if default is None:
            raise FileNotFoundError(f"Skim file {path} not found, with no default.")
        logger.warning(f"Skim file {path} not found, using default.")
        return default


class _CsvSegsInputBackend:
    """Legacy behaviour: one indexed CSV per raw input table."""

    def __init__(self, segs_dir: pathlib.Path):
        self.segs_dir = segs_dir

    def _dir(self, id, jaar, scenario):
        filename = id + jaar
        path = self.segs_dir / scenario
        os.makedirs(path, exist_ok=True)
        return path / filename

    def read(self, id, jaar, type_caster, scenario, has_index_column):
        path = self._dir(id, jaar, scenario).with_suffix(".csv")
        try:
            return utils.read_csv(path, type_caster=type_caster, has_index_column=has_index_column)
        except FileNotFoundError:
            raise DataSourceError(
                f"File SEGS file '{path}' not found. Is the scenario '{scenario}' correct?"
            )


class _GpkgSegsInputBackend:
    """Buurtcode-keyed GeoPackage backend for CBS opendata.

    Every table is read as a layer named *id* (optionally suffixed with
    the year/scenario, mirroring the old CSV naming convention), joined
    onto a canonical zone order derived once from the reference buurten
    geometry layer -- see ikob.geo_utils.
    """

    def __init__(self, config):
        from ikob import geo_utils  # local import: optional geopandas dependency

        self._geo_utils = geo_utils
        paden = config["project"]["paden"]

        gpkg_path = paden.get("segs_bestand", "")
        if not gpkg_path:
            raise DataSourceError(
                "project.paden.segs_format is 'gpkg' but project.paden.segs_bestand is empty."
            )
        self.gpkg_path = pathlib.Path(gpkg_path)
        self.buurtcode_column = paden.get("segs_buurtcode_kolom", "buurtcode")
        buurten_layer = paden.get("segs_buurten_laag", "buurten")

        reference = geo_utils.validate_cbs_buurt_layer(
            self.gpkg_path, buurten_layer, buurtcode_column=self.buurtcode_column,
        )
        self.zone_order = geo_utils.canonical_zone_order(reference[self.buurtcode_column])
        logger.info("SEGS GeoPackage %s: %d buurten found in reference layer '%s'.",
                   self.gpkg_path, len(self.zone_order), buurten_layer)

    def _resolve_layer_name(self, id, jaar, scenario):
        candidates = [f"{id}{jaar}_{scenario}", f"{id}_{scenario}", f"{id}{jaar}", id]
        available = set(self._geo_utils.list_layers(self.gpkg_path))
        for candidate in candidates:
            if candidate in available:
                return candidate
        raise DataSourceError(
            f"None of the candidate layers {candidates} were found in {self.gpkg_path} "
            f"for SEGS table '{id}'. Available layers: {sorted(available)}."
        )

    def read(self, id, jaar, type_caster, scenario, has_index_column):
        layer = self._resolve_layer_name(id, jaar, scenario)
        df = self._geo_utils.read_layer(self.gpkg_path, layer)
        result = self._geo_utils.reindex_to_zone_order(df, self.zone_order, self.buurtcode_column)
        return result.astype(type_caster)


class SegsSource:
    def __init__(self, config):
        paden = config["project"]["paden"]
        self._format = paden.get("segs_format", "csv")

        if self._format == "gpkg":
            self._input_backend = _GpkgSegsInputBackend(config)
            self.segs_dir = pathlib.Path(paden.get("segs_bestand", ""))
        else:
            self.segs_dir = pathlib.Path(paden["segs_directory"])
            if str(self.segs_dir) == "":
                raise DataSourceError("Skims source initialized with empty skims dir")
            self._input_backend = _CsvSegsInputBackend(self.segs_dir)

        self.tmp_dir = get_temporary_directory(config)

    def _segs_output_dir(self, id, jaar, scenario, group="", modifier=""):
        root = self.tmp_dir / "groepenverdeling"
        return self._segs_dir(root, id, jaar, scenario, group, modifier)

    def _segs_dir(self, path, id, jaar, scenario, group="", modifier=""):
        filename = id + jaar
        for postfix in [group, modifier]:
            if postfix:
                filename += f"_{postfix}"
        path = path / scenario
        os.makedirs(path, exist_ok=True)
        return path / filename

    def read(self, id: str, jaar="", type_caster: Type = int, scenario="", group="", modifier="",
             has_index_column=True):
        should_read_from_output = "Verdeling_over_groepen" in id
        if should_read_from_output:
            path = self._segs_output_dir(id=id, jaar=jaar, scenario=scenario, group=group, modifier=modifier)
            path = path.with_suffix(".csv")
            try:
                return utils.read_csv(path, type_caster=type_caster, has_index_column=has_index_column)
            except FileNotFoundError:
                raise DataSourceError(
                    f"File SEGS file '{path}' not found. Is the scenario '{scenario}' correct?"
                )

        # Raw input table: 'id' may still be a bare CSV-era filename like
        # "Beroepsbevolking_inkomensklasse.csv" (as configured under
        # project.motief); strip that suffix so gpkg layer lookups work
        # without editing existing config values when switching format.
        lookup_id = id[:-4] if id.lower().endswith(".csv") else id
        return self._input_backend.read(lookup_id, jaar, type_caster, scenario, has_index_column)

    def write_csv(self, data, id, header, group="", jaar="", modifier="", scenario="",
                  index: utils.CsvIndex = utils.CsvIndex()):
        path = self._segs_output_dir(id, jaar, scenario, group, modifier).with_suffix(".csv")
        return utils.write_csv(data, path, header=header, index=index)


class DataType(enum.Enum):
    DESTINATIONS = "bestemmingen"
    COMPETITION = "concurrentie"
    GENERALIZED_TRAVEL_TIME = "ervarenreistijd"
    WEIGHTS = "gewichten"
    ORIGINS = "inwoners"
    POTENCY = "potenties"


@dataclass(eq=True, frozen=True)
class DataKey:
    id: str
    part_of_day: str
    regime: str = ""
    subtopic: str = ""
    preference: str = ""
    income: str = ""
    hub_name: str = ""
    motive: str = ""
    group: str = ""
    modality: str = ""
    fuel_kind: str = ""

    header: list[str] = field(default_factory=list, compare=False)
    index: utils.CsvIndex = field(default_factory=utils.CsvIndex, compare=False)

    is_temporary: bool = field(default=False, compare=False)

    @staticmethod
    def zone_header(num_zones):
        return ["zone_" + str(i + 1) for i in range(num_zones)]

    @staticmethod
    def zone_index(num_zones):
        return utils.CsvIndex.zone_index(num_zones)


class DataSource:
    OUTPUT_PATH = "OUTPUT.md"

    def __init__(self, config, datatype: DataType):
        self.config = config
        self.project_dir = get_project_directory(config)
        self.cache: dict[DataKey, NDArray] = {}
        self.datatype = datatype

    def _add_id_suffix(self, key: DataKey) -> str:
        id = key.id + key.preference
        for suffix in [key.modality, key.hub_name, key.income]:
            if suffix:
                id += f"_{suffix}"
        return id

    def _make_file_path(self, key: DataKey) -> pathlib.Path:
        base = self._get_base_dir(key)
        id_with_suffix = self._add_id_suffix(key)
        dagdeel = key.part_of_day.lower()
        regime = key.regime.lower()
        path = (
            self.project_dir / base / regime / key.motive / key.group
            / self.datatype.value / key.subtopic / dagdeel / key.fuel_kind
        )
        os.makedirs(path, exist_ok=True)
        return path / id_with_suffix

    def _get_base_dir(self, key: DataKey) -> str:
        if self.datatype in [DataType.COMPETITION, DataType.ORIGINS]:
            return "resultaten"
        if "totaal" in key.id.lower():
            return "resultaten"
        return ""

    def set(self, key: DataKey, data):
        self.cache[key] = data

    def get(self, key: DataKey):
        if key in self.cache:
            return self.cache[key]
        data = self.read_csv(key)
        self.set(key, data)
        return data

    def store(self):
        logger.info("Writing output for data: %s.", self.datatype.value)
        for key, data in self.cache.items():
            self.write_csv(data, key)

    def read_csv(self, key: DataKey) -> NDArray:
        path = self._make_file_path(key).with_suffix(".csv")
        return utils.read_csv(path)

    def write_csv(self, data, key: DataKey, header=[]):
        assert isinstance(key, DataKey)
        if key.is_temporary:
            return
        path = self._make_file_path(key).with_suffix(".csv")
        if not header:
            header = key.header
        utils.write_csv(data, path, header=header, index=key.index)

    def get_write_path(self, key: DataKey) -> pathlib.Path:
        """Return the CSV path for *key*, creating directories as needed.

        Use this when you want to build the path in the calling thread
        (ensuring dirs exist) and then hand the actual write off to an
        ``AsyncCsvWriter``.
        """
        return self._make_file_path(key).with_suffix(".csv")

    # ── Memory management ────────────────────────────────────────────

    def clear_cache(self):
        """Release all cached arrays. Call after store() when data is no longer needed."""
        n = len(self.cache)
        self.cache.clear()
        logger.debug("Cleared %d entries from %s cache.", n, self.datatype.value)

    def clear_temporary(self):
        """Release only temporary (is_temporary=True) entries to free memory."""
        temp_keys = [k for k in self.cache if k.is_temporary]
        for k in temp_keys:
            del self.cache[k]
        if temp_keys:
            logger.debug("Cleared %d temporary entries from %s cache.", len(temp_keys), self.datatype.value)

    def cache_size_mb(self) -> float:
        """Estimate total memory held by cached arrays (MB)."""
        import sys
        total = 0
        for v in self.cache.values():
            try:
                total += v.nbytes
            except AttributeError:
                try:
                    from scipy import sparse
                    if sparse.issparse(v):
                        total += v.data.nbytes + v.indices.nbytes + v.indptr.nbytes
                except (ImportError, AttributeError):
                    total += sys.getsizeof(v)
        return total / (1024 * 1024)

    @staticmethod
    def write_output_md(config):
        path = get_project_directory(config) / DataSource.OUTPUT_PATH
        shutil.copy(DataSource.OUTPUT_PATH, path)