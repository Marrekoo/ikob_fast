import logging
import pathlib
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# ── Performance knobs ────────────────────────────────────────────────
DTYPE = np.float32          # float32 halves memory; float64 if precision needed
USE_SPARSE = True           # store weight matrices as scipy.sparse.csr_matrix
# ─────────────────────────────────────────────────────────────────────

IKOB_INFINITE = 9999.0


def zeros(lengte, dtype=None):
    return np.zeros(lengte, dtype=dtype or DTYPE)


def transpose(matrix):
    return np.asarray(matrix).T


def read_csv(filenaam, type_caster=float, has_index_column=True):
    if not isinstance(filenaam, pathlib.Path):
        filenaam = pathlib.Path(filenaam)

    try:
        matrix = np.loadtxt(filenaam, dtype=type_caster, delimiter=",")
        if has_index_column:
            logger.warning(f"Reading file {filenaam} without headers, but with an index column.")
    except ValueError:
        matrix = np.loadtxt(filenaam, dtype=type_caster, skiprows=1, delimiter=",")
    if has_index_column:
        _check_index_column(matrix, filenaam)
        matrix = matrix[:, 1:]
    if len(matrix.shape) == 2:
        if len(matrix[0, :]) == 1:
            return matrix[:, 0]
        if len(matrix[:, 0]) == 1:
            return matrix[0]
    return matrix


def read_csv_int(filenaam, has_index_column=True):
    return read_csv(filenaam, type_caster=int, has_index_column=has_index_column)


def read_csv_float(filenaam, has_index_column=True):
    return read_csv(filenaam, type_caster=float, has_index_column=has_index_column)


def _check_index_column(matrix: npt.NDArray, filenaam):
    index_column = matrix[:, 0]
    if len(matrix.shape) != 2:
        raise ValueError(
            f"Reading file {filenaam} as a file with index column, but the matrix it contains is not two dimensional."
        )
    prev_index = 0
    for idx in index_column:
        if abs(round(idx) - idx) > 1e-5:
            raise ValueError(f"Csv file {filenaam} has an invalid index column because index {idx} is not integer.")
        idx = int(round(idx))
        if idx - prev_index != 1:
            raise ValueError(f"Csv file {filenaam} has an invalid index column because the index is not sequential.")
        prev_index = idx


@dataclass
class CsvIndex:
    name: str = ""
    values: list[int] = field(default_factory=list)

    @classmethod
    def zone_index(cls, num_zones):
        return cls("zone", list(range(1, num_zones + 1)))


def write_csv(matrix, filenaam, index=CsvIndex(), header=[]):
    if not isinstance(filenaam, pathlib.Path):
        filenaam = pathlib.Path(filenaam)

    try:
        from scipy import sparse
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
    except ImportError:
        pass

    matrix = np.asarray(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, matrix.shape[0])

    is_int = np.isdtype(matrix.dtype, "integral")
    data_fmt = "%d" if is_int else "%.18e"

    has_index = len(index.values) > 0

    if has_index:
        index_col = np.array(index.values).reshape(-1, 1)
        matrix = np.hstack([index_col, matrix])
        header = [index.name, *header]
        fmt = ["%d"] + [data_fmt] * (matrix.shape[1] - 1)
    else:
        fmt = data_fmt

    delim = ","
    header_str = delim.join(header)

    # ── Fast path using pandas ──
    try:
        import pandas as pd

        float_fmt = None if is_int else "%.18e"
        df = pd.DataFrame(matrix)

        # Restore integer formatting for the index column (hstack upcasts to float)
        if has_index:
            df.iloc[:, 0] = df.iloc[:, 0].astype(int)

        with open(filenaam, "w", newline="") as f:
            if header_str:
                f.write(header_str + "\n")
            df.to_csv(f, index=False, header=False, float_format=float_fmt)
        return
    except ImportError:
        pass

    # ── Fallback using np.savetxt ──
    np.savetxt(filenaam, matrix, fmt=fmt, delimiter=delim, header=header_str, comments="")

def group_income_level(naam):
    if naam[-4:] == "hoog":
        if naam[-10:] == "middelhoog":
            return "middelhoog"
        else:
            return "hoog"
    elif naam[-4:] == "laag":
        if naam[-10:] == "middellaag":
            return "middellaag"
        else:
            return "laag"
    else:
        return ""


def find_preference(naam, mod):
    if "vk" in naam:
        Beginvk = naam.find("vk")
        if naam[Beginvk + 2] == "A":
            return "Auto"
        elif naam[Beginvk + 2] == "N":
            return "Neutraal"
        elif naam[Beginvk + 2] == "O":
            return "OV"
        elif naam[Beginvk + 2] == "F":
            return "Fiets"
        else:
            return ""
    elif "GratisAuto" in naam:
        if "GratisAuto_GratisOV" in naam and "OV" in mod and "Auto" in mod:
            return "Neutraal"
        else:
            if "Auto" in mod:
                return "Auto"
            else:
                return "OV"
    elif "GratisOV" in naam:
        return "OV"
    else:
        return ""


def single_group(mod, gr):
    if mod == "Auto":
        if "GratisAuto" in gr:
            return "GratisAuto"
        elif "Wel" in gr:
            return "Auto"
        if "GeenAuto" in gr:
            return "GeenAuto"
        if "GeenRijbewijs" in gr:
            return "GeenRijbewijs"
    if mod == "OV":
        if "GratisOV" in gr:
            return "GratisOV"
        else:
            return "OV"


def combined_group(mod, gr):
    string = ""
    if "Auto" in mod:
        if "GratisAuto" in gr:
            string = "GratisAuto"
        elif "Wel" in gr:
            string = "Auto"
        if "GeenAuto" in gr:
            string = "GeenAuto"
        if "GeenRijbewijs" in gr:
            string = "GeenRijbewijs"
    if "OV" in mod:
        if "GratisOV" in gr:
            string = string + "_GratisOV" if string else "GratisOV"
        else:
            string = string + "_OV" if string else "OV"
    if "EFiets" in mod:
        string = string + "_EFiets"
    elif "Fiets" in mod:
        string = string + "_Fiets"
    return string


# ── Vectorised generalized-travel-time helpers ──────────────────────
#
# Each "*_gtt" function collapses (time, money) into one generalized
# travel time via  gtt = time + tvom_factor * money.  The "*_time_money"
# companions expose the two components *before* that collapse, so a
# per-group tolerance curve (ikob.tolerance_curves) can be evaluated on
# them directly -- e.g. a custom tau (fixedVOT) or a genuine 2-D copula,
# neither of which can be reconstructed from the collapsed gtt alone.
# The "*_gtt" functions are thin wrappers so their output stays
# bit-for-bit identical to before this split.


def compute_bike_time_money(
    bike_time_matrix: npt.NDArray,
    bike_distance_matrix: npt.NDArray,
    bike_cost_euro_per_km: float,
):
    """(time [min], money [euro]) components of the bike GTT."""
    return bike_time_matrix, bike_distance_matrix * bike_cost_euro_per_km


def compute_bike_gtt(
    bike_time_matrix: npt.NDArray,
    bike_distance_matrix: npt.NDArray,
    bike_cost_euro_per_km: float,
    tvom_factor: float,
):
    t, m = compute_bike_time_money(bike_time_matrix, bike_distance_matrix, bike_cost_euro_per_km)
    return (t + tvom_factor * m).astype(DTYPE)


def compute_pt_time_money(pt_time_matrix: npt.NDArray, pt_cost_matrix: npt.NDArray):
    """(time [min], money [euro]) components of the PT GTT.

    Unreachable OD pairs (pt_time <= 0.5) get time = IKOB_INFINITE,
    money = 0, so that t + tau*m == IKOB_INFINITE regardless of tau --
    matching compute_pt_gtt's masking exactly.
    """
    t = np.where(pt_time_matrix > 0.5, pt_time_matrix, IKOB_INFINITE)
    m = np.where(pt_time_matrix > 0.5, pt_cost_matrix, 0.0)
    return t, m


def compute_pt_gtt(pt_time_matrix: npt.NDArray, pt_cost_matrix: npt.NDArray, tvom_factor: float):
    t, m = compute_pt_time_money(pt_time_matrix, pt_cost_matrix)
    return (t + tvom_factor * m).astype(DTYPE)


def compute_car_time_money(
    car_time: npt.NDArray,
    car_dist: npt.NDArray,
    var_rate: float,
    road_pricing: float,
    additional_costs_eurocent: npt.NDArray,
    parking_times_array: npt.NDArray,
    parking_costs_array_eurocent: npt.NDArray,
):
    """(time [min], money [euro]) components of the car GTT."""
    parking_time_matrix = parking_times_array[:, 0][:, np.newaxis] + parking_times_array[:, 1][np.newaxis, :]
    t = car_time + parking_time_matrix
    m = (
        (var_rate + road_pricing) * car_dist
        + additional_costs_eurocent / 100
        + parking_costs_array_eurocent / 100
    )
    return t, m


def compute_car_gtt(
    car_time: npt.NDArray,
    car_dist: npt.NDArray,
    var_rate: float,
    road_pricing: float,
    tvom_factor: float,
    additional_costs_eurocent: npt.NDArray,
    parking_times_array: npt.NDArray,
    parking_costs_array_eurocent: npt.NDArray,
):
    t, m = compute_car_time_money(
        car_time, car_dist, var_rate, road_pricing,
        additional_costs_eurocent, parking_times_array, parking_costs_array_eurocent,
    )
    return (t + tvom_factor * m).astype(DTYPE)


def compute_no_car_time_money(car_time_matrix, car_distance_matrix, time_cost_factor, var_cost_factor):
    """(time [min], money [euro]) components for the GeenAuto/GeenRijbewijs
    (car-sharing / taxi proxy) GTT formula in generalized_travel_time.py."""
    t = car_time_matrix
    m = car_time_matrix * time_cost_factor + car_distance_matrix * var_cost_factor
    return t, m


def compute_free_car_time_money(
    car_time_matrix, car_distance_matrix, road_pricing_electric,
    parking_time_matrix, parking_cost_array, additional_cost_matrix=None,
):
    """(time [min], money [euro]) components for the GratisAuto GTT formula."""
    t = car_time_matrix + parking_time_matrix
    m = car_distance_matrix * road_pricing_electric + parking_cost_array[np.newaxis, :] / 100
    if additional_cost_matrix is not None:
        m = m + additional_cost_matrix / 100
    return t, m


def costs_public_transport(distance, pt_km_price, starting_rate, pricecap, pricecap_value):
    distance = np.where(distance < 0, 0, distance)
    distance = starting_rate + distance * pt_km_price
    if pricecap:
        np.clip(distance, None, pricecap_value, out=distance)
    return distance

# ── Sparse helper ────────────────────────────────────────────────────

def maybe_to_sparse(arr):
    """Convert dense array to CSR sparse if USE_SPARSE is enabled."""
    if not USE_SPARSE:
        return arr
    try:
        from scipy.sparse import csr_matrix
        return csr_matrix(arr)
    except ImportError:
        return arr


def ensure_dense(arr):
    """Ensure array is dense (for operations that need it)."""
    try:
        from scipy import sparse
        if sparse.issparse(arr):
            return arr.toarray()
    except ImportError:
        pass
    return np.asarray(arr)


def sparse_maximum(a, b):
    """Element-wise maximum that works for both dense and sparse."""
    try:
        from scipy import sparse
        if sparse.issparse(a) and sparse.issparse(b):
            return a.maximum(b)
        if sparse.issparse(a):
            a = a.toarray()
        if sparse.issparse(b):
            b = b.toarray()
    except ImportError:
        pass
    return np.maximum(a, b)