"""
Two-dimensional (time x money) tolerance curves for generalized costs.

Microfoundation (frailty/reliability theory): the decay weight is the
survival probability of a latent tolerance budget. Two coupling modes:

- "fixedVOT":  W(t, m) = scaling * S_T(t + tau * m)
  One shared budget; time and money are perfect substitutes. With a
  logistic S_T and the parameters from ikob.constants this reproduces
  the current pipeline (D1 + D2) exactly -- see legacy_spec().

- "copula":    W(t, m) = scaling * exp(-[(-ln S_T)^theta + (-ln S_M)^theta]^(1/theta))
  Gumbel-Hougaard shared frailty. theta = 1 is the fully independent
  series system S_T(t) * S_M(m); theta -> inf is the weakest link
  min(S_T, S_M). Note fixed VOT is NOT the theta -> inf limit; it is a
  structurally different mode (aggregate-then-decay).

Identifiability caveat: with exponential marginals, independence and
fixed VOT coincide exactly (see tests). Distinguishing the modes
empirically requires non-constant hazards.

Curve families
--------------
Beyond the four "smooth" survival families (logistic, weibull,
exponential, loglogistic), three piecewise families are available for
representing genuinely discontinuous or bounded tolerance budgets, ported
from the browser-based prototype editor (TolerancePlaneEditor.jsx):

- "step":       a deterministic hazard atom at c1 (mass 1 - p) followed
                by an optional plateau at level p until a final cutoff
                atom at c2. p = 0 reproduces a single hard cutoff at c1.
- "uniform":    a piecewise-linear survival function (uniform density)
                between an onset L (full tolerance) and a cutoff U (zero
                tolerance).
- "triangular": a piecewise-quadratic survival function (triangular
                density) between L and U, with a movable mode fraction
                that skews the curve convex or concave.

These piecewise families place probability atoms at hard thresholds, so
under the Gumbel copula they can make theta structurally unidentifiable
when *both* marginals are of this kind -- see MarginalCurve.is_deterministic().
"""

import json
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ikob.utils import DTYPE, maybe_to_sparse

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_TIME_MINUTES = 180.0   # IKOB hard cutoff (see single_weights.calculate_weights)
WEIGHT_EPSILON = 0.001     # weights below this are zeroed (idem)

CURVE_FAMILIES = ("logistic", "weibull", "exponential", "loglogistic",
                  "step", "uniform", "triangular")
COUPLING_MODES = ("fixedVOT", "copula")

# Canonical group universe (mirrors _BASE_GROUPS in competition.py).
# Attachments referencing anything else must raise -- see CurveRegistry.
_BASE_GROUPS = [
    "GratisAuto", "GratisAuto_GratisOV",
    "WelAuto_GratisOV", "WelAuto_vkAuto", "WelAuto_vkNeutraal",
    "WelAuto_vkFiets", "WelAuto_vkOV",
    "GeenAuto_GratisOV", "GeenAuto_vkNeutraal",
    "GeenAuto_vkFiets", "GeenAuto_vkOV",
    "GeenRijbewijs_GratisOV", "GeenRijbewijs_vkNeutraal",
    "GeenRijbewijs_vkFiets", "GeenRijbewijs_vkOV",
]

BASE_GROUPS = tuple(_BASE_GROUPS)  # public alias, used by ikob.curve_attachment
_INCOME_LEVELS = ["laag", "middellaag", "middelhoog", "hoog"]
_INCOME_LEVELS = ["laag", "middellaag", "middelhoog", "hoog"]
KNOWN_GROUPS = frozenset(f"{bg}_{ig}" for ig in _INCOME_LEVELS for bg in _BASE_GROUPS)


@dataclass(frozen=True)
class MarginalCurve:
    """Survival function S(x) of a one-dimensional tolerance budget.
    ... (family/a/b/c docstring unchanged) ...
    """
    family: str
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0

    def __post_init__(self):
        if self.family not in CURVE_FAMILIES:
            raise ValueError(f"Unknown curve family {self.family!r}; expected one of {CURVE_FAMILIES}.")

    def is_deterministic(self) -> bool:
        return self.family == "step" and self.b == 0.0

    def survival(self, x) -> npt.NDArray:
        # ... unchanged from before ...
        x = np.asarray(x, dtype=DTYPE)
        if self.family == "logistic":
            exponent = (x - self.b) * self.a
            return (1.0 / (1.0 + np.exp(np.clip(exponent, -60.0, 60.0)))).astype(DTYPE)
        x = np.maximum(x, 0.0)
        if self.family == "weibull":
            b = max(self.b, 1e-9)
            return np.exp(-np.power(x / b, self.a)).astype(DTYPE)
        if self.family == "exponential":
            b = max(self.b, 1e-9)
            return np.exp(-x / b).astype(DTYPE)
        if self.family == "loglogistic":
            b = max(self.b, 1e-9)
            return (1.0 / (1.0 + np.power(x / b, self.a))).astype(DTYPE)
        if self.family == "step":
            c1, p, c2 = self.a, self.b, max(self.c, self.a)
            return np.where(x < c1, 1.0, np.where(x < c2, p, 0.0)).astype(DTYPE)
        if self.family == "uniform":
            lo, hi = self.a, max(self.b, self.a + 1e-9)
            return np.clip((hi - x) / (hi - lo), 0.0, 1.0).astype(DTYPE)
        lo, hi = self.a, max(self.b, self.a + 1e-9)
        mode_frac = min(max(self.c, 0.0), 1.0)
        peak = lo + mode_frac * (hi - lo)
        eps = 1e-9
        if peak <= lo + eps:
            s = np.power((hi - x) / (hi - lo), 2)
        elif peak >= hi - eps:
            s = 1.0 - np.power((x - lo) / (hi - lo), 2)
        else:
            left = 1.0 - np.power(x - lo, 2) / ((hi - lo) * (peak - lo))
            right = np.power(hi - x, 2) / ((hi - lo) * (hi - peak))
            s = np.where(x <= peak, left, right)
        s = np.where(x <= lo, 1.0, np.where(x >= hi, 0.0, s))
        return s.astype(DTYPE)

    # -- diagnostics for relating a curve to observed drop-out behaviour --

    def hazard(self, x) -> npt.NDArray:
        """Instantaneous hazard rate f(x)/S(x).

        Closed-form per family. Returns 0 wherever a family has no
        continuous hazard (e.g. between the atoms of a 'step' curve);
        see atoms() for the jumps themselves, which a finite hazard/
        density value cannot represent (they are Dirac impulses).
        """
        x = np.asarray(x, dtype=DTYPE)
        if self.family == "logistic":
            # h(x) = a * (1 - S(x)); derived from S = 1/(1+e^z), z = a(x-b).
            s = self.survival(x)
            return (self.a * (1.0 - s)).astype(DTYPE)
        x = np.maximum(x, 0.0)
        if self.family == "weibull":
            b = max(self.b, 1e-9)
            xb = np.maximum(x / b, 1e-12)
            return (self.a / b * np.power(xb, self.a - 1.0)).astype(DTYPE)
        if self.family == "exponential":
            b = max(self.b, 1e-9)
            return np.full_like(x, 1.0 / b, dtype=DTYPE)
        if self.family == "loglogistic":
            b = max(self.b, 1e-9)
            xb = np.maximum(x / b, 1e-12)
            num = self.a / b * np.power(xb, self.a - 1.0)
            den = 1.0 + np.power(xb, self.a)
            return (num / den).astype(DTYPE)
        if self.family == "step":
            return np.zeros_like(x, dtype=DTYPE)
        if self.family == "uniform":
            lo, hi = self.a, max(self.b, self.a + 1e-9)
            h = np.where((x >= lo) & (x < hi), 1.0 / np.maximum(hi - x, 1e-9), 0.0)
            return h.astype(DTYPE)
        # triangular: derive the piecewise-linear density, then h = f / S.
        lo, hi = self.a, max(self.b, self.a + 1e-9)
        mode_frac = min(max(self.c, 0.0), 1.0)
        peak = lo + mode_frac * (hi - lo)
        eps = 1e-9
        if peak <= lo + eps:
            f = 2.0 * (hi - x) / np.maximum((hi - lo) ** 2, eps)
        elif peak >= hi - eps:
            f = 2.0 * (x - lo) / np.maximum((hi - lo) ** 2, eps)
        else:
            f_left = 2.0 * (x - lo) / np.maximum((hi - lo) * (peak - lo), eps)
            f_right = 2.0 * (hi - x) / np.maximum((hi - lo) * (hi - peak), eps)
            f = np.where(x <= peak, f_left, f_right)
        f = np.where((x >= lo) & (x <= hi), f, 0.0)
        s = self.survival(x)
        return (f / np.maximum(s, 1e-9)).astype(DTYPE)

    def density(self, x) -> npt.NDArray:
        """Probability density -dS/dx, i.e. hazard(x) * survival(x).

        See atoms() for the 'step' family's point masses, which this
        method cannot represent as a finite value.
        """
        return (self.hazard(x) * self.survival(x)).astype(DTYPE)

    def atoms(self) -> list[tuple[float, float]]:
        """Point-mass jumps (x, probability mass) in the survival function.

        Only the 'step' family has these; every other family is
        absolutely continuous and returns an empty list. The editor
        overlays these as dashed markers on the hazard/density
        mini-plots, since a Dirac impulse has no finite curve value.
        """
        if self.family != "step":
            return []
        c1, p, c2 = self.a, min(max(self.b, 0.0), 1.0), max(self.c, self.a)
        if c2 <= c1 + 1e-9:
            return [(c1, 1.0)]
        pts = []
        if p < 1.0:
            pts.append((c1, 1.0 - p))
        if p > 0.0:
            pts.append((c2, p))
        return pts

@dataclass(frozen=True)
class ToleranceSpec:
    """A complete W(t, m) specification on the (time x money) plane."""
    time_curve: MarginalCurve
    money_curve: MarginalCurve | None
    mode: str                  # "fixedVOT" | "copula"
    tau: float = 0.0           # minutes per euro (fixedVOT only)
    theta: float = 1.0         # Gumbel dependence (copula only), >= 1
    scaling: float = 1.0       # multiplicative scaling (Tables 9-11 compat)
    ikob_cutoff: bool = True   # apply 180-min and epsilon cutoffs

    def __post_init__(self):
        if self.mode not in COUPLING_MODES:
            raise ValueError(f"Unknown coupling mode {self.mode!r}; expected one of {COUPLING_MODES}.")
        if self.mode == "copula" and self.money_curve is None:
            raise ValueError("Copula mode requires a money curve.")
        if self.mode == "copula" and self.theta < 1.0:
            raise ValueError(f"Gumbel copula requires theta >= 1, got {self.theta}.")

    def is_degenerate_copula(self) -> bool:
        """True when theta is structurally unidentifiable (both marginals
        are hard-cutoff 'step' curves), mirroring the editor's warning."""
        return (self.mode == "copula"
                and self.time_curve.is_deterministic()
                and self.money_curve is not None
                and self.money_curve.is_deterministic())


def weight_matrix(spec: ToleranceSpec, time_skim, money_skim) -> npt.NDArray:
    """Evaluate W(t, m) element-wise. Dense float32 output."""
    t = np.asarray(time_skim, dtype=DTYPE)
    m = np.asarray(money_skim, dtype=DTYPE)

    if spec.mode == "fixedVOT":
        g = t + spec.tau * m
        w = spec.time_curve.survival(g) * spec.scaling
        if spec.ikob_cutoff:
            w = np.where(g < MAX_TIME_MINUTES, w, 0.0)
    else:
        s_t = spec.time_curve.survival(t)
        s_m = spec.money_curve.survival(m)
        # A budget fully exhausted (a hazard atom, e.g. a hard step
        # cutoff) makes -ln(0) diverge; the exact weakest-link limit
        # there is W = 0, regardless of ikob_cutoff.
        exhausted = (s_t <= 0.0) | (s_m <= 0.0)
        lt = -np.log(np.clip(s_t, 1e-12, 1.0))
        lm = -np.log(np.clip(s_m, 1e-12, 1.0))
        w = np.exp(-np.power(np.power(lt, spec.theta) + np.power(lm, spec.theta),
                             1.0 / spec.theta)) * spec.scaling
        w = np.where(exhausted, 0.0, w)
        if spec.ikob_cutoff:
            w = np.where(t < MAX_TIME_MINUTES, w, 0.0)

    w = w.astype(DTYPE)
    if spec.ikob_cutoff:
        w[w < WEIGHT_EPSILON] = 0.0
    return w


def calculate_weights_2d(time_skim, money_skim, spec: ToleranceSpec):
    """Two-dimensional counterpart of single_weights.calculate_weights.

    Takes the *un-collapsed* (time, money) skim pair instead of a
    pre-aggregated GTT matrix, and a saved ToleranceSpec instead of the
    (modality, preference, decay_curve_name) constants lookup.
    """
    return maybe_to_sparse(weight_matrix(spec, time_skim, money_skim))


def legacy_spec(modality, preference, decay_curve_name, tvom_factor) -> ToleranceSpec:
    """Spec that reproduces the current pipeline exactly (golden check).

    calculate_weights_2d(t, m, legacy_spec(...)) is bit-for-bit identical
    in structure to calculate_weights(t + tvom_factor * m, ...).
    """
    from ikob.constants import work_constants  # local import: avoid cycles
    alpha, omega, scaling = work_constants(modality, preference, decay_curve_name)
    return ToleranceSpec(
        time_curve=MarginalCurve("logistic", alpha, omega),
        money_curve=None,
        mode="fixedVOT",
        tau=tvom_factor,
        scaling=scaling,
    )


# -- JSON (de)serialisation; schema shared with the tolerance editor ---

def _curve_to_dict(curve: MarginalCurve) -> dict:
    return {"family": curve.family, "a": curve.a, "b": curve.b, "c": curve.c}


def _curve_from_dict(d: dict) -> MarginalCurve:
    return MarginalCurve(d["family"], float(d.get("a", 0.0)), float(d.get("b", 0.0)),
                         float(d.get("c", 0.0)))


def spec_to_dict(spec: ToleranceSpec) -> dict:
    if spec.mode == "fixedVOT":
        koppeling = {"modus": "fixedVOT", "tau": spec.tau}
    else:
        koppeling = {"modus": "copula", "theta": spec.theta}
    return {
        "tijd": _curve_to_dict(spec.time_curve),
        "geld": _curve_to_dict(spec.money_curve) if spec.money_curve else None,
        "koppeling": koppeling,
        "scaling": spec.scaling,
        "ikob_cutoff": spec.ikob_cutoff,
    }


def spec_from_dict(d: dict) -> ToleranceSpec:
    koppeling = d["koppeling"]
    mode = koppeling["modus"]
    geld = d.get("geld")
    return ToleranceSpec(
        time_curve=_curve_from_dict(d["tijd"]),
        money_curve=_curve_from_dict(geld) if geld else None,
        mode=mode,
        tau=float(koppeling.get("tau", 0.0)),
        theta=float(koppeling.get("theta", 1.0)),
        scaling=float(d.get("scaling", 1.0)),
        ikob_cutoff=bool(d.get("ikob_cutoff", True)),
    )


def load_library(path) -> list[dict]:
    path = pathlib.Path(path)
    with open(path) as f:
        parsed = json.load(f)
    entries = parsed if isinstance(parsed, list) else parsed.get("vervalscurven")
    if not isinstance(entries, list):
        raise ValueError(f"{path} bevat geen geldige curve-bibliotheek "
                         "(verwacht een lijst of {'vervalscurven': [...]}).")
    for entry in entries:
        spec_from_dict(entry["spec"])  # fail fast on malformed specs
    return entries


def save_library(path, entries: list[dict]):
    path = pathlib.Path(path)
    with open(path, "w") as f:
        json.dump({"versie": SCHEMA_VERSION, "vervalscurven": entries}, f, indent=2)
    logger.info("Curve library with %d entries written to %s.", len(entries), path)


class CurveRegistry:
    """Group -> ToleranceSpec lookup for the pipeline.

    Strict by construction: attachments referencing an unknown group name
    raise instead of silently falling through to the default constants
    tables (same hardening rationale as constants._validate for the
    historical 'E-fiets' bug).
    """

    def __init__(self, entries: list[dict]):
        self._by_group: dict[str, ToleranceSpec] = {}
        for entry in entries:
            spec = spec_from_dict(entry["spec"])
            for group in entry.get("groepen", []):
                if group not in KNOWN_GROUPS:
                    raise ValueError(
                        f"Curve set {entry.get('naam', '?')!r} is attached to unknown "
                        f"group {group!r}. Known groups follow the pattern "
                        "'<autobezit>_<voorkeur>_<inkomen>', e.g. 'GeenAuto_vkOV_laag'."
                    )
                if group in self._by_group:
                    raise ValueError(
                        f"Group {group!r} has more than one curve set attached "
                        f"(second one: {entry.get('naam', '?')!r})."
                    )
                self._by_group[group] = spec

    @classmethod
    def from_json(cls, path) -> "CurveRegistry":
        return cls(load_library(path))

    def spec_for(self, group: str) -> ToleranceSpec | None:
        """Spec for *group*, or None (caller falls back to constants tables)."""
        return self._by_group.get(group)

    def __contains__(self, group):
        return group in self._by_group

    def __len__(self):
        return len(self._by_group)