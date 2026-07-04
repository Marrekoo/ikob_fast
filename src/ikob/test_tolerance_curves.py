import numpy as np
import pytest

from ikob.configuration_definition import DecayCurveName
from ikob.single_weights import calculate_weights
from ikob.tolerance_curves import (
    CurveRegistry, MarginalCurve, ToleranceSpec,
    legacy_spec, spec_from_dict, spec_to_dict, weight_matrix,
)
from ikob.utils import ensure_dense


@pytest.fixture
def skims():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, 220.0, (60, 60)).astype(np.float32)
    m = rng.uniform(0.0, 20.0, (60, 60)).astype(np.float32)
    return t, m


def test_fixed_vot_reproduces_legacy_pipeline(skims):
    """legacy_spec through the 2-D path == current D1+D2 collapse."""
    t, m = skims
    tau = 9.0  # TVOM 'middellaag', motief werk
    for preference in ["Auto", "Neutraal", "Fiets", "OV"]:
        spec = legacy_spec("Auto", preference, DecayCurveName.WORK_AND_SOCIAL, tau)
        reference = ensure_dense(
            calculate_weights(t + tau * m, "Auto", preference,
                              DecayCurveName.WORK_AND_SOCIAL))
        new = weight_matrix(spec, t, m)
        np.testing.assert_allclose(new, reference, rtol=1e-6, atol=1e-7)


def test_copula_theta_one_is_independent_product(skims):
    t, m = skims
    s_t = MarginalCurve("logistic", 0.125, 45.0)
    s_m = MarginalCurve("logistic", 0.7, 8.0)
    spec = ToleranceSpec(s_t, s_m, "copula", theta=1.0, ikob_cutoff=False)
    np.testing.assert_allclose(weight_matrix(spec, t, m),
                               s_t.survival(t) * s_m.survival(m),
                               rtol=1e-5, atol=1e-6)


def test_exponential_marginals_make_modes_indistinguishable(skims):
    """The identifiability caveat: with constant hazards, independence
    coincides exactly with fixed VOT at tau = lambda_T / lambda_M."""
    t, m = skims
    lam_t, lam_m = 60.0, 10.0
    independent = ToleranceSpec(
        MarginalCurve("exponential", 0.0, lam_t),
        MarginalCurve("exponential", 0.0, lam_m),
        "copula", theta=1.0, ikob_cutoff=False)
    fixed = ToleranceSpec(
        MarginalCurve("exponential", 0.0, lam_t), None,
        "fixedVOT", tau=lam_t / lam_m, ikob_cutoff=False)
    np.testing.assert_allclose(weight_matrix(independent, t, m),
                               weight_matrix(fixed, t, m), rtol=1e-5, atol=1e-6)


def test_spec_json_roundtrip():
    spec = ToleranceSpec(MarginalCurve("weibull", 2.0, 60.0),
                         MarginalCurve("loglogistic", 3.0, 9.0),
                         "copula", theta=2.5)
    assert spec_from_dict(spec_to_dict(spec)) == spec


def test_registry_rejects_unknown_group():
    entry = {"naam": "test", "groepen": ["GeenAuto_vkOV_laag"],
             "spec": spec_to_dict(legacy_spec("Auto", "OV",
                                              DecayCurveName.WORK_AND_SOCIAL, 9.0))}
    assert len(CurveRegistry([entry])) == 1

    bad = dict(entry, groepen=["GeenAuto_vkOV_lag"])  # typo must raise, not fall through
    with pytest.raises(ValueError, match="unknown"):
        CurveRegistry([bad])

    with pytest.raises(ValueError, match="more than one"):
        CurveRegistry([entry, dict(entry, naam="tweede")])