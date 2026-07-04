import pytest

from ikob.configuration_definition import DecayCurveName
from ikob.curve_attachment import (
    ConflictingCurveAttachment, resolve_spec_for_computation, sibling_base_groups,
)
from ikob.tolerance_curves import CurveRegistry, MarginalCurve, ToleranceSpec, legacy_spec, spec_to_dict


def _entry(name, groups, spec):
    return {"naam": name, "groepen": groups, "spec": spec_to_dict(spec)}


def test_resolve_returns_none_without_registry():
    assert resolve_spec_for_computation(None, "WelAuto_vkAuto", "Auto", "laag") is None


def test_resolve_returns_none_when_group_not_attached():
    spec = legacy_spec("Auto", "OV", DecayCurveName.WORK_AND_SOCIAL, 9.0)
    registry = CurveRegistry([_entry("t", ["WelAuto_vkAuto_laag"], spec)])
    assert resolve_spec_for_computation(registry, "WelAuto_vkAuto", "Auto", "hoog") is None


def test_resolve_applies_attached_spec():
    spec = legacy_spec("Auto", "OV", DecayCurveName.WORK_AND_SOCIAL, 9.0)
    registry = CurveRegistry([_entry("t", ["WelAuto_vkAuto_laag"], spec)])
    assert resolve_spec_for_computation(registry, "WelAuto_vkAuto", "Auto", "laag") == spec


def test_ov_component_is_shared_across_car_possession_categories():
    """WelAuto_vkOV and GeenAuto_vkOV share the OV weight matrix -- the
    legacy decay curve never depended on car ownership for this modality."""
    siblings = sibling_base_groups("WelAuto_vkOV", "OV", "laag")
    assert "WelAuto_vkOV" in siblings
    assert "GeenAuto_vkOV" in siblings
    assert "GeenRijbewijs_vkOV" in siblings


def test_auto_component_of_welauto_is_not_shared_with_geenauto():
    """The 'Auto' component uses a different underlying skim (private car
    vs. car-sharing/taxi proxy) for WelAuto vs. GeenAuto/GeenRijbewijs."""
    siblings = sibling_base_groups("WelAuto_vkNeutraal", "Auto", "laag")
    assert siblings == ["WelAuto_vkNeutraal"]


def test_conflicting_attachment_on_shared_ov_component_raises():
    spec_a = ToleranceSpec(MarginalCurve("weibull", 2.0, 60.0), None, "fixedVOT", tau=9.0)
    spec_b = ToleranceSpec(MarginalCurve("weibull", 3.0, 40.0), None, "fixedVOT", tau=9.0)
    registry = CurveRegistry([
        _entry("a", ["WelAuto_vkOV_laag"], spec_a),
        _entry("b", ["GeenAuto_vkOV_laag"], spec_b),
    ])
    with pytest.raises(ConflictingCurveAttachment, match="share one underlying weight matrix"):
        resolve_spec_for_computation(registry, "WelAuto_vkOV", "OV", "laag")


def test_same_attachment_on_shared_ov_component_is_not_a_conflict():
    spec = ToleranceSpec(MarginalCurve("weibull", 2.0, 60.0), None, "fixedVOT", tau=9.0)
    registry = CurveRegistry([
        _entry("shared", ["WelAuto_vkOV_laag", "GeenAuto_vkOV_laag"], spec),
    ])
    assert resolve_spec_for_computation(registry, "WelAuto_vkOV", "OV", "laag") == spec