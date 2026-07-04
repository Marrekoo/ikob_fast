import logging

from ikob.configuration_definition import DecayCurveName

logger = logging.getLogger(__name__)

# Modalities that have decay-curve parameters defined (Tables 9-11 in
# IKOB-algorithm.pdf). Note the canonical spelling "EFiets": the
# user-facing config uses "E-fiets", but callers must translate to this
# spelling before calling into this module (see single_weights.py).
_KNOWN_MODALITIES = frozenset({"Auto", "OV", "Fiets", "EFiets"})

# Preferences ("voorkeuren") that the parameter tables account for.
# "Neutraal" and "" both fall through to the base parameters on purpose.
_KNOWN_PREFERENCES = frozenset({"Auto", "Neutraal", "Fiets", "OV", ""})


def _validate(modality, preference):
    """Reject unknown modality/preference strings instead of silently
    falling through to the base (car) parameters.

    Historically, an unrecognised modality (e.g. the misspelled
    "E-fiets") would skip every branch below and return the car
    parameters, producing plausible-looking but wrong weights. Failing
    fast here makes that class of bug impossible.
    """
    if modality not in _KNOWN_MODALITIES:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of "
            f"{sorted(_KNOWN_MODALITIES)}. "
            "(Hint: the config spelling 'E-fiets' must be translated to "
            "'EFiets' before requesting decay parameters.)"
        )
    if preference not in _KNOWN_PREFERENCES:
        raise ValueError(
            f"Unknown preference {preference!r} for modality {modality!r}; "
            f"expected one of {sorted(p for p in _KNOWN_PREFERENCES if p)} "
            "or an empty string."
        )


def work_constants(modality, preference, decay_curve_name: DecayCurveName):
    """Returns the value from Table 9-11 in IKOB-algorithm.pdf"""
    _validate(modality, preference)

    if decay_curve_name == DecayCurveName.WORK_AND_SOCIAL:
        return _work_constants(modality, preference)
    elif decay_curve_name == DecayCurveName.DAILY_SHOPPING_AND_HEALTH:
        return _daily_shopping_constants(modality, preference)
    elif decay_curve_name == DecayCurveName.NON_DAILY_SHOPPING_AND_EDUCATION:
        return _non_daily_shopping_constants(modality, preference)
    else:
        raise ValueError(f"Unknown decay_curve_name: '{decay_curve_name}'")


def _work_constants(modality, preference):
    alpha = 0.125
    omega = 45
    scaling = 1
    if modality == "Fiets":
        alpha = 0.225
        omega = 25
    elif modality == "EFiets":
        alpha = 0.175
        omega = 35
    if preference == "Auto":
        if modality == "Auto":
            omega = 50
        elif modality == "OV":
            omega = 30
            scaling = 0.95
    elif preference == "OV":
        if modality == "Auto":
            scaling = 0.96
            alpha = 0.125
            omega = 45
        elif modality == "OV":
            alpha = 0.12
            omega = 60
    elif preference == "Fiets":
        if modality == "Auto":
            scaling = 0.75
        elif modality == "Fiets":
            alpha = 0.175
            omega = 35
        elif modality == "EFiets":
            alpha = 0.125
            omega = 55
    return alpha, omega, scaling


def _daily_shopping_constants(modality, preference):
    alpha = 0.225
    omega = 12.5
    scaling = 1
    if modality in ("Fiets", "EFiets"):
        omega = 10
        if preference == "Fiets":
            omega = 12.5
    if modality == "Auto" and preference == "Fiets":
        scaling = 0.75
    if modality == "OV" and preference == "OV":
        alpha = 0.175

    return alpha, omega, scaling


def _non_daily_shopping_constants(modality, preference):
    alpha = 0.225
    omega = 20
    scaling = 1
    if modality in ("Fiets", "EFiets"):
        omega = 15
        if preference == "Fiets":
            omega = 20
    if modality == "Auto" and preference == "Fiets":
        scaling = 0.75
    if modality == "OV" and preference == "OV":
        alpha = 0.175

    return alpha, omega, scaling