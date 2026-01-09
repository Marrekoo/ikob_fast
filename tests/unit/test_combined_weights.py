import numpy as np


def test_has_preference_rules():
    from ikob.combined_weights import has_preference

    assert has_preference(kind_car="GeenAuto", kind_pt="OV", preference="Auto") is False
    assert has_preference(kind_car="GeenRijbewijs", kind_pt="GratisOV", preference="Neutraal") is False
    assert has_preference(kind_car="GeenRijbewijs", kind_pt="GratisOV", preference="OV") is True

    assert has_preference(kind_car="GratisAuto", kind_pt="OV", preference="Auto") is True
    assert has_preference(kind_car="GratisAuto", kind_pt="OV", preference="Neutraal") is False
    assert has_preference(kind_car="GratisAuto", kind_pt="GratisOV", preference="Neutraal") is True
    assert has_preference(kind_car="GratisAuto", kind_pt="GratisOV", preference="OV") is False

    assert has_preference(kind_car="Auto", kind_pt="GratisOV", preference="OV") is True
    assert has_preference(kind_car="Auto", kind_pt="GratisOV", preference="Auto") is False


def test_calculate_combined_weights_takes_elementwise_max():
    import ikob.combined_weights as cw
    from ikob.datasource import DataKey

    # Prepare
    pod = "Restdag"
    motive = "werk"
    regime = "Basis"

    bike = np.array([[1.0, 1.0], [1.0, 1.0]])
    pt = np.array([[2.0, 0.0], [0.0, 2.0]])
    car = np.array([[0.5, 3.0], [3.0, 0.5]])

    class _FakeSingleWeights:
        def get(self, key: DataKey):
            # calculate_combined_weights requests many combinations across:
            # - incomes (4)
            # - preferences (4)
            # - pt kinds (OV, GratisOV)
            # - car kinds (Auto, GeenAuto, GeenRijbewijs, GratisAuto)
            # - fuels for Auto
            # For this unit test we only care that the combinator uses elementwise max,
            # so we return simple matrices by broad category.
            if key.id == "Fiets_vk":
                return bike

            if key.id in {"OV_vk", "GratisOV_vk"}:
                return pt

            if key.id in {"Auto_vk", "GeenAuto_vk", "GeenRijbewijs_vk", "GratisAuto_vk"}:
                return car

            raise KeyError(key)

    config = {
        "__filename__": "pytest",
        "project": {
            "motieven": [motive],
            "beprijzingsregime": regime,
            "paden": {
                "output_directory": "out",
                "skims_directory": "skims",
                "segs_directory": "segs",
            },
        },
        "skims": {"dagsoort": [pod]},
    }

    # Act
    combined = cw.calculate_combined_weights(config, _FakeSingleWeights())

    # Assert
    # Example: Auto_OV_vk is computed as max(Auto_vk, OV_vk) for each cell.
    key = DataKey(
        "Auto_OV_vk",
        part_of_day=pod,
        income="laag",
        regime=regime,
        motive=motive,
        preference="OV",
        subtopic="combinaties",
        fuel_kind="fossiel",
    )

    out = combined.get(key)
    expected = np.maximum(pt, car)
    np.testing.assert_allclose(out, expected)
