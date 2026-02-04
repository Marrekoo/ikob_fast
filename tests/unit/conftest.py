import numpy as np
import pytest


class SegsCapture:
    """Test double for SegsSource that captures reads and writes in memory."""

    def __init__(self, data_by_key):
        self.data_by_key = data_by_key
        self.writes_csv = []
        self.writes_xlsx = []

    def read(self, id: str, type_caster=int, scenario=""):
        key = (id, scenario)
        if key not in self.data_by_key:
            raise KeyError(f"Missing SEGS fixture for id={id!r}, scenario={scenario!r}")
        return np.array(self.data_by_key[key], dtype=type_caster)

    def write_csv(self, data, id, header, group="", modifier="", scenario=""):
        self.writes_csv.append(
            {
                "id": id,
                "group": group,
                "modifier": modifier,
                "scenario": scenario,
                "header": list(header),
                "data": np.array(data),
            }
        )

    def write_xlsx(self, data, id, header, group="", modifier="", scenario=""):
        self.writes_xlsx.append(
            {
                "id": id,
                "group": group,
                "modifier": modifier,
                "scenario": scenario,
                "header": list(header),
                "data": np.array(data),
            }
        )


@pytest.fixture
def segs_capture(monkeypatch):
    """Factory fixture to patch SegsSource to an in-memory dict-backed provider.

    Usage:
        cap = segs_capture({("SomeId", "2023"): array, ...})
    """

    def _make(data_by_key) -> SegsCapture:
        # Some modules import SegsSource directly ("from ikob.datasource import SegsSource"),
        # so patching only ikob.datasource.SegsSource is not sufficient.
        import ikob.competition as competition
        import ikob.datasource as datasource
        import ikob.distribute_over_groups as distribute_over_groups
        import ikob.employment_opportunities as employment_opportunities
        import ikob.potential_companies as potential_companies

        capture = SegsCapture(data_by_key)

        monkeypatch.setattr(datasource, "SegsSource", lambda _: capture)
        monkeypatch.setattr(distribute_over_groups, "SegsSource", lambda _: capture)
        monkeypatch.setattr(employment_opportunities, "SegsSource", lambda _: capture)
        monkeypatch.setattr(potential_companies, "SegsSource", lambda _: capture)
        monkeypatch.setattr(competition, "SegsSource", lambda _: capture)
        return capture

    return _make
