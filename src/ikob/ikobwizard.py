"""Combined IKOB configuration + run wizard.

Guided, step-by-step flow:

    1. Zonedata       -- pick CSV or GeoPackage zone data; for GeoPackage,
                         map raw columns onto the variables IKOB needs,
                         directly or via a derivation (ratio / shares).
    2. Reistijd & kosten (skims) -- file-based or generated (r5py/OSRM)
                         skims, plus an optional study-area + perimeter
                         selection that narrows the modelled zone set.
    3. Project        -- motief, TVOM keuze, decay curve, inkomensgroepen.
    4. Waarde van tijd
    5. Verdeling groepen
    6. Ketens & hubs
    7. Geavanceerd
    8. Uitvoeren      -- full validation + run.

Each step has a status dot in the sidebar (gray = not yet checked,
green = OK, amber = warning, red = error) that updates automatically,
shortly after you stop editing.

This module is purely additive: ikob.ikobconfig and ikob.ikobrunner are
unchanged and remain fully usable on their own. This wizard reuses their
internals (load_config, saveConfig, run_scripts, and the
ikob.config.build/validate/widgets helpers) directly.

Launch standalone:  python -m ikob.ikobwizard [-v] [-p project.json]
"""

import argparse
import contextlib
import logging
import pathlib
import sys
import threading
from dataclasses import dataclass
from enum import Enum

# ruff: noqa: F403,F405
from tkinter import *
from tkinter import filedialog, messagebox, ttk

from ikob import segs_mapping
from ikob.config import build, validate
from ikob.config.widgets import PAD, ScrollableArea, pathWidget
from ikob.configuration_definition import default_config, default_configuration_definition, project_name
from ikob.ikobconfig import ConfigSaveError, _project_filename, load_config, saveConfig
from ikob.ikobrunner import ConfigApp as _RunnerConfigApp
from ikob.ikobrunner import run_scripts

logger = logging.getLogger(__name__)

RUN_STEP_LABELS = _RunnerConfigApp.stappen
RECHECK_DELAY_MS = 500


# ── Status model ───────────────────────────────────────────────────────

class StepStatus(Enum):
    UNKNOWN = "unknown"
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


STATUS_COLORS = {
    StepStatus.UNKNOWN: "#9ca3af",
    StepStatus.OK: "#16a34a",
    StepStatus.WARNING: "#d97706",
    StepStatus.ERROR: "#dc2626",
}

STATUS_BANNER_BG = {
    StepStatus.UNKNOWN: "#f3f4f6",
    StepStatus.OK: "#dcfce7",
    StepStatus.WARNING: "#fef9c3",
    StepStatus.ERROR: "#fee2e2",
}

_SEVERITY = {StepStatus.UNKNOWN: 0, StepStatus.OK: 1, StepStatus.WARNING: 2, StepStatus.ERROR: 3}


def _combine(status, messages, new_status, new_messages):
    combined = status if _SEVERITY[status] >= _SEVERITY[new_status] else new_status
    return combined, messages + list(new_messages)


class _ListLogHandler(logging.Handler):
    def __init__(self, level=logging.WARNING):
        super().__init__(level=level)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(self.format(record))


@contextlib.contextmanager
def capture_log_messages(level=logging.WARNING):
    handler = _ListLogHandler(level=level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        root_logger.removeHandler(handler)


# ── Per-step checkers ────────────────────────────────────────────────
#
# Deliberately lightweight for the six configuration steps (existence
# checks, small reads, mapping-completeness checks that don't require
# opening the gpkg); the full, authoritative check (matching
# FileValidator exactly, including resolving every mapped variable)
# only ever runs on the final "Uitvoeren" step, via _check_run.


def _check_zonedata(config):
    status, messages = StepStatus.OK, []
    try:
        paden = config["project"]["paden"]
        segs_format = paden.get("segs_format", "csv")

        if segs_format == "csv":
            segs_dir = paden.get("segs_directory", "")
            if not segs_dir:
                return StepStatus.ERROR, ["Geen SEGS directory opgegeven."]
            if not pathlib.Path(segs_dir).is_dir():
                return StepStatus.ERROR, [f"SEGS directory niet gevonden: '{segs_dir}'."]
            try:
                from ikob.datasource import SegsSource
                segs_source = SegsSource(config)
            except Exception as err:
                return StepStatus.ERROR, [f"SEGS-bron ongeldig: {err}"]
            scenario = config["project"].get("verstedelijkingsscenario", "")
            for field, label in [("reizende populatie", "reizende populatie"),
                                 ("bestemmingsplaatsen", "bestemmingsplaatsen")]:
                value = config["project"]["motief"].get(field, "")
                if not value:
                    status, messages = _combine(status, messages, StepStatus.ERROR,
                                                [f"Geen bestand opgegeven voor '{label}'."])
                    continue
                try:
                    content = segs_source.read(pathlib.Path(value).name, scenario=scenario)
                    messages.append(f"'{label}' gevonden ({content.shape[0]} zones).")
                except Exception as err:
                    status, messages = _combine(status, messages, StepStatus.ERROR,
                                                [f"Kon '{label}' niet lezen: {err}"])
            return status, messages

        # gpkg
        gpkg_path = paden.get("segs_bestand", "")
        if not gpkg_path:
            return StepStatus.ERROR, ["Geen GeoPackage-bestand opgegeven."]
        try:
            from ikob import geo_utils
        except ImportError as err:
            return StepStatus.ERROR, [f"GeoPackage-ondersteuning niet beschikbaar: {err}"]

        buurten_layer = paden.get("segs_buurten_laag", "buurten")
        buurtcode_column = paden.get("segs_buurtcode_kolom", "buurtcode")
        try:
            gdf = geo_utils.validate_cbs_buurt_layer(gpkg_path, buurten_layer, buurtcode_column)
            messages.append(f"Referentielaag '{buurten_layer}' OK ({len(gdf)} buurten).")
        except geo_utils.GeoDataError as err:
            return StepStatus.ERROR, [str(err)]

        mapping = paden.get("segs_mapping", {}) or {}
        motive_fields = ["reizende populatie", "bestemmingsplaatsen"]
        complete, missing = segs_mapping.mapping_is_complete(mapping, motive_fields=motive_fields)
        if not complete:
            status, messages = _combine(status, messages, StepStatus.ERROR,
                                        [f"Nog niet gekoppeld: {m}" for m in missing])
        else:
            messages.append("Alle vereiste variabelen zijn gekoppeld.")

    except Exception as err:
        return StepStatus.ERROR, [f"Onverwachte fout: {err}"]
    return status, messages


def _check_skims(config):
    status, messages = StepStatus.OK, []
    try:
        skims = config["skims"]
        if not skims.get("dagsoort"):
            status, messages = _combine(status, messages, StepStatus.ERROR, ["Selecteer minstens één dagsoort."])

        skims_bron = skims.get("skims_bron", "bestanden")

        if skims_bron == "bestanden":
            paden = config["project"]["paden"]
            skims_dir = paden.get("skims_directory", "")
            if not skims_dir:
                status, messages = _combine(status, messages, StepStatus.ERROR, ["Geen skims directory opgegeven."])
            elif not pathlib.Path(skims_dir).is_dir():
                status, messages = _combine(status, messages, StepStatus.ERROR,
                                            [f"Skims directory niet gevonden: '{skims_dir}'."])
            else:
                missing = []
                skims_path = pathlib.Path(skims_dir)
                for pod in skims.get("dagsoort", []):
                    for skim_id in ("Auto_Tijd", "Fiets_Tijd", "OV_Tijd"):
                        candidate = (skims_path / pod / skim_id).with_suffix(".csv")
                        if not candidate.exists():
                            missing.append(str(candidate))
                if missing:
                    status, messages = _combine(status, messages, StepStatus.ERROR,
                                                [f"Skimbestand niet gevonden: '{m}'" for m in missing])
                else:
                    messages.append("Alle basis-skimbestanden gevonden.")

        elif skims_bron in ("r5py", "osrm"):
            try:
                from ikob.config.geo_validate import validate_geo_inputs
                with capture_log_messages() as msgs:
                    ok = validate_geo_inputs(config)
                derived = StepStatus.ERROR if not ok else (StepStatus.WARNING if msgs else StepStatus.OK)
                status, messages = _combine(status, messages, derived, msgs)
            except Exception as err:
                status, messages = _combine(status, messages, StepStatus.WARNING,
                                            [f"Kon {skims_bron}-configuratie niet controleren: {err}"])

            segs_format = config["project"]["paden"].get("segs_format", "csv")
            studiegebied = config["project"]["paden"].get("studiegebied", {}) or {}
            if segs_format == "gpkg" and not studiegebied.get("zone_codes"):
                status, messages = _combine(status, messages, StepStatus.WARNING,
                                            ["Geen studiegebied ingesteld: alle buurten in de GeoPackage "
                                             "worden gerouteerd (kan zeer lang duren bij landelijke data)."])
        else:
            status, messages = _combine(status, messages, StepStatus.ERROR, [f"Onbekende skims-bron '{skims_bron}'."])

        zone_file = skims.get("zone_locaties_bestand", "")
        if zone_file and not pathlib.Path(zone_file).exists():
            status, messages = _combine(status, messages, StepStatus.WARNING,
                                        [f"Zone-locatiebestand niet gevonden: '{zone_file}'."])

        parking_file = skims.get("parkeerzoektijden_bestand", "")
        if parking_file and not pathlib.Path(parking_file).exists():
            status, messages = _combine(status, messages, StepStatus.WARNING,
                                        [f"Parkeerzoektijden-bestand niet gevonden: '{parking_file}'."])

    except Exception as err:
        return StepStatus.ERROR, [f"Onverwachte fout: {err}"]
    return status, messages


def _check_project(config):
    status, messages = StepStatus.OK, []
    try:
        project = config["project"]
        if not project.get("naam"):
            status, messages = _combine(status, messages, StepStatus.WARNING, ["Geen projectnaam opgegeven."])
        if not project["paden"].get("output_directory", ""):
            status, messages = _combine(status, messages, StepStatus.WARNING, ["Geen output directory opgegeven."])

        tol = project["motief"].get("tolerantiecurven", "")
        if tol:
            try:
                from ikob.tolerance_curves import load_library
                entries = load_library(tol)
                messages.append(f"Curve-bibliotheek geladen: {len(entries)} curveset(s).")
            except Exception as err:
                status, messages = _combine(status, messages, StepStatus.ERROR,
                                            [f"Curve-bibliotheek ongeldig: {err}"])

        if not project.get("welke_inkomensgroepen"):
            status, messages = _combine(status, messages, StepStatus.ERROR, ["Selecteer minstens één inkomensgroep."])
    except Exception as err:
        return StepStatus.ERROR, [f"Onverwachte fout: {err}"]
    return status, messages


def _check_tvom(config):
    status, messages = StepStatus.OK, []
    try:
        tvom = config.get("TVOM", {})
        for scope_key in ("werk", "overig"):
            scope = tvom.get(scope_key, {})
            for level, value in scope.items():
                try:
                    if float(value) <= 0:
                        status, messages = _combine(status, messages, StepStatus.WARNING,
                                                    [f"TVOM {scope_key}/{level} is niet positief ({value})."])
                except (TypeError, ValueError):
                    continue
    except Exception as err:
        return StepStatus.ERROR, [f"Onverwachte fout: {err}"]
    return status, messages


def _check_verdeling(config):
    status, messages = StepStatus.OK, []
    try:
        verdeling = config.get("verdeling", {})
        for level, value in verdeling.get("Percelektrisch", {}).items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not (0 <= v <= 100):
                status, messages = _combine(status, messages, StepStatus.WARNING,
                                            [f"Elektrisch percentage voor '{level}' buiten bereik 0-100: {v}."])
        gratis_ov = verdeling.get("GratisOVpercentage", None)
        try:
            if gratis_ov is not None and not (0 <= float(gratis_ov) <= 100):
                status, messages = _combine(status, messages, StepStatus.WARNING,
                                            [f"'Gratis OV' fractie buiten bereik: {gratis_ov}."])
        except (TypeError, ValueError):
            pass
    except Exception as err:
        return StepStatus.ERROR, [f"Onverwachte fout: {err}"]
    return status, messages


def _check_ketens(config):
    status, messages = StepStatus.OK, []
    try:
        ketens = config.get("ketens", {})
        chains = ketens.get("chains", {})
        if chains.get("gebruiken"):
            bestand = chains.get("bestand", "")
            if not bestand:
                status, messages = _combine(status, messages, StepStatus.ERROR, ["Geen hub-bestand opgegeven."])
            elif not pathlib.Path(bestand).exists():
                status, messages = _combine(status, messages, StepStatus.ERROR,
                                            [f"Hub-bestand niet gevonden: '{bestand}'."])
            else:
                try:
                    from ikob.chain_generator import Hubs
                    from ikob.datasource import read_csv_from_config
                    hubs_raw = read_csv_from_config(config, key="ketens", id="chains", has_index_column=False)
                    if not Hubs.validate(hubs_raw):
                        status, messages = _combine(status, messages, StepStatus.ERROR, ["Hub-bestand is ongeldig."])
                    else:
                        messages.append(f"{len(hubs_raw)} hub(s) gevonden.")
                except Exception as err:
                    status, messages = _combine(status, messages, StepStatus.ERROR,
                                                [f"Kon hub-bestand niet lezen: {err}"])
            if not chains.get("naam hub"):
                status, messages = _combine(status, messages, StepStatus.WARNING, ["Geen naam opgegeven voor de hubs."])

        bestemmingslijst = ketens.get("bestemmingslijst", {})
        if bestemmingslijst.get("gebruiken"):
            bestand = bestemmingslijst.get("bestand", "")
            if not bestand:
                status, messages = _combine(status, messages, StepStatus.ERROR, ["Geen bestemmingslijst-bestand opgegeven."])
            elif not pathlib.Path(bestand).exists():
                status, messages = _combine(status, messages, StepStatus.ERROR,
                                            [f"Bestemmingslijst-bestand niet gevonden: '{bestand}'."])
    except Exception as err:
        return StepStatus.ERROR, [f"Onverwachte fout: {err}"]
    return status, messages


def _check_geavanceerd(config):
    status, messages = StepStatus.OK, []
    try:
        geavanceerd = config.get("geavanceerd", {})
        for section, kind in [
            ("kunstmab", "kunstmatig-autobezit-bestand"),
            ("parkeerkosten", "parkeerkosten-bestand"),
            ("additionele_kosten", "additionele-kosten-bestand"),
        ]:
            block = geavanceerd.get(section, {})
            if block.get("gebruiken"):
                bestand = block.get("bestand", "")
                if not bestand:
                    status, messages = _combine(status, messages, StepStatus.ERROR, [f"Geen {kind} opgegeven."])
                elif not pathlib.Path(bestand).exists():
                    status, messages = _combine(status, messages, StepStatus.ERROR,
                                                [f"{kind.capitalize()} niet gevonden: '{bestand}'."])

        if not geavanceerd.get("welke_groepen"):
            status, messages = _combine(status, messages, StepStatus.ERROR, ["Selecteer minstens één groep."])
    except Exception as err:
        return StepStatus.ERROR, [f"Onverwachte fout: {err}"]
    return status, messages


def _check_run(config):
    try:
        with capture_log_messages() as messages:
            ok = validate.FileValidator(config).validate_input_files()
        if ok:
            return StepStatus.OK, (messages or ["Alle invoerbestanden zijn gevalideerd."])
        return StepStatus.ERROR, (messages or ["Validatie van de invoerbestanden is mislukt."])
    except Exception as err:
        return StepStatus.ERROR, [f"Onverwachte fout tijdens validatie: {err}"]


@dataclass
class WizardStep:
    key: str
    title: str
    checker: callable


def _build_steps():
    return [
        WizardStep("zonedata", "1. Zonedata", _check_zonedata),
        WizardStep("skims", "2. Reistijd & kosten", _check_skims),
        WizardStep("project", "3. Project", _check_project),
        WizardStep("tvom", "4. Waarde van tijd", _check_tvom),
        WizardStep("verdeling", "5. Verdeling groepen", _check_verdeling),
        WizardStep("ketens", "6. Ketens & hubs", _check_ketens),
        WizardStep("geavanceerd", "7. Geavanceerd", _check_geavanceerd),
        WizardStep("run", "8. Uitvoeren", _check_run),
    ]


# ── Small mapping-row widgets for the Zonedata step ─────────────────

class _PerZoneRow:
    def __init__(self, master, row, var_id, label):
        self.var_id = var_id
        Label(master, text=label + ":", wraplength=260, justify=LEFT).grid(row=row, column=0, sticky="w", **PAD)
        self.soort = StringVar(value="direct")
        soort_frame = Frame(master)
        soort_frame.grid(row=row, column=1, sticky="w")
        Radiobutton(soort_frame, text="Direct", variable=self.soort, value="direct",
                   command=self._toggle).pack(side=LEFT)
        Radiobutton(soort_frame, text="Verhouding a/b", variable=self.soort, value="ratio",
                   command=self._toggle).pack(side=LEFT)

        self.direct_frame = Frame(master)
        self.kolom = ttk.Combobox(self.direct_frame, width=28)
        self.kolom.pack(side=LEFT)

        self.ratio_frame = Frame(master)
        self.teller = ttk.Combobox(self.ratio_frame, width=16)
        self.teller.pack(side=LEFT)
        Label(self.ratio_frame, text=" / ").pack(side=LEFT)
        self.noemer = ttk.Combobox(self.ratio_frame, width=16)
        self.noemer.pack(side=LEFT)
        Label(self.ratio_frame, text=" ×").pack(side=LEFT)
        self.schaal = Entry(self.ratio_frame, width=6)
        self.schaal.insert(0, "1.0")
        self.schaal.pack(side=LEFT)

        self.direct_frame.grid(row=row, column=2, sticky="w", **PAD)
        self.ratio_frame.grid(row=row, column=2, sticky="w", **PAD)
        self._toggle()

    def _toggle(self):
        if self.soort.get() == "direct":
            self.ratio_frame.grid_remove()
            self.direct_frame.grid()
        else:
            self.direct_frame.grid_remove()
            self.ratio_frame.grid()

    def set_columns(self, columns):
        self.kolom["values"] = columns
        self.teller["values"] = columns
        self.noemer["values"] = columns

    def to_dict(self):
        if self.soort.get() == "direct":
            return {"soort": "direct", "kolom": self.kolom.get()}
        try:
            schaal = float(self.schaal.get())
        except ValueError:
            schaal = 1.0
        return {"soort": "ratio", "teller_kolom": self.teller.get(),
                "noemer_kolom": self.noemer.get(), "schaal": schaal}

    def load(self, entry):
        if not entry:
            return
        self.soort.set(entry.get("soort", "direct"))
        if entry.get("soort") == "ratio":
            self.teller.set(entry.get("teller_kolom", ""))
            self.noemer.set(entry.get("noemer_kolom", ""))
            self.schaal.delete(0, "end")
            self.schaal.insert(0, str(entry.get("schaal", 1.0)))
        else:
            self.kolom.set(entry.get("kolom", ""))
        self._toggle()

    def bind_recheck(self, callback):
        for w in (self.kolom, self.teller, self.noemer):
            w.bind("<<ComboboxSelected>>", callback)
            w.bind("<KeyRelease>", callback)
        self.schaal.bind("<KeyRelease>", callback)


class _MotiveRow:
    def __init__(self, master, row, field_name, label, class_labels, naam_var):
        self.field_name = field_name
        Label(master, text=label + ":", wraplength=260, justify=LEFT).grid(row=row, column=0, sticky="w", **PAD)

        name_frame = Frame(master)
        name_frame.grid(row=row, column=1, sticky="w", **PAD)
        Label(name_frame, text="Naam:").pack(side=LEFT)
        Entry(name_frame, textvariable=naam_var, width=24).pack(side=LEFT)

        self.soort = StringVar(value="direct")
        soort_frame = Frame(master)
        soort_frame.grid(row=row, column=2, sticky="w")
        Radiobutton(soort_frame, text="Kolom per klasse", variable=self.soort, value="direct",
                   command=self._toggle).pack(side=LEFT)
        Radiobutton(soort_frame, text="Totaal × aandeel", variable=self.soort, value="shares",
                   command=self._toggle).pack(side=LEFT)

        self.direct_frame = Frame(master)
        self.direct_boxes = []
        for label_txt in class_labels:
            sub = Frame(self.direct_frame)
            sub.pack(side=LEFT, padx=2)
            Label(sub, text=label_txt).pack()
            cb = ttk.Combobox(sub, width=14)
            cb.pack()
            self.direct_boxes.append(cb)

        self.shares_frame = Frame(master)
        sub = Frame(self.shares_frame)
        sub.pack(side=LEFT, padx=2)
        Label(sub, text="totaal").pack()
        self.totaal_box = ttk.Combobox(sub, width=14)
        self.totaal_box.pack()
        self.share_boxes = []
        for label_txt in class_labels:
            sub = Frame(self.shares_frame)
            sub.pack(side=LEFT, padx=2)
            Label(sub, text=label_txt).pack()
            cb = ttk.Combobox(sub, width=14)
            cb.pack()
            self.share_boxes.append(cb)

        self.direct_frame.grid(row=row + 1, column=1, columnspan=2, sticky="w", **PAD)
        self.shares_frame.grid(row=row + 1, column=1, columnspan=2, sticky="w", **PAD)
        self._toggle()

    def _toggle(self):
        if self.soort.get() == "direct":
            self.shares_frame.grid_remove()
            self.direct_frame.grid()
        else:
            self.direct_frame.grid_remove()
            self.shares_frame.grid()

    def set_columns(self, columns):
        for cb in [*self.direct_boxes, self.totaal_box, *self.share_boxes]:
            cb["values"] = columns

    def to_dict(self):
        if self.soort.get() == "direct":
            return {"soort": "direct", "kolommen": [cb.get() for cb in self.direct_boxes]}
        return {"soort": "shares", "totaal_kolom": self.totaal_box.get(),
                "aandeel_kolommen": [cb.get() for cb in self.share_boxes]}

    def load(self, entry):
        if not entry:
            return
        self.soort.set(entry.get("soort", "direct"))
        if entry.get("soort") == "shares":
            self.totaal_box.set(entry.get("totaal_kolom", ""))
            for cb, val in zip(self.share_boxes, entry.get("aandeel_kolommen", [])):
                cb.set(val)
        else:
            for cb, val in zip(self.direct_boxes, entry.get("kolommen", [])):
                cb.set(val)
        self._toggle()

    def bind_recheck(self, callback):
        for w in [*self.direct_boxes, self.totaal_box, *self.share_boxes]:
            w.bind("<<ComboboxSelected>>", callback)
            w.bind("<KeyRelease>", callback)


class _UrbanizationRow:
    def __init__(self, master, row, var_id, label, class_labels, preview_callback):
        self.var_id = var_id
        Label(master, text=label + ":", wraplength=260, justify=LEFT).grid(row=row, column=0, sticky="w", **PAD)

        self.soort = StringVar(value="handmatig")
        soort_frame = Frame(master)
        soort_frame.grid(row=row, column=1, sticky="w")
        Radiobutton(soort_frame, text="Uit gpkg-laag", variable=self.soort, value="laag",
                   command=self._toggle).pack(side=LEFT)
        Radiobutton(soort_frame, text="Handmatig", variable=self.soort, value="handmatig",
                   command=self._toggle).pack(side=LEFT)

        self.laag_frame = Frame(master)
        self.laag_box = ttk.Combobox(self.laag_frame, width=16)
        self.laag_box.pack(side=LEFT)
        Button(self.laag_frame, text="Bekijk kolommen",
              command=lambda: preview_callback(self.laag_box.get())).pack(side=LEFT, padx=4)
        Label(self.laag_frame, text="sleutel:").pack(side=LEFT)
        self.sleutel_entry = Entry(self.laag_frame, width=12)
        self.sleutel_entry.pack(side=LEFT)
        self.waarde_entries = []
        for label_txt in class_labels:
            Label(self.laag_frame, text=label_txt).pack(side=LEFT)
            e = Entry(self.laag_frame, width=8)
            e.pack(side=LEFT)
            self.waarde_entries.append(e)

        self.manual_frame = Frame(master)
        header = Frame(self.manual_frame)
        header.pack(anchor="w")
        Label(header, text="Sted.graad", width=10).pack(side=LEFT)
        for label_txt in class_labels:
            Label(header, text=label_txt, width=10).pack(side=LEFT)
        self.manual_entries = []
        for grade in range(1, 6):
            row_frame = Frame(self.manual_frame)
            row_frame.pack(anchor="w")
            Label(row_frame, text=str(grade), width=10).pack(side=LEFT)
            row_entries = []
            for _ in class_labels:
                e = Entry(row_frame, width=10)
                e.pack(side=LEFT)
                row_entries.append(e)
            self.manual_entries.append(row_entries)

        self.laag_frame.grid(row=row, column=2, sticky="w", **PAD)
        self.manual_frame.grid(row=row, column=2, sticky="w", **PAD)
        self._toggle()

    def _toggle(self):
        if self.soort.get() == "laag":
            self.manual_frame.grid_remove()
            self.laag_frame.grid()
        else:
            self.laag_frame.grid_remove()
            self.manual_frame.grid()

    def set_layers(self, layers):
        self.laag_box["values"] = layers

    def to_dict(self, gpkg_path):
        if self.soort.get() == "laag":
            return {"soort": "laag", "bestand": gpkg_path, "laag": self.laag_box.get(),
                    "sleutel_kolom": self.sleutel_entry.get(),
                    "waarde_kolommen": [e.get() for e in self.waarde_entries]}
        rows = []
        for row_entries in self.manual_entries:
            values = []
            for e in row_entries:
                try:
                    values.append(float(e.get()))
                except ValueError:
                    values.append(0.0)
            rows.append(values)
        return {"soort": "handmatig", "waarden": rows}

    def load(self, entry):
        if not entry:
            return
        self.soort.set(entry.get("soort", "handmatig"))
        if entry.get("soort") == "laag":
            self.laag_box.set(entry.get("laag", ""))
            self.sleutel_entry.delete(0, "end")
            self.sleutel_entry.insert(0, entry.get("sleutel_kolom", ""))
            for e, val in zip(self.waarde_entries, entry.get("waarde_kolommen", [])):
                e.delete(0, "end")
                e.insert(0, val)
        else:
            for row_entries, values in zip(self.manual_entries, entry.get("waarden", [])):
                for e, val in zip(row_entries, values):
                    e.delete(0, "end")
                    e.insert(0, str(val))
        self._toggle()

    def bind_recheck(self, callback):
        widgets = [self.laag_box, self.sleutel_entry, *self.waarde_entries]
        for row_entries in self.manual_entries:
            widgets.extend(row_entries)
        for w in widgets:
            w.bind("<<ComboboxSelected>>", callback)
            w.bind("<KeyRelease>", callback)


# ── Main wizard window ───────────────────────────────────────────────

class WizardApp(Tk):
    def __init__(self, initial_project=None):
        super().__init__()
        self.title("IKOB wizard")

        self._template = default_configuration_definition()
        build.addTkVarsTemplate(self._template)

        # Custom, non-declarative state (not representable by the plain
        # leaf-widget template system): the gpkg column mapping and the
        # study-area/perimeter selection. Stored directly in the project
        # config on save/run under project.paden.segs_mapping /
        # project.paden.studiegebied (config already tolerates extra
        # keys beyond the template -- see validate.validateConfigWithTemplate,
        # strict=False).
        self._zone_columns: list[str] = []
        self._zone_layers: list[str] = []
        self._studiegebied: dict = {}
        self._buurten_gdf_cache = None
        self._buurten_gdf_cache_key = None

        self._steps = _build_steps()
        self._status: dict[str, tuple] = {s.key: (StepStatus.UNKNOWN, []) for s in self._steps}
        self._current_index = 0
        self._current_path: str | None = None
        self._dirty = False
        self._recheck_job = None

        self._build_toolbar(self)
        body = Frame(self)
        body.pack(side=TOP, fill="both", expand=True)
        self._build_sidebar(body)
        self._build_content(body)
        self._build_navbar(self)

        self._default_bg = self._sidebar_rows[self._steps[0].key]["frame"].cget("background")
        self._bind_traces(self._template)
        self._toggle_zonedata_format()

        if initial_project:
            try:
                loaded = load_config(initial_project)
            except (ValueError, IOError) as err:
                messagebox.showwarning(title="Project laden", message=f"Kon '{initial_project}' niet laden:\n{err}")
            else:
                self._apply_loaded_config(loaded)
                self._current_path = initial_project

        self._show_step(0)
        self._update_title()
        self._fit_to_screen()
        self.after(50, self._recheck_all)

    # -- layout -----------------------------------------------------------

    def _fit_to_screen(self):
        self.update_idletasks()
        width = min(1150, self.winfo_screenwidth() - 80)
        height = min(780, self.winfo_screenheight() - 80)
        self.geometry(f"{width}x{height}")

    def _build_toolbar(self, master):
        bar = Frame(master)
        bar.pack(side=TOP, fill="x")
        Button(bar, text="Nieuw", command=self._cmd_new).pack(side=LEFT, padx=6, pady=6)
        Button(bar, text="Laden ...", command=self._cmd_load).pack(side=LEFT, padx=6, pady=6)
        Button(bar, text="Opslaan ...", command=self._cmd_save).pack(side=LEFT, padx=6, pady=6)
        Button(bar, text="🔄 Controleren", command=self._recheck_all).pack(side=RIGHT, padx=6, pady=6)
        self._path_label = Label(bar, text="", anchor="w")
        self._path_label.pack(side=LEFT, padx=12)
        return bar

    def _build_sidebar(self, master):
        frame = Frame(master, width=230)
        frame.pack(side=LEFT, fill="y")
        frame.pack_propagate(False)
        self._sidebar_rows = {}
        for i, step in enumerate(self._steps):
            row = Frame(frame, cursor="hand2")
            row.pack(fill="x", padx=6, pady=3)
            dot = Canvas(row, width=16, height=16, highlightthickness=0)
            dot.pack(side=LEFT, padx=(2, 6))
            oval = dot.create_oval(2, 2, 14, 14, fill=STATUS_COLORS[StepStatus.UNKNOWN], outline="")
            label = Label(row, text=step.title, anchor="w", justify=LEFT, wraplength=170)
            label.pack(side=LEFT, fill="x", expand=True)
            for widget in (row, dot, label):
                widget.bind("<Button-1>", lambda _e, idx=i: self._show_step(idx))
            self._sidebar_rows[step.key] = {"frame": row, "dot": dot, "oval": oval, "label": label}
        return frame

    def _build_content(self, master):
        container = Frame(master)
        container.pack(side=LEFT, fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self._pages = {}
        self._page_banners = {}
        for step in self._steps:
            page = Frame(container)
            page.grid(row=0, column=0, sticky="nsew")

            banner = Label(page, text="Nog niet gecontroleerd.", anchor="w", justify=LEFT,
                          wraplength=750, padx=8, pady=6, background=STATUS_BANNER_BG[StepStatus.UNKNOWN])
            banner.pack(side=TOP, fill="x")
            self._page_banners[step.key] = banner

            if step.key == "zonedata":
                self._build_zonedata_page(page)
            elif step.key == "skims":
                self._build_skims_page(page)
            elif step.key == "project":
                self._build_project_page(page)
            elif step.key == "run":
                self._build_run_page(page)
            else:
                scroll = ScrollableArea(page)
                scroll.pack(side=TOP, fill="both", expand=True)
                build._addWidgets(scroll.body, self._template[step.key if step.key != "tvom" else "TVOM"])

            self._pages[step.key] = page
        return container

    # -- Zonedata page ------------------------------------------------------

    def _build_zonedata_page(self, page):
        scroll = ScrollableArea(page)
        scroll.pack(side=TOP, fill="both", expand=True)
        master = scroll.body

        paden = self._template["project"]["paden"]
        motief = self._template["project"]["motief"]

        Label(master, text="Formaat van de zonedata:", font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w", **PAD)
        format_frame = Frame(master)
        format_frame.grid(row=0, column=1, columnspan=2, sticky="w")
        Radiobutton(format_frame, text="CSV (bestaand formaat, één bestand per variabele)",
                   variable=paden["segs_format"]["tkvar"], value="csv",
                   command=self._toggle_zonedata_format).pack(anchor="w")
        Radiobutton(format_frame, text="GeoPackage (bv. CBS-opendata buurten)",
                   variable=paden["segs_format"]["tkvar"], value="gpkg",
                   command=self._toggle_zonedata_format).pack(anchor="w")

        self._csv_frame = Frame(master)
        self._csv_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        pathWidget(self._csv_frame, "SEGS directory", paden["segs_directory"]["tkvar"], row=0)
        pathWidget(self._csv_frame, "Reizende populatie bestand", motief["reizende populatie"]["tkvar"],
                  row=1, file=True)
        pathWidget(self._csv_frame, "Bestemmingen bestand", motief["bestemmingsplaatsen"]["tkvar"],
                  row=2, file=True)

        self._gpkg_frame = Frame(master)
        self._gpkg_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        self._build_gpkg_zonedata_frame(self._gpkg_frame, paden, motief)

    def _build_gpkg_zonedata_frame(self, master, paden, motief):
        grow = 0
        pathWidget(master, "GeoPackage-bestand", paden["segs_bestand"]["tkvar"], row=grow, file=True)
        grow += 1

        row1 = Frame(master)
        row1.grid(row=grow, column=0, columnspan=3, sticky="w", **PAD)
        Button(row1, text="Lees lagen", command=self._load_zone_layers).pack(side=LEFT)
        Label(row1, text="Referentielaag (buurten):").pack(side=LEFT, padx=(12, 2))
        self._buurten_laag_combo = ttk.Combobox(row1, textvariable=paden["segs_buurten_laag"]["tkvar"], width=20)
        self._buurten_laag_combo.pack(side=LEFT)
        Label(row1, text="Buurtcode kolom:").pack(side=LEFT, padx=(12, 2))
        Entry(row1, textvariable=paden["segs_buurtcode_kolom"]["tkvar"], width=16).pack(side=LEFT)
        Button(row1, text="Laad kolommen", command=self._load_zone_columns).pack(side=LEFT, padx=(12, 0))
        grow += 1

        self._zone_columns_label = Label(master, text="Nog geen kolommen geladen.", anchor="w",
                                         justify=LEFT, wraplength=780)
        self._zone_columns_label.grid(row=grow, column=0, columnspan=3, sticky="w", **PAD)
        grow += 1

        per_zone_frame = LabelFrame(master, text="Basisvariabelen per buurt", padx=6, pady=6)
        per_zone_frame.grid(row=grow, column=0, columnspan=3, sticky="ew", **PAD)
        self._per_zone_rows = []
        for i, (var_id, label) in enumerate(segs_mapping.PER_ZONE_VARIABLES):
            row_widget = _PerZoneRow(per_zone_frame, i, var_id, label)
            row_widget.bind_recheck(self._schedule_recheck)
            self._per_zone_rows.append(row_widget)
        grow += 1

        motive_frame = LabelFrame(master, text="Reizende populatie & bestemmingen (per inkomensklasse)",
                                  padx=6, pady=6)
        motive_frame.grid(row=grow, column=0, columnspan=3, sticky="ew", **PAD)
        self._motive_rows = []
        mrow = 0
        for field_name, label, class_labels in segs_mapping.MOTIVE_CLASS_VARIABLES:
            naam_var = motief[field_name]["tkvar"]
            row_widget = _MotiveRow(motive_frame, mrow, field_name, label, class_labels, naam_var)
            row_widget.bind_recheck(self._schedule_recheck)
            self._motive_rows.append(row_widget)
            mrow += 2
        grow += 1

        lookup_frame = LabelFrame(master, text="Opzoektabellen naar stedelijkheidsgraad (survey-tabellen, "
                                              "niet per buurt)", padx=6, pady=6)
        lookup_frame.grid(row=grow, column=0, columnspan=3, sticky="ew", **PAD)
        self._urbanization_rows = []
        for i, (var_id, label, class_labels) in enumerate(segs_mapping.URBANIZATION_LOOKUP_VARIABLES):
            row_widget = _UrbanizationRow(lookup_frame, i, var_id, label, class_labels,
                                          self._preview_layer_columns)
            row_widget.bind_recheck(self._schedule_recheck)
            self._urbanization_rows.append(row_widget)

    def _toggle_zonedata_format(self):
        fmt = self._template["project"]["paden"]["segs_format"]["tkvar"].get()
        if fmt == "gpkg":
            self._csv_frame.grid_remove()
            self._gpkg_frame.grid()
        else:
            self._gpkg_frame.grid_remove()
            self._csv_frame.grid()
        self._schedule_recheck()

    def _load_zone_layers(self):
        gpkg_path = self._template["project"]["paden"]["segs_bestand"]["tkvar"].get()
        try:
            from ikob import geo_utils
            layers = geo_utils.list_layers(gpkg_path)
        except Exception as err:
            messagebox.showerror(title="Fout", message=f"Kon lagen niet lezen: {err}")
            return
        self._zone_layers = layers
        self._buurten_laag_combo["values"] = layers
        for row in self._urbanization_rows:
            row.set_layers(layers)
        messagebox.showinfo(title="Lagen gevonden", message=f"{len(layers)} laag/lagen gevonden:\n" + "\n".join(layers))

    def _load_zone_columns(self):
        gpkg_path = self._template["project"]["paden"]["segs_bestand"]["tkvar"].get()
        layer = self._template["project"]["paden"]["segs_buurten_laag"]["tkvar"].get()
        try:
            from ikob import geo_utils
            columns, dtypes, _ = geo_utils.layer_columns(gpkg_path, layer)
        except Exception as err:
            messagebox.showerror(title="Fout", message=f"Kon kolommen niet lezen: {err}")
            return
        self._zone_columns = columns
        preview = ", ".join(f"{c} ({dtypes[c]})" for c in columns)
        self._zone_columns_label.configure(text=f"{len(columns)} kolommen gevonden: {preview}")
        for row in self._per_zone_rows:
            row.set_columns(columns)
        for row in self._motive_rows:
            row.set_columns(columns)
        if hasattr(self, "_studiegebied_column_combo"):
            self._studiegebied_column_combo["values"] = columns
        self._schedule_recheck()

    def _preview_layer_columns(self, layer_name):
        if not layer_name:
            messagebox.showwarning(title="Kies eerst een laag", message="Kies eerst een laag.")
            return
        gpkg_path = self._template["project"]["paden"]["segs_bestand"]["tkvar"].get()
        try:
            from ikob import geo_utils
            columns, dtypes, _ = geo_utils.layer_columns(gpkg_path, layer_name)
        except Exception as err:
            messagebox.showerror(title="Fout", message=f"Kon kolommen niet lezen: {err}")
            return
        preview = "\n".join(f"{c}  ({dtypes[c]})" for c in columns)
        messagebox.showinfo(title=f"Kolommen van '{layer_name}'", message=preview)

    def _current_segs_mapping(self):
        mapping = segs_mapping.empty_mapping()
        for row in self._per_zone_rows:
            mapping["per_zone"][row.var_id] = row.to_dict()
        for row in self._motive_rows:
            mapping["motive"][row.field_name] = row.to_dict()
        gpkg_path = self._template["project"]["paden"]["segs_bestand"]["tkvar"].get()
        for row in self._urbanization_rows:
            mapping["urbanization_lookup"][row.var_id] = row.to_dict(gpkg_path)
        return mapping

    def _load_segs_mapping(self, mapping):
        mapping = mapping or {}
        for row in self._per_zone_rows:
            row.load(mapping.get("per_zone", {}).get(row.var_id))
        for row in self._motive_rows:
            row.load(mapping.get("motive", {}).get(row.field_name))
        for row in self._urbanization_rows:
            row.load(mapping.get("urbanization_lookup", {}).get(row.var_id))

    # -- Skims page (declarative tab + studiegebied section) ---------------

    def _build_skims_page(self, page):
        scroll = ScrollableArea(page)
        scroll.pack(side=TOP, fill="both", expand=True)
        build._addWidgets(scroll.body, self._template["skims"])
        self._build_studiegebied_section(scroll.body)

    def _build_studiegebied_section(self, master):
        frame = LabelFrame(master, text="Studiegebied (optioneel, alleen bij GeoPackage zonedata)",
                          padx=8, pady=8)
        frame.grid(row=999, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        row1 = Frame(frame)
        row1.pack(fill="x", pady=2)
        Label(row1, text="Kolom met gebiedsnamen (bv. gemeentenaam, wijknaam, buurtnaam, provincienaam):").pack(side=LEFT)
        self._studiegebied_column_combo = ttk.Combobox(row1, width=24, values=self._zone_columns)
        self._studiegebied_column_combo.pack(side=LEFT, padx=6)
        Button(row1, text="Laad namen", command=self._load_area_names).pack(side=LEFT)

        row2 = Frame(frame)
        row2.pack(fill="x", pady=2)
        Label(row2, text="Selecteer gebied(en):").pack(side=LEFT, anchor="n")
        self._area_listbox = Listbox(row2, selectmode=MULTIPLE, height=6, width=40, exportselection=False)
        self._area_listbox.pack(side=LEFT, padx=6)

        row3 = Frame(frame)
        row3.pack(fill="x", pady=2)
        Label(row3, text="Perimeter rondom studiegebied (km):").pack(side=LEFT)
        self._buffer_entry = Entry(row3, width=8)
        self._buffer_entry.insert(0, "10")
        self._buffer_entry.pack(side=LEFT, padx=6)
        Button(row3, text="Bereken studiegebied", command=self._compute_study_area).pack(side=LEFT, padx=6)
        Button(row3, text="Wis (gebruik alle zones)", command=self._clear_study_area).pack(side=LEFT)

        self._studiegebied_status = Label(frame, text=self._format_studiegebied_status(),
                                          anchor="w", justify=LEFT, wraplength=780)
        self._studiegebied_status.pack(fill="x", pady=(6, 0))

    def _format_studiegebied_status(self):
        if not self._studiegebied.get("zone_codes"):
            return "Geen studiegebied ingesteld: alle zones in de referentielaag worden gebruikt."
        core = len(self._studiegebied.get("core_codes", []))
        buf = len(self._studiegebied.get("buffer_codes", []))
        buffer_km = self._studiegebied.get("buffer_km", "?")
        return (f"Studiegebied ingesteld: {core} kernzone(s) + {buf} bufferzone(s) binnen "
                f"{buffer_km} km = {core + buf} zone(s) totaal.")

    def _get_buurten_gdf(self):
        from ikob import geo_utils
        gpkg_path = self._template["project"]["paden"]["segs_bestand"]["tkvar"].get()
        layer = self._template["project"]["paden"]["segs_buurten_laag"]["tkvar"].get()
        cache_key = (gpkg_path, layer)
        if self._buurten_gdf_cache_key != cache_key:
            self._buurten_gdf_cache = geo_utils.read_layer(gpkg_path, layer)
            self._buurten_gdf_cache_key = cache_key
        return self._buurten_gdf_cache

    def _load_area_names(self):
        column = self._studiegebied_column_combo.get()
        if not column:
            messagebox.showwarning(title="Kies eerst een kolom", message="Kies eerst een kolom met gebiedsnamen.")
            return
        try:
            gdf = self._get_buurten_gdf()
            values = sorted(gdf[column].dropna().astype(str).unique().tolist())
        except Exception as err:
            messagebox.showerror(title="Fout", message=f"Kon namen niet laden: {err}")
            return
        self._area_listbox.delete(0, "end")
        for v in values:
            self._area_listbox.insert("end", v)

    def _compute_study_area(self):
        column = self._studiegebied_column_combo.get()
        selected_indices = self._area_listbox.curselection()
        if not column or not selected_indices:
            messagebox.showwarning(title="Onvolledig", message="Kies een kolom en selecteer minstens één gebied.")
            return
        selected_names = {self._area_listbox.get(i) for i in selected_indices}
        try:
            buffer_km = float(self._buffer_entry.get())
        except ValueError:
            messagebox.showerror(title="Fout", message="Perimeter moet een getal zijn.")
            return
        buurtcode_column = self._template["project"]["paden"]["segs_buurtcode_kolom"]["tkvar"].get()
        try:
            from ikob import geo_utils
            gdf = self._get_buurten_gdf()
            mask = gdf[column].astype(str).isin(selected_names)
            core_codes, buffer_codes = geo_utils.dissolve_and_select(gdf, mask, buffer_km, buurtcode_column)
        except Exception as err:
            messagebox.showerror(title="Fout", message=f"Kon studiegebied niet berekenen: {err}")
            return
        self._studiegebied = {
            "laag_kolom": column, "geselecteerd": sorted(selected_names),
            "buffer_km": buffer_km, "core_codes": core_codes, "buffer_codes": buffer_codes,
            "zone_codes": sorted(set(core_codes) | set(buffer_codes)),
        }
        self._studiegebied_status.configure(text=self._format_studiegebied_status())
        self._schedule_recheck()

    def _clear_study_area(self):
        self._studiegebied = {}
        self._studiegebied_status.configure(text=self._format_studiegebied_status())
        self._schedule_recheck()

    # -- Project page (declarative, minus paden + motief file fields) ------

    def _build_project_page(self, page):
        scroll = ScrollableArea(page)
        scroll.pack(side=TOP, fill="both", expand=True)
        project_template = self._template["project"]
        filtered_motief = {k: v for k, v in project_template["motief"].items()
                           if k not in ("reizende populatie", "bestemmingsplaatsen")}
        filtered_project = {k: v for k, v in project_template.items() if k != "paden"}
        filtered_project["motief"] = filtered_motief
        build._addWidgets(scroll.body, filtered_project)

    # -- Run page -----------------------------------------------------------

    def _build_run_page(self, page):
        frame = Frame(page)
        frame.pack(side=TOP, fill="both", expand=True, padx=10, pady=10)

        steps_frame = LabelFrame(frame, text="Welke stappen wil je uitvoeren?", padx=8, pady=8)
        steps_frame.pack(fill="x", pady=(0, 10))
        self._step_checks = [BooleanVar(value=True) for _ in RUN_STEP_LABELS]
        for label, var in zip(RUN_STEP_LABELS, self._step_checks):
            Checkbutton(steps_frame, text=label, variable=var, anchor="w",
                       justify=LEFT, wraplength=800).pack(fill="x", anchor="w")

        options_frame = Frame(frame)
        options_frame.pack(fill="x", pady=(0, 10))
        self._write_weights_var = BooleanVar(value=False)
        Checkbutton(options_frame, text="Schrijf ook de (grote) gewichtenbestanden weg op schijf",
                   variable=self._write_weights_var).pack(anchor="w")

        self._start_btn = Button(frame, text="▶ Start", command=self._start_run,
                                 bg="#16a34a", fg="white", font=("TkDefaultFont", 11, "bold"))
        self._start_btn.pack(pady=(0, 10))

        log_frame = Frame(frame)
        log_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=RIGHT, fill="y")
        self._run_log = Text(log_frame, height=12, state="disabled", wrap="word", yscrollcommand=scrollbar.set)
        self._run_log.pack(side=LEFT, fill="both", expand=True)
        scrollbar.config(command=self._run_log.yview)

    def _build_navbar(self, master):
        bar = Frame(master)
        bar.pack(side=BOTTOM, fill="x")
        self._back_btn = Button(bar, text="◀ Vorige", command=self._go_back)
        self._back_btn.pack(side=LEFT, padx=10, pady=8)
        self._next_btn = Button(bar, text="Volgende ▶", command=self._go_next)
        self._next_btn.pack(side=RIGHT, padx=10, pady=8)
        self._step_label = Label(bar, text="")
        self._step_label.pack(side=TOP)

    # -- navigation ---------------------------------------------------------

    def _show_step(self, idx):
        idx = max(0, min(idx, len(self._steps) - 1))
        self._current_index = idx
        step = self._steps[idx]
        self._pages[step.key].tkraise()
        self._step_label.configure(text=f"Stap {idx + 1} van {len(self._steps)}: {step.title}")
        self._back_btn.configure(state="normal" if idx > 0 else "disabled")
        is_last = idx == len(self._steps) - 1
        self._next_btn.configure(text="Start" if is_last else "Volgende ▶")
        for i, s in enumerate(self._steps):
            row = self._sidebar_rows[s.key]
            bg = "#dbeafe" if i == idx else self._default_bg
            row["frame"].configure(background=bg)
            row["label"].configure(background=bg)
        self._schedule_recheck()

    def _go_back(self):
        self._show_step(self._current_index - 1)

    def _go_next(self):
        if self._current_index == len(self._steps) - 1:
            self._start_run()
        else:
            self._show_step(self._current_index + 1)

    # -- live validation ------------------------------------------------------

    def _bind_traces(self, template):
        for key in template:
            if key == "label":
                continue
            node = template[key]
            if not isinstance(node, dict):
                continue
            if "tkvar" in node:
                var = node["tkvar"]
                if isinstance(var, list):
                    for v in var:
                        v.trace_add("write", self._schedule_recheck)
                else:
                    var.trace_add("write", self._schedule_recheck)
            elif "type" not in node:
                self._bind_traces(node)

    def _schedule_recheck(self, *_args):
        if self._recheck_job is not None:
            self.after_cancel(self._recheck_job)
        self._recheck_job = self.after(RECHECK_DELAY_MS, self._recheck_all)

    def _augmented_config(self):
        config = build.buildConfigDict(self._template)
        config["project"]["paden"]["segs_mapping"] = self._current_segs_mapping()
        config["project"]["paden"]["studiegebied"] = self._studiegebied
        return config

    def _recheck_all(self):
        self._recheck_job = None
        try:
            config = self._augmented_config()
        except Exception as err:
            for step in self._steps:
                self._set_step_status(step.key, StepStatus.ERROR,
                                      [f"Kon configuratie niet samenstellen: {err}"])
            return
        self._dirty = True
        self._update_title()
        for step in self._steps:
            status, messages = step.checker(config)
            self._set_step_status(step.key, status, messages)

    def _set_step_status(self, key, status, messages):
        self._status[key] = (status, messages)
        row = self._sidebar_rows[key]
        row["dot"].itemconfigure(row["oval"], fill=STATUS_COLORS[status])

        banner = self._page_banners[key]
        if status == StepStatus.OK:
            text = "✓ " + ("  |  ".join(messages) if messages else "Alles gecontroleerd, geen problemen gevonden.")
        elif status == StepStatus.WARNING:
            text = "⚠ " + "\n⚠ ".join(messages) if messages else "⚠ Let op."
        elif status == StepStatus.ERROR:
            text = "✗ " + "\n✗ ".join(messages) if messages else "✗ Er ontbreekt iets."
        else:
            text = "Nog niet gecontroleerd."
        banner.configure(text=text, background=STATUS_BANNER_BG[status])

        if key == "run":
            self._refresh_run_start_state()

    # -- new / load / save --------------------------------------------------

    def _confirm_discard(self):
        return messagebox.askyesno(
            title="Niet-opgeslagen wijzigingen",
            message="Er zijn niet-opgeslagen wijzigingen. Doorgaan zonder opslaan?",
        )

    def _update_title(self):
        label = self._current_path or "(niet opgeslagen)"
        if self._dirty:
            label += " *"
        self.title(f"IKOB wizard — {label}")
        if hasattr(self, "_path_label"):
            self._path_label.configure(text=label)

    def _apply_loaded_config(self, config):
        build.setTkVars(self._template, config)
        self._load_segs_mapping(config.get("project", {}).get("paden", {}).get("segs_mapping", {}))
        self._studiegebied = config.get("project", {}).get("paden", {}).get("studiegebied", {}) or {}
        if hasattr(self, "_studiegebied_status"):
            self._studiegebied_status.configure(text=self._format_studiegebied_status())
        self._toggle_zonedata_format()

    def _cmd_new(self):
        if self._dirty and not self._confirm_discard():
            return
        self._apply_loaded_config(default_config())
        self._current_path = None
        self._dirty = False
        self._update_title()
        self._recheck_all()

    def _cmd_load(self):
        if self._dirty and not self._confirm_discard():
            return
        filename = filedialog.askopenfilename(
            title="Kies een .json project bestand.", filetypes=[("project file", ".json")]
        )
        if not filename:
            return
        try:
            loaded = load_config(filename)
        except ValueError as err:
            messagebox.showerror(title="Fout", message=f"Het bestand bevat geen geldige configuratie:\n{err}")
            return
        except IOError as err:
            messagebox.showerror(title="Fout", message=f"Het bestand kan niet worden geladen:\n{err}")
            return
        self._apply_loaded_config(loaded)
        self._current_path = filename
        self._dirty = False
        self._update_title()
        self._recheck_all()
        if loaded.get("project", {}).get("paden", {}).get("segs_format") == "gpkg":
            messagebox.showinfo(
                title="GeoPackage-project geladen",
                message="Klik op 'Lees lagen' en 'Laad kolommen' in stap 1 om de "
                       "kolomkeuzelijsten opnieuw te vullen (de eerder opgeslagen "
                       "keuzes zelf zijn al hersteld).",
            )

    def _cmd_save(self):
        self._do_save(always_prompt=True)

    def _do_save(self, always_prompt=False) -> bool:
        try:
            config = self._augmented_config()
        except Exception as err:
            messagebox.showerror(title="Fout", message=f"Kon de configuratie niet samenstellen vanuit het formulier:\n{err}")
            return False

        filename = self._current_path
        if always_prompt or filename is None:
            try:
                suggested = project_name(config)
            except Exception:
                suggested = "project"
            filename = filedialog.asksaveasfilename(
                title="Kies een .json project bestand.",
                initialfile=_project_filename(suggested),
                filetypes=[("project file", ".json")],
            )
            if not filename:
                return False
            filename = _project_filename(filename, make_safe=False)

        try:
            config_is_valid, messages = saveConfig(filename, config)
        except ConfigSaveError as err:
            messagebox.showerror(title="Fout", message=str(err))
            return False
        except Exception as err:
            messagebox.showerror(title="Fout", message=f"Onverwachte fout bij het opslaan:\n{err}")
            return False

        self._current_path = filename
        self._dirty = False
        self._update_title()
        self._recheck_all()

        if not config_is_valid:
            detail = "\n".join(f"• {m}" for m in messages) if messages else "Zie de stap 'Uitvoeren' voor details."
            messagebox.showwarning(
                title="Opgeslagen",
                message=f"Configuratie opgeslagen naar '{filename}', maar met waarschuwingen:\n\n{detail}",
            )
        return True

    # -- run --------------------------------------------------------------

    def _refresh_run_start_state(self):
        if not hasattr(self, "_start_btn"):
            return
        status, _ = self._status.get("run", (StepStatus.UNKNOWN, []))
        self._start_btn.configure(state="disabled" if status == StepStatus.ERROR else "normal")

    def _append_run_log(self, text):
        self._run_log.configure(state="normal")
        self._run_log.insert("end", text)
        self._run_log.see("end")
        self._run_log.configure(state="disabled")

    def _start_run(self):
        if self._current_index != len(self._steps) - 1:
            self._show_step(len(self._steps) - 1)
            return

        if self._current_path is None or self._dirty:
            if not self._do_save(always_prompt=self._current_path is None):
                return

        skip_steps = [not var.get() for var in self._step_checks]
        write_weights = self._write_weights_var.get()
        project_file = self._current_path

        self._start_btn.configure(state="disabled")
        self._append_run_log(f"Start uitvoering van '{project_file}'…\n")

        def _worker():
            try:
                run_scripts(project_file, skip_steps, write_weights=write_weights)
            except BaseException as err:
                self.after(0, self._on_run_finished, False, str(err))
            else:
                self.after(0, self._on_run_finished, True, None)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_run_finished(self, success, error_message):
        self._refresh_run_start_state()
        if success:
            self._append_run_log("Alle stappen zijn succesvol uitgevoerd.\n")
            messagebox.showinfo(title="Gereed", message="Alle stappen zijn succesvol uitgevoerd.")
        else:
            self._append_run_log(f"FOUT: {error_message}\n")
            messagebox.showerror(title="Fout", message=f"Er is een fout opgetreden:\n{error_message}")


def main():
    parser = argparse.ArgumentParser(
        prog="ikobwizard", description="Launch the combined IKOB configuration & run wizard."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Display logging messages over stdout.")
    parser.add_argument("-p", "--project", help="Optioneel: laad dit .json projectbestand bij het opstarten.")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            stream=sys.stdout, level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s \t -  %(message)s",
        )

    if not validate.validateTemplate(default_configuration_definition()):
        messagebox.showerror(title="Fout", message="De standaard configuratiedefinitie is niet geldig.")
        sys.exit(1)

    app = WizardApp(initial_project=args.project)
    app.mainloop()


if __name__ == "__main__":
    main()