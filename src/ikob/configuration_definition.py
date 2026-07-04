import logging
from enum import Enum, StrEnum

from ikob.config import build
from ikob.utils import IKOB_INFINITE

logger = logging.getLogger(__name__)


class DataType(Enum):
    CHECKBOX = "checkbox"
    CHECKLIST = "checklist"
    CHOICE = "choice"
    DIRECTORY = "directory"
    FILE = "file"
    NUMBER = "number"
    TEXT = "text"


def config_item(
    label: str, data_type: DataType, default: str = "", items: list[str] = [], bounds: list[str] = [], unit: str = ""
):
    msg = "Invalid GUI data type provided."
    assert data_type in DataType, msg

    default_values = {DataType.CHECKBOX: False, DataType.NUMBER: 0}

    if not default:
        default = default_values.get(data_type, default)

    # The default value is expected as list when more items are present.
    if data_type != DataType.CHOICE:
        if items and isinstance(default, str):
            default = [default]

    dictionary = {"label": label, "type": data_type.value, "default": default}

    # Insert all optional values when present.
    keys = ["items", "unit", "bounds"]
    optionals = [items, unit, bounds]
    for key, optional in zip(keys, optionals):
        if optional:
            dictionary[key] = optional

    return dictionary


class DecayCurveName(StrEnum):
    WORK_AND_SOCIAL = "werk en sociaal-recreatief"
    DAILY_SHOPPING_AND_HEALTH = "dagelijkse boodschappen en zorg"
    NON_DAILY_SHOPPING_AND_EDUCATION = "niet-dagelijkse boodschappen en onderwijs"


class TvomType(StrEnum):
    WORK = "werk"
    OTHER = "overig"


def default_project_tab():
    return {
        "label": "Project",
        "naam": config_item("Project naam", DataType.TEXT, default="Project 1"),
        "verstedelijkingsscenario": config_item(
            "Welk verstedelijkingsscenario wordt gebruikt",
            DataType.TEXT,
        ),
        "beprijzingsregime": config_item(
            "Wat is de naam van het beprijzingsregime",
            DataType.TEXT,
            default="Basis",
        ),
        "paden": {
            "label": "Paden",
            "skims_directory": config_item(
                "Basis directory voor CSV-skims (genegeerd als skims.skims_bron niet 'bestanden' is)",
                DataType.DIRECTORY,
            ),
            "segs_format": config_item(
                "Formaat van de SEGS/CBS-brondata",
                DataType.CHOICE,
                default="csv",
                items=["csv", "gpkg"],
            ),
            "segs_directory": config_item("SEGS directory (csv-formaat)", DataType.DIRECTORY),
            "segs_bestand": config_item(
                "SEGS/CBS GeoPackage-bestand (gpkg-formaat)", DataType.FILE
            ),
            "segs_buurten_laag": config_item(
                "Laagnaam met de referentie-buurtgeometrie in het GeoPackage",
                DataType.TEXT,
                default="buurten",
            ),
            "segs_buurtcode_kolom": config_item(
                "Kolomnaam met de (CBS-)buurtcode in elke laag van het GeoPackage",
                DataType.TEXT,
                default="buurtcode",
            ),
            "output_directory": config_item("Output directory", DataType.DIRECTORY, default="output"),
        },
        "motief": {
            "naam": config_item("Naam van het motief", DataType.TEXT, default="werk"),
            "reizende populatie": config_item(
                "Populatie bestand voor dit motief", DataType.FILE, default="Beroepsbevolking_inkomensklasse.csv"
            ),
            "bestemmingsplaatsen": config_item(
                "Bestemmingen bestand voor dit motief", DataType.FILE, default="Arbeidsplaatsen_inkomensklasse.csv"
            ),
            "TVOM": config_item(
                "De te gebruiken tijdswaarde van geld (TVOM tab)",
                DataType.CHOICE,
                default=TvomType.WORK,
                items=list(TvomType),
            ),
            "reistijdvervalscurve": config_item(
                "De te gebruiken reistijd vervalscurve",
                DataType.CHOICE,
                default=DecayCurveName.WORK_AND_SOCIAL,
                items=list(DecayCurveName),
            ),
            "tolerantiecurven": config_item(
                "Curve-bibliotheek (optioneel, .json van de tolerantiecurve-editor) — "
                "overschrijft de reistijdvervalscurve voor de groepen die erin staan",
                DataType.FILE,
                default="",
            ),
        },
        "fiets of E-fiets": {
            "label": "Rekenen met Fiets of E-fiets",
            "E-fiets": config_item(
                "Met E-fiets",
                DataType.CHECKBOX,
            ),
        },
        "welke_inkomensgroepen": config_item(
            "Welke inkomensgroepen moeten worden meegenomen",
            DataType.CHECKLIST,
            default=["laag", "middellaag", "middelhoog", "hoog"],
            items=["laag", "middellaag", "middelhoog", "hoog"],
        ),
    }


def default_skims_tab():
    return {
        "label": "Gegeneraliseerde Reistijd Berekenen",
        "dagsoort": config_item(
            "Dagsoorten",
            DataType.CHECKLIST,
            default="Restdag",
            items=["Ochtendspits", "Restdag", "Avondspits"],
        ),
        "skims_bron": config_item(
            "Bron van de skims (tijd/afstand/kosten-matrices)",
            DataType.CHOICE,
            default="bestanden",
            items=["bestanden", "r5py", "osrm"],
        ),
        "zone_locaties_bestand": config_item(
            "Zone-locaties voor route-gebaseerde skims (r5py/OSRM): csv met kolommen "
            "zone,lon,lat, of gpkg met een laag 'zones' (puntgeometrie + kolom "
            "'buurtcode'); leeg = gebruik buurt-centroïden uit de SEGS GeoPackage "
            "(vereist paden.segs_format = 'gpkg')",
            DataType.FILE,
            default="",
        ),
        "OV kosten": {
            "starttarief": config_item(
                "Starttarief",
                DataType.NUMBER,
                default=75,
                unit="Eurocent",
            ),
            "kmkosten": config_item(
                "Variabele kosten",
                DataType.NUMBER,
                default=12,
                unit="Eurocent/km",
            ),
        },
        "OV kostenbestand": {
            "label": "Bestaat er een apart OV-kostenbestand?",
            "gebruiken": config_item(
                "Er is een apart OV-kostenbestand",
                DataType.CHECKBOX,
            ),
        },
        "pricecap": {
            "label": "Is er een maximum OV-prijs (price cap)?",
            "gebruiken": config_item(
                "pricecap",
                DataType.CHECKBOX,
            ),
            "getal": config_item("Wat is de pricecap in Euros", DataType.NUMBER, default=IKOB_INFINITE),
        },
        "Kosten auto fossiele brandstof": {
            "variabele kosten": config_item(
                "variabele kosten",
                DataType.NUMBER,
                default=16,
                unit="Eurocent/km",
            ),
            "kmheffing": config_item(
                "Kilometerheffing",
                DataType.NUMBER,
                unit="Eurocent/km",
            ),
        },
        "Kosten elektrische auto": {
            "variabele kosten": config_item("variabele kosten", DataType.NUMBER, default=5, unit="Eurocent/km"),
            "kmheffing": config_item(
                "Kilometerheffing",
                DataType.NUMBER,
                unit="Eurocent/km",
            ),
        },
        "parkeerzoektijden_bestand": config_item(
            "Parkeerzoektijden bestand",
            DataType.FILE,
        ),
        "varkostenga": {
            "label": "Variabele kosten geen auto",
            "GeenAuto": config_item(
                "Deelauto (bezit geen auto, wel rijbewijs)",
                DataType.NUMBER,
                default=0.33,
                bounds=[0, IKOB_INFINITE],
                unit="Euro/km",
            ),
            "GeenRijbewijs": config_item(
                "Taxi (bezit geen rijbewijs)",
                DataType.NUMBER,
                default=2.40,
                bounds=[0, IKOB_INFINITE],
                unit="Euro/km",
            ),
        },
        "tijdkostenga": {
            "label": "Tijd kosten geen auto",
            "GeenAuto": config_item(
                "Deelauto (bezit geen auto, wel rijbewijs)",
                DataType.NUMBER,
                default=0.05,
                bounds=[0, IKOB_INFINITE],
                unit="Euro/Minuut",
            ),
            "GeenRijbewijs": config_item(
                "Taxi (bezit geen rijbewijs)",
                DataType.NUMBER,
                default=0.40,
                bounds=[0, IKOB_INFINITE],
                unit="Euro/Minuut",
            ),
        },
        "bike_cost_ct_per_km": config_item(
            "Fiets kosten of -vergoeding (negatief bedrag is vergoeding)",
            DataType.NUMBER,
            default=0.0,
            unit="Eurocent/km",
        ),
        "r5py": {
            "label": "Skims genereren met r5py (R5-routing op OSM + GTFS)",
            "osm_pbf": config_item("OpenStreetMap .osm.pbf bestand", DataType.FILE),
            "gtfs_directory": config_item("Map met GTFS .zip-bestanden (voor OV)", DataType.DIRECTORY),
            "vertrekdatum": config_item("Vertrekdatum (JJJJ-MM-DD) voor de routing", DataType.TEXT, default="2024-04-10"),
            "vertrektijd_ochtendspits": config_item("Vertrektijd ochtendspits (UU:MM)", DataType.TEXT, default="08:00"),
            "vertrektijd_restdag": config_item("Vertrektijd restdag (UU:MM)", DataType.TEXT, default="11:00"),
            "vertrektijd_avondspits": config_item("Vertrektijd avondspits (UU:MM)", DataType.TEXT, default="17:00"),
            "max_reistijd_minuten": config_item("Maximale reistijd om te routeren", DataType.NUMBER, default=180),
        },
        "osrm": {
            "label": "Skims genereren met OSRM (alleen auto en fiets, geen OV)",
            "auto_server": config_item(
                "OSRM-server URL voor auto (profiel 'driving')", DataType.TEXT, default="http://localhost:5000"
            ),
            "fiets_server": config_item(
                "OSRM-server URL voor fiets (profiel 'cycling')", DataType.TEXT, default="http://localhost:5001"
            ),
        },
    }


def default_tvom_tab():
    levels = ["Hoog", "Middelhoog", "Middellaag", "Laag"]
    werk_values = [4, 6, 9, 12]

    werk_levels = {
        level.lower(): config_item(level, DataType.NUMBER, default=value, unit="Minuten/Euro")
        for level, value in zip(levels, werk_values)
    }

    overig_values = [4.8, 7.25, 10.9, 15.5]
    overig_levels = {
        level.lower(): config_item(level, DataType.NUMBER, default=value, unit="Minuten/Euro")
        for level, value in zip(levels, overig_values)
    }

    return {
        "label": "Waarde van tijd",
        TvomType.WORK: {
            "label": "Waarde van 1€ kosten in gegeneraliseerde reistijd per inkomensgroep, motief werk",
            **werk_levels,
        },
        TvomType.OTHER: {
            "label": "Waarde van 1€ kosten in gegeneraliseerde reistijd per inkomensgroep, motief overig",
            **overig_levels,
        },
    }


def default_verdeling_tab():
    levels = ["Laag", "Middellaag", "Middelhoog", "Hoog"]

    electric_share = {level.lower(): config_item(level, DataType.NUMBER, unit="%") for level in levels}

    return {
        "label": "Verdeling Over Groepen",
        "Percelektrisch": electric_share,
        "GratisOVpercentage": config_item(
            "Gratis OV",
            DataType.NUMBER,
            default=0.03,
            bounds=[0, 100],
            unit="(fractie)",
        ),
    }


def default_advanced_tab():
    additionele_kosten_label = "Additionele kosten, dit zijn extra kosten die gemaakt worden bij bijvoorbeeld een cordonheffing, waarbij voor sommige verplaatsingen wel extra kosten gelden en voor andere verplaatsingen niet (bedragen in eurocenten)."

    return {
        "label": "Geavanceerd",
        "kunstmab": {
            "label": "Kunstmatig autobezit (afgedwongen lager autobezit bv door strenge parkeernormen)",
            "gebruiken": config_item(
                "Gebruik kunstmatig autobezit",
                DataType.CHECKBOX,
            ),
            "bestand": config_item(
                "Kunstmatig autobezit bestand",
                DataType.FILE,
            ),
        },
        "parkeerkosten": {
            "label": "Is er een bestand met parkeerkosten per zone?",
            "gebruiken": config_item(
                "Parkeerkosten",
                DataType.CHECKBOX,
            ),
            "bestand": config_item(
                "Parkeerkosten bestand (bedragen zijn in eurocenten (dus €2,20 wordt weergegeven als 220)",
                DataType.FILE,
            ),
        },
        "additionele_kosten": {
            "label": additionele_kosten_label,
            "gebruiken": config_item(
                "Additionele kosten",
                DataType.CHECKBOX,
            ),
            "bestand": config_item(
                "Additionele kosten bestand",
                DataType.FILE,
            ),
        },
        "welke_groepen": config_item(
            "Welke groepen moeten worden meegenomen qua autobezit",
            DataType.CHECKLIST,
            default="alle groepen",
            items=["alle groepen", "alleen autobezitters"],
        ),
    }


def default_chains_and_hubs_tab():
    return {
        "label": "Ketens",
        "chains": {
            "label": "Definitie van de set hubs",
            "gebruiken": config_item(
                "Wel ketens en hubs",
                DataType.CHECKBOX,
            ),
            "bestand": config_item(
                "Bestand met de hubs",
                DataType.FILE,
            ),
            "naam hub": config_item(
                "Wat is de naam van de verzameling hubs?",
                DataType.TEXT,
            ),
        },
        "bestemmingslijst": {
            "label": "bestemmingslijst gebruiken",
            "gebruiken": config_item("bestemmingslijst", DataType.CHECKBOX),
            "bestand": config_item(
                "bestand met de bestemmingslijst",
                DataType.FILE,
            ),
        },
    }


def default_configuration_definition():
    """
    The default configuration definition for IKOB.

    The configuration contains the label attribute:
      - label: The label text for an input field, tab, or frame.

    For each leaf in the configuration additional attributes are defined:
      - type (required): the kind of input:
          text
          number
          directory
          file
          checkbox
          checklist
          choice
      - unit: a label after the input field for ``text`` and ``number``
      - default: the default input value
      - items: a list of items to choose from
      - range: the minimum and maximum allowed values for type ``number``
    """

    project_tab = default_project_tab()
    skims_tab = default_skims_tab()
    tvom_tab = default_tvom_tab()
    verdeling_tab = default_verdeling_tab()
    chains_and_hubs_tab = default_chains_and_hubs_tab()
    advanced_tab = default_advanced_tab()

    return {
        "project": project_tab,
        "skims": skims_tab,
        "TVOM": tvom_tab,
        "verdeling": verdeling_tab,
        "ketens": chains_and_hubs_tab,
        "geavanceerd": advanced_tab,
    }


def project_name(config):
    """Extract the project name from the project configuration."""
    return config["project"]["naam"]


def default_config():
    """Provide the configuration using the default config definition."""
    template = default_configuration_definition()
    config = build.buildConfigDict(template)
    return config