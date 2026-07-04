import argparse
import logging
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import BooleanVar, Button, Frame, StringVar, Tk, Widget, filedialog, messagebox

from ikob.combined_weights import calculate_combined_weights
from ikob.competition import competition_on_citizens, competition_on_destinations
from ikob.config import validate, widgets
from ikob.datasource import DataSource, DataType
from ikob.distribute_over_groups import distribute_population_over_groups
from ikob.generalized_travel_time import generalized_travel_time
from ikob.ikobconfig import get_config_from_args, load_config
from ikob.reachable_destinations import reachable_destinations
from ikob.reachable_population import reachable_population
from ikob.single_weights import calculate_single_weights
from ikob.tolerance_curves import CurveRegistry

logger = logging.getLogger(__name__)


def _load_curve_registry(config) -> CurveRegistry | None:
    """Load the optional curve-library attached to this project's motief
    config (see configuration_definition.default_project_tab), if any."""
    curve_lib_path = config.get("project", {}).get("motief", {}).get("tolerantiecurven", "")
    if not curve_lib_path:
        return None
    registry = CurveRegistry.from_json(curve_lib_path)
    logger.info("Loaded curve-library %s with %d group attachment(s).", curve_lib_path, len(registry))
    return registry


def run_scripts(project_file, skip_steps: list[bool] | None = None, write_weights: bool = False):
    """
    Run through all IKOB steps.

    ROUND-2 CHANGES:
    - Combined weights are lazy (recipes only) → ~50 % less peak RAM
    - GTT cache cleared immediately after D2
    - D4/D5 and D6/D7 still run in parallel
    - Optional curve-library attachments (project.motief.tolerantiecurven)
      override the legacy decay curve for specific groups -- see
      ikob.curve_attachment.
    """
    logger.info("Reading project file: %s.", project_file)
    config = get_config_from_args(project_file)

    valid = validate.FileValidator(config).validate_input_files()
    if not valid:
        raise ValueError("Invalid input files, see console warnings.")

    curve_registry = _load_curve_registry(config)

    logger.info("Starting simulations...")
    if not skip_steps:
        skip_steps = [False] * 8

    # ── D1: Generalized travel time ──
    if not skip_steps[0]:
        travel_time = generalized_travel_time(config, curve_registry=curve_registry)
    else:
        travel_time = DataSource(config, DataType.GENERALIZED_TRAVEL_TIME)

    # ── B: Distribute population over groups ──
    if not skip_steps[1]:
        distribute_population_over_groups(config)

    # ── D2: Single weights ──
    if not skip_steps[2]:
        single_weights = calculate_single_weights(config, travel_time, curve_registry=curve_registry)
    else:
        single_weights = DataSource(config, DataType.WEIGHTS)

    # GTT is no longer needed – free immediately
    if not skip_steps[0]:
        if write_weights:
            travel_time.store()
        logger.info("GTT cache: %.1f MB – clearing.", travel_time.cache_size_mb())
        travel_time.clear_cache()

    # ── D3: Combined weights (now lazy – milliseconds, 0 bytes) ──
    if not skip_steps[3]:
        combined_weights = calculate_combined_weights(config, single_weights)
    else:
        combined_weights = DataSource(config, DataType.WEIGHTS)

    logger.info("Single-weights cache: %.1f MB.  Combined-weight recipes: %s.",
                single_weights.cache_size_mb(),
                combined_weights.recipe_count() if hasattr(combined_weights, 'recipe_count') else "N/A (loaded)")

    # ── D4 & D5: run in parallel ──
    def _run_d4():
        if not skip_steps[4]:
            return reachable_destinations(config, single_weights, combined_weights)
        return DataSource(config, DataType.DESTINATIONS)

    def _run_d5():
        if not skip_steps[5]:
            return reachable_population(config, single_weights, combined_weights)
        return DataSource(config, DataType.ORIGINS)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_d4 = pool.submit(_run_d4)
        future_d5 = pool.submit(_run_d5)
        opportunities = future_d4.result()
        origins = future_d5.result()

    # ── D6 & D7: run in parallel ──
    def _run_d6():
        if not skip_steps[6]:
            return competition_on_destinations(config, single_weights, combined_weights, origins)
        return DataSource(config, DataType.COMPETITION)

    def _run_d7():
        if not skip_steps[7]:
            return competition_on_citizens(config, single_weights, combined_weights, opportunities)
        return DataSource(config, DataType.COMPETITION)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_d6 = pool.submit(_run_d6)
        future_d7 = pool.submit(_run_d7)
        competition_destinations = future_d6.result()
        competition_citizens = future_d7.result()

    logger.info("All simulations completed.")

    # ── Write output ──
    logger.info("Writing output to disk...")
    sources_to_save = [opportunities, origins, competition_citizens, competition_destinations]
    if write_weights:
        single_weights.store()
        combined_weights.store()          # materialises recipes one-by-one

    for container in sources_to_save:
        container.store()

    DataSource.write_output_md(config)

    # Final cleanup
    for container in sources_to_save:
        container.clear_cache()
    single_weights.clear_cache()
    combined_weights.clear_cache()
    logger.info("All output written and memory released.")


# ── User interface (unchanged) ───────────────────────────────────────

class ConfigApp(Tk):
    PAD_X = 5
    PAD_Y = 5

    stappen = (
        "Gegeneraliseerde reistijd berekenen uit tijd en kosten",
        "Verdeling van de groepen over de buurten of zones",
        "Gewichten (reistijdvervalscurven) voor auto, OV, fiets en E-fiets apart",
        "Maximum gewichten van meerdere modaliteiten",
        "Bereikbaarheid arbeidsplaatsen voor inwoners",
        "Potentie bereikbaarheid voor bedrijven en instellingen",
        "Concurrentiepositie voor bereik arbeidsplaatsen",
        "Concurrentiepositie voor bedrijven qua bereikbaarheid",
    )

    def __init__(self):
        super().__init__()
        self.title("IKOB Runner")
        self._checks = [BooleanVar(value=True) for _ in self.stappen]
        self._configvar = StringVar()
        self.run_button = None
        self.create_widgets()

    def create_widgets(self):
        self.widgets: list[Widget] = []
        frame = Frame()
        frame.pack(expand=1, fill="both", padx=self.PAD_X, pady=self.PAD_Y)
        self.widgets.extend(widgets.pathWidget(frame, "Project", self._configvar, file=True))
        self.widgets.append(frame)
        labels = [stap for stap in self.stappen]
        self.widgets.extend(widgets.checklistWidget(frame, "Stappen", labels, self._checks, row=1, itemsperrow=1))
        button = Button(master=frame, text="Start", command=lambda: threading.Thread(target=self.run_cmd).start())
        button.grid(row=2, column=2, sticky="ew", padx=self.PAD_X, pady=self.PAD_Y)
        self.run_button = button
        self.widgets.append(button)

    def run_cmd(self):
        project_file = self._configvar.get()
        skip_steps = [not check.get() for check in self._checks]
        if self.run_button is None:
            raise ValueError("attempt to disable run button, but run button is None.")
        self.run_button.configure(state="disabled")
        try:
            run_scripts(project_file, skip_steps, write_weights=False)
        except BaseException as err:
            msg = f"An error occurred: {err}"
            messagebox.showerror(title="FOUT", message=msg)
        else:
            msg = "Alle stappen zijn succesvol uitgevoerd."
            messagebox.showinfo(title="Gereed", message=msg)
        self.run_button.configure(state="active")

    def cmdLaadProject(self):
        filename = filedialog.askopenfilename(
            title="Kies een .json project bestand.",
            filetypes=[("project file", ".json")],
        )
        if filename:
            try:
                _ = load_config(filename)
            except ValueError:
                messagebox.showerror(title="Fout", message="Het bestand bevat geen geldige configuratie.")
            except IOError:
                messagebox.showerror(title="Fout", message="Het bestand kan niet worden geladen.")


def main():
    parser = argparse.ArgumentParser(prog="ikobrunner", description="Launch the IKOB runner GUI.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Display logging messages over stdout.")
    parser.add_argument("-p", "--project",
                        help="Optional path to the project to execute. No GUI is shown if provided.")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            stream=sys.stdout, level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s \t -  %(message)s"
        )
    if not args.project:
        App = ConfigApp()
        App.mainloop()
    else:
        try:
            run_scripts(args.project)
        except BaseException:
            logger.error(traceback.format_exc())
        else:
            logger.info("Alle steps successfully executed.")


if __name__ == "__main__":
    main()