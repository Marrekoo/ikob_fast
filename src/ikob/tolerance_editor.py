"""Tkinter editor for two-dimensional (time x money) tolerance curves.

Launch standalone:  python -m ikob.tolerance_editor [-v] [-b bibliotheek.json]

The saved JSON library is consumed by ikob.tolerance_curves.CurveRegistry
and is schema-compatible with the browser-based prototype editor.

Layout note: the control column can be taller than a small/laptop
screen, so the whole window content lives inside a vertically
scrollable frame (mouse wheel or the scrollbar) rather than requiring a
fixed minimum screen height.
"""

import argparse
import logging
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ikob.tolerance_curves import (
    CURVE_FAMILIES, KNOWN_GROUPS, MAX_TIME_MINUTES,
    MarginalCurve, ToleranceSpec,
    load_library, save_library, spec_from_dict, spec_to_dict, weight_matrix,
)

logger = logging.getLogger(__name__)

PALETTE = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e",
           "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"]
ZERO_COLOR = "#1a1a2e"

FAMILY_LABELS = {
    "logistic": "Logistisch (IKOB-sigmoïde)",
    "weibull": "Weibull",
    "exponential": "Exponentieel (geheugenloos)",
    "loglogistic": "Log-logistisch (zware staart)",
    "step": "Trapsgewijs (hazard-atomen)",
    "uniform": "Stuksgewijs lineair (uniforme dichtheid)",
    "triangular": "Stuksgewijs kwadratisch (driehoekige dichtheid)",
}
LABEL_TO_FAMILY = {v: k for k, v in FAMILY_LABELS.items()}
PAD = {"padx": 4, "pady": 2}


def _param_defs(family, max_x, unit):
    """Return [(key, label, lo, hi, resolution, default), ...] for *family*."""
    half = round(max_x * 0.35 * 2) / 2
    if family == "logistic":
        return [
            ("a", "α  (steilheid)", round(1.8 / max_x, 4), round(120.0 / max_x, 4),
             round(0.6 / max_x, 4), round(6.0 / max(half, 1e-9), 3)),
            ("b", f"ω  (halfwaarde-tolerantie, {unit})", 0.5, max_x, 0.5, half),
        ]
    if family == "weibull":
        return [
            ("a", "k  (vorm)", 0.3, 8.0, 0.1, 2.0),
            ("b", f"λ  (schaal, {unit})", 0.5, max_x, 0.5, half),
        ]
    if family == "exponential":
        return [
            ("b", f"λ  (gemiddelde tolerantie, {unit})", 0.5, max_x, 0.5, half),
        ]
    if family == "loglogistic":
        return [
            ("a", "k  (vorm)", 0.3, 8.0, 0.1, 3.0),
            ("b", f"λ  (mediane tolerantie, {unit})", 0.5, max_x, 0.5, half),
        ]
    if family == "step":
        return [
            ("a", f"c₁  (eerste atoom, {unit})", 0.0, max_x, 0.5,
             round(max_x * 0.3 * 2) / 2),
            ("b", "p  (plateauniveau na c₁; 0 = harde cutoff bij c₁)", 0.0, 1.0, 0.01, 0.0),
            ("c", f"c₂  (eindcutoff, atoom van massa p, {unit})", 0.0, max_x, 0.5,
             round(max_x * 0.6 * 2) / 2),
        ]
    if family == "uniform":
        return [
            ("a", f"L  (onset — volledige tolerantie eronder, {unit})", 0.0, max_x, 0.5,
             round(max_x * 0.2 * 2) / 2),
            ("b", f"U  (cutoff — nul tolerantie erboven, {unit})", 0.0, max_x, 0.5,
             round(max_x * 0.6 * 2) / 2),
        ]
    return [
        ("a", f"L  (onset, {unit})", 0.0, max_x, 0.5, round(max_x * 0.15 * 2) / 2),
        ("b", f"U  (cutoff, {unit})", 0.0, max_x, 0.5, round(max_x * 0.65 * 2) / 2),
        ("c", "modus-positie  (0 = convex, 1 = concaaf)", 0.0, 1.0, 0.01, 0.5),
    ]


class _ScrollableRoot(ttk.Frame):
    """A vertically scrollable container filling its master.

    The control column (curve panels + coupling sliders) can be taller
    than a small screen; this makes the whole window scroll (wheel or
    scrollbar) instead of getting clipped or forcing a minimum screen
    height on the user.
    """

    def __init__(self, master):
        super().__init__(master)
        self._canvas = tk.Canvas(self, highlightthickness=0)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vbar.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.body = ttk.Frame(self._canvas)
        window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>",
                       lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfigure(window, width=e.width))

        def _wheel(event):
            if sys.platform == "darwin":
                self._canvas.yview_scroll(int(-event.delta), "units")
            else:
                self._canvas.yview_scroll(int(-event.delta / 120), "units")

        def _wheel_linux(event):
            self._canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

        self._canvas.bind_all("<MouseWheel>", _wheel)
        self._canvas.bind_all("<Button-4>", _wheel_linux)
        self._canvas.bind_all("<Button-5>", _wheel_linux)


class CurvePanel(ttk.LabelFrame):
    """Family selector + dynamic parameter sliders + three small plots
    (survival, hazard, density) so a marginal can be related to observed
    drop-out behaviour (e.g. 'a uniform drop-out between x and y minutes'
    shows up as a flat density and a rising hazard toward the cutoff)."""

    MINI_W, MINI_H = 92, 92
    MAX_SLIDERS = 3

    def __init__(self, master, title, unit, max_x, color, on_change):
        super().__init__(master, text=title)
        self.unit = unit
        self.color = color
        self.max_x = max_x
        self._on_change = on_change

        self._family = "logistic"
        self._combo = ttk.Combobox(self, state="readonly",
                                   values=[FAMILY_LABELS[f] for f in CURVE_FAMILIES])
        self._combo.set(FAMILY_LABELS[self._family])
        self._combo.bind("<<ComboboxSelected>>", self._family_changed)
        self._combo.pack(fill="x", **PAD)

        self._sliders = []
        for _ in range(self.MAX_SLIDERS):
            label = ttk.Label(self)
            var = tk.DoubleVar()
            scale = tk.Scale(self, variable=var, orient="horizontal",
                             showvalue=True, command=self._changed)
            self._sliders.append({"label": label, "var": var, "scale": scale, "key": None})

        self._plot_frame = ttk.Frame(self)
        self._plot_s = tk.Canvas(self._plot_frame, width=self.MINI_W, height=self.MINI_H,
                                 bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self._plot_h = tk.Canvas(self._plot_frame, width=self.MINI_W, height=self.MINI_H,
                                 bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self._plot_f = tk.Canvas(self._plot_frame, width=self.MINI_W, height=self.MINI_H,
                                 bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self._plot_s.grid(row=0, column=0, padx=2)
        self._plot_h.grid(row=0, column=1, padx=2)
        self._plot_f.grid(row=0, column=2, padx=2)
        self._plot_frame.pack(**PAD)

        self._interactive = [self._combo] + [s["scale"] for s in self._sliders]
        self._apply_family(self._family, reset_params=True)

    # -- family / parameters -----------------------------------------

    def _current_values(self):
        return {slot["key"]: float(slot["var"].get())
                for slot in self._sliders if slot["key"] is not None}

    def _apply_family(self, family, reset_params):
        self._family = family
        defs = _param_defs(family, self.max_x, self.unit)
        current = {} if reset_params else self._current_values()

        for slot in self._sliders:
            slot["label"].pack_forget()
            slot["scale"].pack_forget()
            slot["key"] = None

        for slot, (key, label, lo, hi, step, default) in zip(self._sliders, defs):
            slot["key"] = key
            slot["label"].configure(text=label)
            slot["scale"].configure(from_=lo, to=hi, resolution=step, state="normal")
            value = default if reset_params else current.get(key, default)
            value = min(max(value, lo), hi)
            slot["var"].set(value)
            slot["label"].pack(anchor="w", **PAD)
            slot["scale"].pack(fill="x", **PAD)

        self._plot_frame.pack(**PAD)  # re-append below the (re)packed sliders
        self._draw_plot()

    def _family_changed(self, _event):
        self._apply_family(LABEL_TO_FAMILY[self._combo.get()], reset_params=True)
        self._on_change()

    def _changed(self, _value=None):
        self._draw_plot()
        self._on_change()

    # -- public API ----------------------------------------------------

    def get_curve(self) -> MarginalCurve:
        values = {"a": 0.0, "b": 0.0, "c": 0.0}
        values.update(self._current_values())
        return MarginalCurve(self._family, values["a"], values["b"], values["c"])

    def set_curve(self, curve: MarginalCurve):
        self._combo.set(FAMILY_LABELS[curve.family])
        self._apply_family(curve.family, reset_params=True)
        values = {"a": curve.a, "b": curve.b, "c": curve.c}
        for slot in self._sliders:
            if slot["key"] is not None:
                slot["var"].set(values[slot["key"]])
        self._draw_plot()

    def set_max_x(self, max_x):
        self.max_x = max_x
        self._apply_family(self._family, reset_params=False)

    def set_enabled(self, enabled: bool):
        state = "readonly" if enabled else "disabled"
        self._combo.configure(state=state)
        scale_state = "normal" if enabled else "disabled"
        for slot in self._sliders:
            if slot["key"] is not None:
                slot["scale"].configure(state=scale_state)

    # -- plotting --------------------------------------------------------

    @staticmethod
    def _robust_max(ys):
        finite = ys[np.isfinite(ys)]
        if finite.size == 0:
            return 1.0
        return float(max(np.nanpercentile(finite, 95), 1e-9))

    def _draw_mini(self, canvas, xs, ys, color, y_max, title, atoms=None):
        canvas.delete("all")
        w, h = self.MINI_W, self.MINI_H
        left, right, top, bottom = 4, w - 4, 13, h - 13
        canvas.create_text(w / 2, 2, text=title, anchor="n", font=("TkDefaultFont", 7))
        canvas.create_line(left, bottom, right, bottom, fill="#999999")
        canvas.create_line(left, top, left, bottom, fill="#999999")
        y_max = max(y_max, 1e-9)

        points = []
        for x, y in zip(xs, ys):
            yy = y_max if not np.isfinite(y) else min(max(float(y), 0.0), y_max)
            px = left + (x / self.max_x) * (right - left)
            py = top + (1.0 - yy / y_max) * (bottom - top)
            points += [px, py]
        if len(points) >= 4:
            canvas.create_line(*points, fill=color, width=1.5)

        for ax, mass in (atoms or []):
            px = left + (ax / self.max_x) * (right - left)
            canvas.create_line(px, top, px, bottom, fill="#dc2626", dash=(2, 2))
            canvas.create_text(px, top, text=f"{mass:.0%}", fill="#dc2626",
                               font=("TkDefaultFont", 6), anchor="s")

        canvas.create_text(right, bottom + 2, text=f"{self.max_x:g}{self.unit}",
                           anchor="ne", font=("TkDefaultFont", 6))

    def _draw_plot(self):
        xs = np.linspace(0.0, self.max_x, 241)
        curve = self.get_curve()
        s = curve.survival(xs)
        h = curve.hazard(xs)
        f = curve.density(xs)
        atoms = curve.atoms()

        self._draw_mini(self._plot_s, xs, s, self.color, 1.0, "S(x)")
        h_max = 1.0 if atoms else self._robust_max(h)
        f_max = 1.0 if atoms else self._robust_max(f)
        self._draw_mini(self._plot_h, xs, h, "#dc2626", h_max, "h(x)", atoms)
        self._draw_mini(self._plot_f, xs, f, "#ea580c", f_max, "f(x)", atoms)


class ToleranceEditorApp(tk.Tk):
    NX, NY, ZOOM = 140, 95, 4

    def __init__(self, library_path: pathlib.Path | None = None):
        super().__init__()
        self.title("IKOB tolerantiecurve-editor")
        self._library_path = library_path
        self._entries: list[dict] = []
        self._redraw_job = None
        self._img = None

        self._mode = tk.StringVar(value="copula")
        self._tau = tk.DoubleVar(value=9.0)
        self._theta = tk.DoubleVar(value=1.0)
        self._cutoff = tk.BooleanVar(value=True)
        self._max_money = tk.DoubleVar(value=25.0)
        self._probe = tk.StringVar(value="beweeg over het vlak om W te peilen")

        self._scroll = _ScrollableRoot(self)
        self._scroll.pack(fill="both", expand=True)
        body = self._scroll.body

        self._build_left(body)
        self._build_right(body)
        self._build_library(body)
        self._mode_changed()

        if library_path is not None and library_path.exists():
            try:
                self._entries = load_library(library_path)
                self._refresh_listbox()
                logger.info("Loaded %d curve sets from %s.", len(self._entries), library_path)
            except (ValueError, OSError) as err:
                messagebox.showwarning("Bibliotheek", f"Kon {library_path} niet laden:\n{err}")

        self._schedule_redraw()
        self._fit_to_screen()

    # -- layout -----------------------------------------------------------

    def _fit_to_screen(self):
        self.update_idletasks()
        req_w = self._scroll.body.winfo_reqwidth() + 24  # room for the scrollbar
        req_h = self._scroll.body.winfo_reqheight()
        screen_h = self.winfo_screenheight()
        height = min(req_h, screen_h - 80)
        self.geometry(f"{req_w}x{height}")

    def _build_left(self, master):
        left = ttk.Frame(master)
        left.grid(row=0, column=0, sticky="ns", **PAD)

        self._time_panel = CurvePanel(left, "Tijdtolerantie  S_T(t)", "min",
                                      MAX_TIME_MINUTES, "#2563eb", self._schedule_redraw)
        self._time_panel.pack(fill="x", **PAD)
        self._money_panel = CurvePanel(left, "Geldtolerantie  S_M(m)", "€",
                                       float(self._max_money.get()), "#059669",
                                       self._schedule_redraw)
        self._money_panel.pack(fill="x", **PAD)

        coupling = ttk.LabelFrame(left, text="Koppelingsstructuur")
        coupling.pack(fill="x", **PAD)
        ttk.Radiobutton(coupling, text="Vaste tijdswaarde — W = S_T(t + τ·m)  (huidige IKOB)",
                        variable=self._mode, value="fixedVOT",
                        command=self._mode_changed).pack(anchor="w", **PAD)
        self._tau_scale = tk.Scale(coupling, variable=self._tau, orient="horizontal",
                                   from_=1.0, to=30.0, resolution=0.5,
                                   label="τ  (minuten per euro — TVOM)",
                                   command=self._schedule_redraw)
        self._tau_scale.pack(fill="x", **PAD)
        ttk.Radiobutton(coupling, text="Tweedimensionaal — Gumbel-copula van S_T en S_M",
                        variable=self._mode, value="copula",
                        command=self._mode_changed).pack(anchor="w", **PAD)
        self._theta_scale = tk.Scale(coupling, variable=self._theta, orient="horizontal",
                                     from_=1.0, to=10.0, resolution=0.05,
                                     label="θ  (1 = onafhankelijk, groot = zwakste schakel)",
                                     command=self._schedule_redraw)
        self._theta_scale.pack(fill="x", **PAD)

        opts = ttk.Frame(coupling)
        opts.pack(fill="x", **PAD)
        ttk.Checkbutton(opts, text="IKOB-cutoffs (180 min, W < 0.001 → 0)",
                        variable=self._cutoff,
                        command=self._schedule_redraw).pack(side="left")
        ttk.Label(opts, text="   geld-as max €").pack(side="left")
        spin = ttk.Spinbox(opts, from_=5, to=100, increment=5, width=5,
                           textvariable=self._max_money, command=self._money_axis_changed)
        spin.bind("<Return>", lambda _e: self._money_axis_changed())
        spin.pack(side="left")

        self._degenerate_warning = ttk.Label(
            coupling, foreground="#92400e", background="#fffbeb",
            wraplength=340, justify="left",
            text="Beide marginalen zijn deterministisch (harde cutoffs): W is voor "
                 "elke θ dezelfde rechthoek. De koppelingsparameter is hier niet "
                 "identificeerbaar — voeg spreiding toe aan minstens één marginaal "
                 "als θ informatief moet zijn.")

    def _build_right(self, master):
        right = ttk.Frame(master)
        right.grid(row=0, column=1, sticky="n", **PAD)
        ttk.Label(right, text="W(t, m) op het (tijd × geld)-vlak — rechte evenwijdige "
                              "contouren ⇒ vaste tijdswaarde; kromming ⇒ imperfecte "
                              "substitutie; scherpe randen ⇒ hazard-atomen "
                              "(deterministische budgetten)", wraplength=560).pack(anchor="w", **PAD)
        self._canvas = tk.Canvas(right, width=self.NX * self.ZOOM,
                                 height=self.NY * self.ZOOM,
                                 highlightthickness=1, highlightbackground="#999999")
        self._canvas.pack(**PAD)
        self._canvas.bind("<Motion>", self._on_motion)
        self._axis_label = ttk.Label(right)
        self._axis_label.pack(anchor="w", **PAD)
        ttk.Label(right, textvariable=self._probe, font=("TkFixedFont", 9)).pack(anchor="w", **PAD)

        legend = tk.Canvas(right, width=self.NX * self.ZOOM, height=14,
                           highlightthickness=0)
        step = (self.NX * self.ZOOM) / len(PALETTE)
        for i, color in enumerate(PALETTE):
            legend.create_rectangle(i * step, 0, (i + 1) * step, 14,
                                    fill=color, outline="")
        legend.pack(**PAD)
        ttk.Label(right, text="W = 0  →  W = 1").pack(anchor="w", **PAD)

    def _build_library(self, master):
        lib = ttk.LabelFrame(master, text="Curve-bibliotheek")
        lib.grid(row=1, column=0, columnspan=2, sticky="ew", **PAD)

        row = ttk.Frame(lib)
        row.pack(fill="x", **PAD)
        ttk.Label(row, text="Naam:").pack(side="left")
        self._name = ttk.Entry(row, width=24)
        self._name.pack(side="left", **PAD)
        ttk.Label(row, text="Groepen (komma-gescheiden):").pack(side="left")
        self._groups = ttk.Entry(row, width=48)
        self._groups.pack(side="left", fill="x", expand=True, **PAD)
        ttk.Button(row, text="Opslaan in bibliotheek", command=self._save_entry).pack(side="left", **PAD)

        body = ttk.Frame(lib)
        body.pack(fill="x", **PAD)
        self._listbox = tk.Listbox(body, height=5)
        self._listbox.pack(side="left", fill="x", expand=True, **PAD)
        buttons = ttk.Frame(body)
        buttons.pack(side="left", fill="y")
        ttk.Button(buttons, text="Laden", command=self._load_entry).pack(fill="x", **PAD)
        ttk.Button(buttons, text="Verwijderen", command=self._delete_entry).pack(fill="x", **PAD)
        ttk.Button(buttons, text="Bibliotheek openen…", command=self._open_file).pack(fill="x", **PAD)
        ttk.Button(buttons, text="Bibliotheek opslaan…", command=self._save_file).pack(fill="x", **PAD)

    # -- spec <-> widgets ---------------------------------------------------

    def _current_spec(self) -> ToleranceSpec:
        return ToleranceSpec(
            time_curve=self._time_panel.get_curve(),
            money_curve=self._money_panel.get_curve(),
            mode=self._mode.get(),
            tau=float(self._tau.get()),
            theta=float(self._theta.get()),
            ikob_cutoff=bool(self._cutoff.get()),
        )

    def _apply_spec(self, spec: ToleranceSpec, geld_as_max=None):
        self._cutoff.set(spec.ikob_cutoff)
        if geld_as_max:
            self._max_money.set(geld_as_max)
            self._money_panel.set_max_x(float(geld_as_max))
        self._time_panel.set_curve(spec.time_curve)
        if spec.money_curve is not None:
            self._money_panel.set_curve(spec.money_curve)
        self._mode.set(spec.mode)
        if spec.mode == "fixedVOT":
            self._tau.set(spec.tau)
        else:
            self._theta.set(spec.theta)
        self._mode_changed()

    # -- event handlers -------------------------------------------------------

    def _mode_changed(self):
        fixed = self._mode.get() == "fixedVOT"
        self._money_panel.set_enabled(not fixed)
        self._tau_scale.configure(state="normal" if fixed else "disabled")
        self._theta_scale.configure(state="disabled" if fixed else "normal")
        self._schedule_redraw()

    def _money_axis_changed(self):
        try:
            max_money = max(5.0, float(self._max_money.get()))
        except (tk.TclError, ValueError):
            return
        self._money_panel.set_max_x(max_money)
        self._schedule_redraw()

    def _schedule_redraw(self, *_):
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(60, self._redraw)

    def _redraw(self):
        self._redraw_job = None
        spec = self._current_spec()
        max_money = max(5.0, float(self._max_money.get()))

        if spec.is_degenerate_copula():
            self._degenerate_warning.pack(fill="x", **PAD)
        else:
            self._degenerate_warning.pack_forget()

        ts = (np.arange(self.NX) + 0.5) / self.NX * MAX_TIME_MINUTES
        ms = (np.arange(self.NY) + 0.5) / self.NY * max_money
        T = np.broadcast_to(ts, (self.NY, self.NX))
        M = np.broadcast_to(ms[:, None], (self.NY, self.NX))
        W = weight_matrix(spec, T, M)[::-1]      # money axis points up

        bands = np.clip((W * 10).astype(int), 0, len(PALETTE) - 1)
        colors = np.asarray(PALETTE, dtype=object)[bands]
        colors[W <= 0.0] = ZERO_COLOR
        data = " ".join("{" + " ".join(row) + "}" for row in colors)

        base = tk.PhotoImage(width=self.NX, height=self.NY)
        base.put(data)
        self._img = base.zoom(self.ZOOM, self.ZOOM)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._img)
        self._axis_label.configure(
            text=f"x: 0 – {MAX_TIME_MINUTES:g} min    y: 0 – €{max_money:g}")

    def _on_motion(self, event):
        fx = event.x / (self.NX * self.ZOOM)
        fy = event.y / (self.NY * self.ZOOM)
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            return
        max_money = max(5.0, float(self._max_money.get()))
        t = fx * MAX_TIME_MINUTES
        m = (1.0 - fy) * max_money
        w = float(weight_matrix(self._current_spec(),
                                np.array([[t]]), np.array([[m]]))[0, 0])
        self._probe.set(f"t = {t:6.1f} min   m = €{m:6.2f}   W = {w:.4f}")

    # -- library ----------------------------------------------------------------

    def _refresh_listbox(self):
        self._listbox.delete(0, "end")
        for entry in self._entries:
            koppeling = entry["spec"]["koppeling"]
            mode = (f"vaste TVOM, τ={koppeling['tau']}" if koppeling["modus"] == "fixedVOT"
                    else f"copula, θ={koppeling['theta']}")
            groups = ", ".join(entry.get("groepen", [])) or "(geen groepen)"
            self._listbox.insert("end", f"{entry['naam']}  —  {mode}  —  {groups}")

    def _save_entry(self):
        name = self._name.get().strip()
        if not name:
            messagebox.showerror("Fout", "Geef de curveset eerst een naam.")
            return
        groups = [g.strip() for g in self._groups.get().split(",") if g.strip()]
        unknown = sorted(g for g in groups if g not in KNOWN_GROUPS)
        if unknown:
            messagebox.showerror(
                "Onbekende groep(en)",
                "Niet opgeslagen; deze groepsnamen bestaan niet:\n"
                + "\n".join(unknown)
                + "\n\nVerwacht patroon: bv. 'GeenAuto_vkOV_laag'.")
            return
        taken = {g: e["naam"] for e in self._entries if e["naam"] != name
                 for g in e.get("groepen", [])}
        clashes = sorted(f"{g} (al in {taken[g]!r})" for g in groups if g in taken)
        if clashes:
            messagebox.showerror("Dubbele koppeling",
                                 "Niet opgeslagen; groep(en) al gekoppeld:\n" + "\n".join(clashes))
            return

        spec_dict = spec_to_dict(self._current_spec())
        spec_dict["geld_as_max"] = float(self._max_money.get())
        self._entries = [e for e in self._entries if e["naam"] != name]
        self._entries.append({"naam": name, "groepen": groups, "spec": spec_dict})
        self._refresh_listbox()

    def _selected_entry(self):
        selection = self._listbox.curselection()
        if not selection:
            messagebox.showinfo("Bibliotheek", "Selecteer eerst een curveset in de lijst.")
            return None
        return self._entries[selection[0]]

    def _load_entry(self):
        entry = self._selected_entry()
        if entry is None:
            return
        self._apply_spec(spec_from_dict(entry["spec"]),
                         geld_as_max=entry["spec"].get("geld_as_max"))
        self._name.delete(0, "end")
        self._name.insert(0, entry["naam"])
        self._groups.delete(0, "end")
        self._groups.insert(0, ", ".join(entry.get("groepen", [])))

    def _delete_entry(self):
        entry = self._selected_entry()
        if entry is None:
            return
        self._entries = [e for e in self._entries if e is not entry]
        self._refresh_listbox()

    def _open_file(self):
        filename = filedialog.askopenfilename(
            title="Kies een .json curve-bibliotheek.",
            filetypes=[("curve bibliotheek", ".json")])
        if not filename:
            return
        try:
            self._entries = load_library(filename)
        except (ValueError, OSError, KeyError) as err:
            messagebox.showerror("Fout", f"Kon bibliotheek niet laden:\n{err}")
            return
        self._library_path = pathlib.Path(filename)
        self._refresh_listbox()

    def _save_file(self):
        initial = self._library_path.name if self._library_path else "toleranties.json"
        filename = filedialog.asksaveasfilename(
            title="Bibliotheek opslaan als…", initialfile=initial,
            defaultextension=".json", filetypes=[("curve bibliotheek", ".json")])
        if not filename:
            return
        try:
            save_library(filename, self._entries)
        except OSError as err:
            messagebox.showerror("Fout", f"Kon bibliotheek niet opslaan:\n{err}")
            return
        self._library_path = pathlib.Path(filename)
        messagebox.showinfo("Opgeslagen",
                            f"{len(self._entries)} curveset(s) opgeslagen naar {filename}.")


def main():
    parser = argparse.ArgumentParser(prog="ikobtoleranties",
                                     description="Launch the IKOB tolerance-curve editor GUI.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Display logging messages over stdout.")
    parser.add_argument("-b", "--bibliotheek", default="toleranties.json",
                        help="Pad naar de curve-bibliotheek (JSON). Wordt geladen indien aanwezig.")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            stream=sys.stdout, level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s \t -  %(message)s")

    app = ToleranceEditorApp(library_path=pathlib.Path(args.bibliotheek))
    app.mainloop()


if __name__ == "__main__":
    main()