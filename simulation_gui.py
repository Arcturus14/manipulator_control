"""Tkinter front end for configuring and launching manipulator simulations."""
from __future__ import annotations

import copy
import json
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

from utils.yaml_config import DEFAULT_CONFIG, dump_config, load_config, validate_config


ROOT = Path(__file__).resolve().parent
TRAJECTORIES = ("circle", "infinity", "rectangle", "complex_3d_wave", "multiscale_3d")
CONTROLLERS = ("mppi", "ilqr", "sqp")
OBSTACLES = ("none", "static", "dynamic")
BENCHMARKS = ("baseline", "static", "dynamic", "hardware", "mean_update")
OUTPUT_PATTERN = re.compile(r"^\[OUTPUT\]\s+(.+)$")
# Edit this constant to change the mean-update value exposed by the GUI.
GUI_MEAN_UPDATE_ALPHA = 0.9


def _csv(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _parse_csv(value: str, converter: Callable[[str], Any], label: str) -> list[Any]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{label}: enter at least one value.")
    try:
        return [converter(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"{label}: enter valid comma-separated values.") from error


def _nested_get(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        value = value[part]
    return value


def _nested_set(data: dict[str, Any], dotted: str, value: Any) -> None:
    target = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


class SimulationGUI:
    """Configuration editor and non-blocking launcher for ``main.py``."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Manipulator Control Simulation")
        self.root.geometry("1180x960")
        self.root.minsize(980, 820)

        self.config = load_config(None, {})
        self.variables: dict[str, tk.Variable] = {}
        self.parsers: dict[str, Callable[[str], Any]] = {}
        self.widgets: dict[str, ttk.Widget] = {}
        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.temp_config: Path | None = None
        self.output_directory: Path | None = None

        self.command_var = tk.StringVar(value="run")
        self.mean_update_enabled_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.preset_var = tk.StringVar()
        self.result_var = tk.StringVar(value="Output: not created yet")

        self._configure_style()
        self._build_layout()
        self._load_into_form(self.config)
        self._refresh_state()
        self.root.after(100, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("TkDefaultFont", 15, "bold"))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("Run.TButton", font=("TkDefaultFont", 10, "bold"))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Manipulator Simulation Launcher", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        preset = ttk.Frame(outer)
        preset.pack(fill="x", pady=(0, 8))
        ttk.Label(preset, text="Preset").pack(side="left")
        preset_files = sorted((ROOT / "configs").glob("*.yaml"))
        self.preset_paths = {path.name: path for path in preset_files}
        combo = ttk.Combobox(
            preset, textvariable=self.preset_var,
            values=list(self.preset_paths), state="readonly", width=38,
        )
        combo.pack(side="left", padx=6)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._load_preset())
        ttk.Button(preset, text="Open YAML", command=self._open_yaml).pack(side="left", padx=2)
        ttk.Button(preset, text="Save YAML", command=self._save_yaml).pack(side="left", padx=2)
        ttk.Button(preset, text="Defaults", command=self._reset_defaults).pack(side="left", padx=2)
        ttk.Button(preset, text="Preview configuration", command=self._preview).pack(side="right")

        mode = ttk.LabelFrame(outer, text="Run mode", padding=8, style="Section.TLabelframe")
        mode.pack(fill="x", pady=(0, 8))
        for text, value in (("Single run", "run"), ("Comparison", "compare"), ("Benchmark", "benchmark")):
            ttk.Radiobutton(
                mode, text=text, value=value, variable=self.command_var,
                command=self._refresh_state,
            ).pack(side="left", padx=(0, 18))

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.experiment_tab = ttk.Frame(self.notebook, padding=10)
        self.controller_tab = ttk.Frame(self.notebook, padding=10)
        self.cost_tab = ttk.Frame(self.notebook, padding=10)
        self.obstacle_tab = ttk.Frame(self.notebook, padding=10)
        self.benchmark_tab = ttk.Frame(self.notebook, padding=10)
        self.output_tab = ttk.Frame(self.notebook, padding=10)
        for frame, title in (
            (self.experiment_tab, "Experiment"), (self.controller_tab, "Controller"),
            (self.cost_tab, "Costs"), (self.obstacle_tab, "Obstacles"),
            (self.benchmark_tab, "Benchmark"), (self.output_tab, "Output"),
        ):
            self.notebook.add(frame, text=title)

        self._build_experiment_tab()
        self._build_controller_tab()
        self._build_cost_tab()
        self._build_obstacle_tab()
        self._build_benchmark_tab()
        self._build_output_tab()
        self._build_monitor(outer)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(10, 0))
        self.run_button = ttk.Button(controls, text="Run", style="Run.TButton", command=self._start)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        self.open_button = ttk.Button(
            controls, text="Open output folder", command=self._open_result, state="disabled"
        )
        self.open_button.pack(side="left")
        ttk.Label(controls, textvariable=self.result_var).pack(side="left", padx=12)

    def _register(self, path: str, kind: type = str) -> tk.Variable:
        if kind is bool:
            variable: tk.Variable = tk.BooleanVar()
            self.parsers[path] = lambda value: value.lower() == "true"
        else:
            variable = tk.StringVar()
            self.parsers[path] = kind
        self.variables[path] = variable
        return variable

    @staticmethod
    def _section(parent: ttk.Frame, title: str, column: int, row: int, colspan: int = 1) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=10, style="Section.TLabelframe")
        frame.grid(row=row, column=column, columnspan=colspan, sticky="nsew", padx=5, pady=5)
        return frame

    def _entry(self, parent: ttk.Frame, row: int, label: str, path: str,
               kind: type = float, width: int = 18) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=self._register(path, kind), width=width)
        self.widgets[path] = entry
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
        parent.columnconfigure(1, weight=1)
        return entry

    def _combo(self, parent: ttk.Frame, row: int, label: str, path: str,
               values: tuple[str, ...], callback: Callable[[], None] | None = None) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        combo = ttk.Combobox(
            parent, textvariable=self._register(path), values=values,
            state="readonly", width=22,
        )
        self.widgets[path] = combo
        combo.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)
        if callback:
            combo.bind("<<ComboboxSelected>>", lambda _event: callback())
        parent.columnconfigure(1, weight=1)
        return combo

    def _check(self, parent: ttk.Frame, row: int, text: str, path: str) -> ttk.Checkbutton:
        check = ttk.Checkbutton(parent, text=text, variable=self._register(path, bool))
        self.widgets[path] = check
        check.grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        return check

    def _build_experiment_tab(self) -> None:
        tab = self.experiment_tab
        tab.columnconfigure((0, 1), weight=1)
        basic = self._section(tab, "Basic experiment", 0, 0)
        self._combo(basic, 0, "Controller", "controller.type", CONTROLLERS, self._refresh_state)
        self._combo(basic, 1, "Trajectory", "experiment.trajectory", TRAJECTORIES)
        self._entry(basic, 2, "Duration [s]", "experiment.duration")
        self._entry(basic, 3, "Random seed", "controller.seed", int)
        self._check(basic, 4, "Enable disturbance", "experiment.disturbance")
        self._check(basic, 5, "Enable DOB", "experiment.dob")

        sim = self._section(tab, "Simulation", 1, 0)
        self._entry(sim, 0, "Control period dt [s]", "simulation.dt")
        ttk.Label(
            sim, text="Changing dt also changes the controller deadline and integration step.",
            foreground="#666666", wraplength=380,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        comparison = self._section(tab, "Comparison", 0, 1, colspan=2)
        self.comparison_frame = comparison
        self._combo(comparison, 0, "Comparison type", "comparison.type", ("controller", "dob"), self._refresh_state)
        self._combo(comparison, 1, "Controller A", "comparison.controller_a", CONTROLLERS)
        self._combo(comparison, 2, "Controller B", "comparison.controller_b", CONTROLLERS)
        ttk.Label(
            comparison,
            text="DOB comparison uses the basic controller selected above and runs DOB OFF/ON.",
            foreground="#666666",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_controller_tab(self) -> None:
        tab = self.controller_tab
        tab.columnconfigure((0, 1), weight=1)
        mppi = self._section(tab, "MPPI", 0, 0)
        self._entry(mppi, 0, "Horizon", "controller.horizon", int)
        self._entry(mppi, 1, "Samples", "controller.samples", int)
        self._entry(mppi, 2, "Temperature λ", "controller.temperature")
        ttk.Checkbutton(
            mppi,
            text=f"Enable mean update (fixed α={GUI_MEAN_UPDATE_ALPHA:g})",
            variable=self.mean_update_enabled_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=3)
        self._entry(mppi, 4, "Noise variance J1..J6", "controller.noise_variance", str, 34)
        self._entry(mppi, 5, "Temporal noise rho", "controller.temporal_noise_rho")

        local = self._section(tab, "Local optimizer", 1, 0)
        self._entry(local, 0, "iLQR max iterations", "controller.ilqr_max_iterations", int)
        self._entry(local, 1, "SQP max iterations", "controller.sqp_max_iterations", int)

        note = self._section(tab, "Robot model", 0, 1, colspan=2)
        ttk.Label(
            note,
            text=("Torque/velocity limits, damping, DH, mass, and inertia values are preserved "
                  "from the loaded YAML and are not edited in the GUI."),
            foreground="#666666", wraplength=800,
        ).grid(row=0, column=0, sticky="w")

    def _build_cost_tab(self) -> None:
        tab = self.cost_tab
        tab.columnconfigure((0, 1, 2), weight=1)
        labels = (
            ("position", "Position"), ("terminal", "Terminal"),
            ("velocity", "Velocity"), ("action", "Action"),
            ("smoothness", "Smoothness"),
            ("obstacle_smoothness", "Obstacle smoothness"),
        )
        for column, controller in enumerate(CONTROLLERS):
            frame = self._section(tab, controller.upper(), column, 0)
            for row, (key, label) in enumerate(labels):
                self._entry(frame, row, label, f"cost.{controller}.{key}")

    def _build_obstacle_tab(self) -> None:
        tab = self.obstacle_tab
        tab.columnconfigure((0, 1), weight=1)
        common = self._section(tab, "Obstacle mode", 0, 0, colspan=2)
        self._combo(common, 0, "Mode", "obstacle.mode", OBSTACLES, self._refresh_state)
        self._entry(common, 1, "Static obstacle count", "obstacle.count", int)

        self.static_frame = self._section(tab, "Static obstacle", 0, 1)
        self._obstacle_entries(self.static_frame, "static")
        self.dynamic_frame = self._section(tab, "Dynamic obstacle", 1, 1)
        self._obstacle_entries(self.dynamic_frame, "dynamic")

    def _obstacle_entries(self, frame: ttk.Frame, mode: str) -> None:
        prefix = f"obstacle.{mode}"
        self._entry(frame, 0, "Cube size [m]", f"{prefix}.size")
        self._entry(frame, 1, "MPPI weight", f"{prefix}.mppi_weight")
        self._entry(frame, 2, "iLQR weight", f"{prefix}.ilqr_weight")
        self._entry(frame, 3, "SQP weight", f"{prefix}.sqp_weight")
        self._entry(frame, 4, "Barrier gain", f"{prefix}.gain")
        self._entry(frame, 5, "Margin [m]", f"{prefix}.margin")
        self._entry(frame, 6, "Activation zone [m]", f"{prefix}.zone")
        if mode == "dynamic":
            self._entry(frame, 7, "Speed ratio", f"{prefix}.speed_ratio")
            self._entry(frame, 8, "Phase offset", f"{prefix}.phase_offset")

    def _build_benchmark_tab(self) -> None:
        tab = self.benchmark_tab
        tab.columnconfigure((0, 1), weight=1)
        selection = self._section(tab, "Benchmark configuration", 0, 0)
        self._combo(selection, 0, "Type", "benchmark.type", BENCHMARKS)
        self._entry(selection, 1, "Trajectories", "benchmark.trajectories", str, 38)
        self._entry(selection, 2, "Seeds", "benchmark.seeds", str, 38)
        self._entry(selection, 3, "Cases", "benchmark.cases", str, 38)
        self._entry(selection, 4, "Dynamic weights", "benchmark.weights", str, 38)
        self._entry(selection, 5, "Mean-update α values", "benchmark.mean_update_alphas", str, 38)

        flags = self._section(tab, "Benchmark output", 1, 0)
        self._check(flags, 0, "Require CUDA", "benchmark.require_cuda")
        self._check(flags, 1, "Save representative raw result", "benchmark.save_result")
        ttk.Label(
            flags,
            text="Separate list values with commas, for example: seeds = 7, 17, 27",
            foreground="#666666", wraplength=360,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_output_tab(self) -> None:
        tab = self.output_tab
        tab.columnconfigure(0, weight=1)
        output = self._section(tab, "Output options", 0, 0)
        self._entry(output, 0, "Output root", "output.root", str, 50)
        self._check(output, 1, "Show animation after completion", "output.show")
        self._check(output, 2, "Generate joint torque plot", "output.torque_plot")
        self._check(
            output, 3, r"Plot disturbance estimation error |d - d_hat|",
            "output.disturbance_error_plot",
        )

    def _build_monitor(self, parent: ttk.Frame) -> None:
        log_frame = ttk.LabelFrame(
            parent, text="Execution monitor", padding=8, style="Section.TLabelframe"
        )
        log_frame.pack(fill="x", pady=(8, 0))
        columns = ("case", "rmse", "max_error", "mean_time", "worst_time", "dob")
        self.metrics = ttk.Treeview(log_frame, columns=columns, show="headings", height=4)
        headings = (
            ("case", "Case", 130), ("rmse", "RMSE [mm]", 100),
            ("max_error", "Max error [mm]", 110), ("mean_time", "Mean [ms]", 100),
            ("worst_time", "Worst [ms]", 100), ("dob", "DOB RMS [Nm]", 110),
        )
        for name, title, width in headings:
            self.metrics.heading(name, text=title)
            self.metrics.column(name, width=width, anchor="center")
        self.metrics.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame, height=8, wrap="word", state="disabled",
            font=("DejaVu Sans Mono", 9),
        )
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")

    def _load_into_form(self, cfg: dict[str, Any]) -> None:
        self.config = copy.deepcopy(cfg)
        self.mean_update_enabled_var.set(
            float(cfg["controller"]["mean_update_alpha"]) < 1.0
        )
        for path, variable in self.variables.items():
            if path == "comparison.controller_a":
                value = cfg["comparison"]["controllers"][0]
            elif path == "comparison.controller_b":
                value = cfg["comparison"]["controllers"][1]
            else:
                value = _nested_get(cfg, path)
            variable.set(_csv(value) if isinstance(value, list) else value)
        self._refresh_state()

    def _config_from_form(self) -> dict[str, Any]:
        cfg = copy.deepcopy(self.config)
        csv_fields: dict[str, Callable[[str], Any]] = {
            "controller.noise_variance": float,
            "benchmark.trajectories": str,
            "benchmark.seeds": int,
            "benchmark.cases": str,
            "benchmark.weights": float,
            "benchmark.mean_update_alphas": float,
        }
        labels = {
            "controller.noise_variance": "Noise variance",
            "benchmark.trajectories": "Benchmark trajectories",
            "benchmark.seeds": "Benchmark seeds",
            "benchmark.cases": "Benchmark cases",
            "benchmark.weights": "Benchmark weights",
            "benchmark.mean_update_alphas": "Mean-update alpha",
        }
        for path, variable in self.variables.items():
            if path.startswith("comparison.controller_"):
                continue
            raw = variable.get()
            if path in csv_fields:
                value = _parse_csv(str(raw), csv_fields[path], labels[path])
            elif isinstance(variable, tk.BooleanVar):
                value = bool(raw)
            else:
                try:
                    value = self.parsers[path](str(raw).strip())
                except ValueError as error:
                    raise ValueError(f"{path}: enter a valid numeric value.") from error
            _nested_set(cfg, path, value)

        cfg["comparison"]["controllers"] = [
            str(self.variables["comparison.controller_a"].get()),
            str(self.variables["comparison.controller_b"].get()),
        ]
        cfg["controller"]["mean_update_alpha"] = (
            GUI_MEAN_UPDATE_ALPHA if self.mean_update_enabled_var.get() else 1.0
        )
        validate_config(cfg)
        if (self.command_var.get() == "compare"
                and cfg["comparison"]["type"] == "controller"
                and cfg["comparison"]["controllers"][0] == cfg["comparison"]["controllers"][1]):
            raise ValueError("Select two different controllers for a controller comparison.")
        return cfg

    def _load_preset(self) -> None:
        path = self.preset_paths.get(self.preset_var.get())
        if path is not None:
            self._load_path(path)

    def _open_yaml(self) -> None:
        name = filedialog.askopenfilename(
            title="Open configuration YAML", initialdir=ROOT / "configs",
            filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*")),
        )
        if name:
            self._load_path(Path(name))

    def _load_path(self, path: Path) -> None:
        try:
            cfg = load_config(path, {})
        except Exception as error:
            messagebox.showerror("Configuration error", str(error), parent=self.root)
            return
        self._load_into_form(cfg)
        stem = path.stem.lower()
        self.command_var.set("benchmark" if "benchmark" in stem else "compare" if "compare" in stem else "run")
        self.status_var.set(f"Loaded: {path.name}")
        self._refresh_state()

    def _save_yaml(self) -> None:
        try:
            cfg = self._config_from_form()
        except ValueError as error:
            messagebox.showerror("Configuration error", str(error), parent=self.root)
            return
        name = filedialog.asksaveasfilename(
            title="Save configuration YAML", initialdir=ROOT / "configs",
            defaultextension=".yaml", filetypes=(("YAML", "*.yaml"),),
        )
        if name:
            dump_config(cfg, Path(name))
            self.config = copy.deepcopy(cfg)
            self.status_var.set(f"Saved: {Path(name).name}")

    def _reset_defaults(self) -> None:
        self.preset_var.set("")
        self._load_into_form(load_config(DEFAULT_CONFIG, {}))
        self.command_var.set("run")
        self.status_var.set("Defaults restored")
        self._refresh_state()

    def _preview(self) -> None:
        try:
            cfg = self._config_from_form()
        except ValueError as error:
            messagebox.showerror("Configuration error", str(error), parent=self.root)
            return
        command = f"{sys.executable} -u main.py {self.command_var.get()} --config <GUI_CONFIG.yaml>"
        window = tk.Toplevel(self.root)
        window.title("Configuration preview")
        window.geometry("760x680")
        text = tk.Text(window, wrap="none")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text.insert("end", f"# command\n{command}\n\n")
        text.insert("end", yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        text.configure(state="disabled")

    @staticmethod
    def _set_children_state(frame: ttk.Frame, state: str) -> None:
        for child in frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _refresh_state(self) -> None:
        if not hasattr(self, "comparison_frame"):
            return
        compare = self.command_var.get() == "compare"
        self.widgets["comparison.type"].configure(state="readonly" if compare else "disabled")
        comparison_type = str(self.variables["comparison.type"].get())
        controller_state = "readonly" if compare and comparison_type == "controller" else "disabled"
        self.widgets["comparison.controller_a"].configure(state=controller_state)
        self.widgets["comparison.controller_b"].configure(state=controller_state)
        mode = str(self.variables.get("obstacle.mode", tk.StringVar(value="none")).get())
        self._set_children_state(self.static_frame, "normal" if mode == "static" else "disabled")
        self._set_children_state(self.dynamic_frame, "normal" if mode == "dynamic" else "disabled")
        benchmark_state = "normal" if self.command_var.get() == "benchmark" else "disabled"
        try:
            self.notebook.tab(self.benchmark_tab, state=benchmark_state)
        except tk.TclError:
            pass

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self) -> None:
        if self.process is not None:
            return
        try:
            cfg = self._config_from_form()
        except ValueError as error:
            messagebox.showerror("Configuration error", str(error), parent=self.root)
            return

        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="manipulator_gui_",
            encoding="utf-8", delete=False,
        )
        try:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
        finally:
            handle.close()
        self.temp_config = Path(handle.name)
        command = [
            sys.executable, "-u", str(ROOT / "main.py"),
            self.command_var.get(), "--config", str(self.temp_config),
        ]
        self._append_log("\n$ " + " ".join(command) + "\n")
        try:
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            self.process = None
            self._cleanup_temp()
            messagebox.showerror("Launch failed", str(error), parent=self.root)
            return

        self.config = copy.deepcopy(cfg)
        self.output_directory = None
        for item in self.metrics.get_children():
            self.metrics.delete(item)
        self.open_button.configure(state="disabled")
        self.result_var.set("Output: waiting for output directory")
        self.status_var.set("Running")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.reader_thread = threading.Thread(target=self._read_process, daemon=True)
        self.reader_thread.start()

    def _read_process(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.events.put(("log", line))
            match = OUTPUT_PATTERN.match(line.strip())
            if match:
                self.events.put(("output", match.group(1)))
        return_code = process.wait()
        self.events.put(("done", return_code))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self._append_log(value)
                elif kind == "output":
                    path = Path(value)
                    self.output_directory = path if path.is_absolute() else ROOT / path
                    self.result_var.set(f"Output: {self.output_directory}")
                    self.open_button.configure(state="normal")
                elif kind == "done":
                    self._finish(int(value))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _finish(self, return_code: int) -> None:
        self.process = None
        self.reader_thread = None
        self._cleanup_temp()
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if return_code == 0:
            self.status_var.set("Completed")
            self._append_log("\n[GUI] Run completed.\n")
            self._show_metrics()
        elif return_code < 0:
            self.status_var.set("Stopped")
            self._append_log("\n[GUI] Run stopped.\n")
        else:
            self.status_var.set(f"Failed (exit {return_code})")
            self._append_log(f"\n[GUI] Run failed with exit code {return_code}.\n")

    def _show_metrics(self) -> None:
        if self.output_directory is None:
            return
        metrics_path = self.output_directory / "metrics.json"
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        rows = 0
        if isinstance(data, dict):
            for case, values in data.items():
                if not isinstance(values, dict) or "rmse_err_mm" not in values:
                    continue
                def shown(name: str) -> str:
                    value = values.get(name)
                    return "-" if value is None else f"{float(value):.3f}"
                self.metrics.insert("", "end", values=(
                    case, shown("rmse_err_mm"), shown("max_err_mm"),
                    shown("mean_solve_ms"), shown("worst_solve_ms"),
                    shown("dob_rms_nm"),
                ))
                rows += 1
        if rows == 0:
            self._append_log(f"[GUI] Detailed benchmark metrics: {metrics_path}\n")

    def _stop(self) -> None:
        process = self.process
        if process is None:
            return
        if not messagebox.askyesno("Stop run", "Stop the current run?", parent=self.root):
            return
        self.status_var.set("Stopping")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.root.after(3000, lambda pid=process.pid: self._force_stop(pid))

    def _force_stop(self, pid: int) -> None:
        process = self.process
        if process is None or process.pid != pid or process.poll() is not None:
            return
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _open_result(self) -> None:
        path = self.output_directory
        if path is None or not path.exists():
            messagebox.showwarning("Output folder", "The output folder does not exist.", parent=self.root)
            return
        try:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as error:
            messagebox.showerror("Failed to open folder", str(error), parent=self.root)

    def _cleanup_temp(self) -> None:
        if self.temp_config is not None:
            try:
                self.temp_config.unlink(missing_ok=True)
            except OSError:
                pass
            self.temp_config = None

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno(
                    "Quit", "A simulation is running. Stop it and close the GUI?",
                    parent=self.root):
                return
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self._cleanup_temp()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
