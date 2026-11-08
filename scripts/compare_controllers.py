"""
First comparison of PID and NN-MPC on the 4 standard days (run run_pid_baseline.py and run_nn_mpc.py first).

    python scripts/compare_controllers.py     -> results/comparison.csv, results/comparison.png

Both controllers get the same instruction (hold 22 °C), start from the same warm greenhouse and see
the same weather. The table follows the plan: for every metric the PID, the NN-MPC and the difference,
as a total over the 4 days (energy, hours, degree-hours, movement), a mean (RMSE, compute time) or
the maximum (largest deviation, slowest step). The heater-only PID (the plan's baseline) is listed too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # figures go to files only, no windows (safe next to the MQTT threads)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.emulator.weather import ALL_PROFILES
from src.utils.config import resolve_path

METRICS = ["total_kwh", "heater_kwh", "vent_kwh", "temp_rmse_c", "temp_max_dev_c", "hours_in_band",
           "cold_degree_hours", "warm_degree_hours", "humidity_rmse_pct", "vpd_rmse_kpa", "heater_travel_pct",
           "latency_mean_ms", "latency_max_ms"]
SUM = {"total_kwh", "heater_kwh", "vent_kwh", "hours_in_band", "cold_degree_hours", "warm_degree_hours",
       "heater_travel_pct"}
MAX = {"temp_max_dev_c", "latency_max_ms"}
CONTROLLERS = ["pid_heat", "pid_heat_vent", "nn_mpc"]
COLORS = {"pid_heat": "#b9b8b3", "pid_heat_vent": "#52514e", "nn_mpc": "#2a78d6"}
LABELS = {"pid_heat": "PID, heater only", "pid_heat_vent": "PID, heater + vents", "nn_mpc": "NN-MPC"}
PANELS = [("total_kwh", "Energy per day (kWh)"), ("temp_rmse_c", "Temperature RMSE from 22 °C (°C)"),
          ("hours_in_band", "Hours in 21–23 °C"), ("warm_degree_hours", "Overheating above 23 °C (°C·h)")]


# Per metric: total / mean / max over the days for each controller, NN-MPC minus PID (heater + vents)
def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        agg = "sum" if metric in SUM else "max" if metric in MAX else "mean"
        values = {c: runs.loc[runs["controller"] == c, metric].agg(agg) for c in CONTROLLERS}
        pid, mpc = values["pid_heat_vent"], values["nn_mpc"]
        pct = 100 * (mpc - pid) / pid if pid and not metric.startswith("latency") else float("nan")
        rows.append({"metric": metric, "aggregate": agg, **values, "difference": mpc - pid, "difference_pct": pct})
    return pd.DataFrame(rows)


# Four bar charts: energy, RMSE, hours in the band and overheating, per weather profile and controller
def plot(runs: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    x = np.arange(len(ALL_PROFILES))
    width = 0.27
    for ax, (metric, title) in zip(axes.flat, PANELS):
        for j, c in enumerate(CONTROLLERS):
            values = [runs.loc[(runs["controller"] == c) & (runs["profile"] == p), metric].iloc[0] for p in ALL_PROFILES]
            ax.bar(x + (j - 1) * width, values, width, color=COLORS[c], label=LABELS[c])
        ax.set_xticks(x, ALL_PROFILES)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("PID vs NN-MPC, 24 hours per weather profile, same target 22 °C", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Read both result tables, build the comparison, save it and the figure
def main():
    pid = pd.read_csv(resolve_path("results/pid_baseline.csv"))
    mpc = pd.read_csv(resolve_path("results/nn_mpc_runs.csv"))
    runs = pd.concat([pid, mpc], ignore_index=True)
    table = comparison_table(runs)
    table.round(4).to_csv(resolve_path("results/comparison.csv"), index=False)
    plot(runs, resolve_path("results/comparison.png"))
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))
    print("\nSaved results/comparison.csv and results/comparison.png")


if __name__ == "__main__":
    main()
