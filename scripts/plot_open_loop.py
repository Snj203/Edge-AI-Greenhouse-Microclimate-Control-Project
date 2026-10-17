"""
The emulator without control (heater and vents off) on every weather profile,
plus a test of how the heater and the vents change the climate.

    python scripts/plot_open_loop.py   -> results/emulator/ (open_loop_overview.png,
                                          actuator_response.png, open_loop_summary.csv)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.emulator.simulation import create_emulator, run
from src.emulator.weather import ALL_PROFILES, build_weather
from src.utils.config import ensure_dir, load_config

COLORS = {"dataset": "#2d3748", "summer_clear": "#c53030", "winter_cloudy": "#2b6cb0", "spring_variable": "#2f855a"}


# Key numbers of one run without control
def summarize(profile, log, emu, cfg) -> dict:
    target = cfg["target"]["temperature"]
    hours_per_step = cfg["emulator"]["timestep_minutes"] / 60
    t = log["temperature"]
    return {
        "profile": profile,
        "outdoor_min_c": round(log["outdoor_temp"].min(), 1), "outdoor_max_c": round(log["outdoor_temp"].max(), 1),
        "indoor_min_c": round(t.min(), 1), "indoor_max_c": round(t.max(), 1),
        "rmse_vs_target_c": round(((t - target) ** 2).mean() ** 0.5, 2),
        "hours_below_20c": round((t < target - 2).sum() * hours_per_step, 1),
        "hours_above_24c": round((t > target + 2).sum() * hours_per_step, 1),
        "humidity_range": f"{log['humidity'].min():.0f}-{log['humidity'].max():.0f}",
        "irrigations": emu.irrigation_count,
    }


# All weather profiles on one figure
def plot_overview(logs: dict, target: float, path: Path):
    panels = [("temperature", "Temperature (°C)"), ("humidity", "Humidity (%RH)"), ("vpd", "VPD (kPa)"),
              ("light_ppfd", "PPFD (µmol/m²/s)"), ("soil_moisture", "Soil moisture (%)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 13), sharex=True)
    for profile, log in logs.items():
        time = pd.to_datetime(log["time"])
        hours = (time - time.iloc[0]).dt.total_seconds() / 3600
        for ax, (col, label) in zip(axes, panels):
            ax.plot(hours, log[col], color=COLORS[profile], label=profile)
            ax.set_ylabel(label)
    axes[0].axhline(target, color="black", ls=":", lw=1)
    axes[0].legend(ncol=4, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("hour of day")
    axes[-1].set_xticks(range(0, 25, 3))
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Greenhouse climate without control, for each weather profile", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# The same spring day with the heater or the vents held at a fixed level
def plot_actuator_response(cfg, path: Path):
    weather = build_weather(cfg, "spring_variable")
    cases = {"off": (0, 0), "heater 50 %": (50, 0), "heater 100 %": (100, 0),
             "vent 50 %": (0, 50), "vent 100 %": (0, 100)}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, (heater, vent) in cases.items():
        log = run(create_emulator(weather, cfg), weather, lambda obs, k, h=heater, v=vent: (h, v))
        time = pd.to_datetime(log["time"])
        axes[0].plot(time, log["temperature"], label=name)
        axes[1].plot(time, log["humidity"], label=name)
    axes[0].axhline(cfg["target"]["temperature"], color="black", ls=":", lw=1)
    axes[0].set_ylabel("Temperature (°C)")
    axes[1].set_ylabel("Humidity (%RH)")
    axes[0].legend(ncol=5, fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Actuator response — spring day, constant heater / vent settings", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Run every profile without control, save the table and the figures
def main():
    cfg = load_config()
    out = ensure_dir("results/emulator")
    logs, rows = {}, []
    for profile in ALL_PROFILES:
        weather = build_weather(cfg, profile)
        emu = create_emulator(weather, cfg)
        logs[profile] = run(emu, weather)
        rows.append(summarize(profile, logs[profile], emu, cfg))

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "open_loop_summary.csv", index=False)
    print(summary.to_string(index=False))

    plot_overview(logs, cfg["target"]["temperature"], out / "open_loop_overview.png")
    plot_actuator_response(cfg, out / "actuator_response.png")
    print(f"\nPlots saved to {out}")


if __name__ == "__main__":
    main()
