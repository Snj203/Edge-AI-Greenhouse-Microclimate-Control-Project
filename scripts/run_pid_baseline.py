"""
PID baseline: 24-hour runs with both PID variants on every weather profile.

    python scripts/run_pid_baseline.py              # over the Mosquitto broker (lockstep), ~1 min
    python scripts/run_pid_baseline.py --offline    # direct calls, no broker needed

    pid_heat       heater only (the plan's baseline)
    pid_heat_vent  heater + vent loop (the fair baseline for the NN-MPC, which also controls both)

Every run is warm-started from the same heated greenhouse (src/controllers/experiment.py).

    -> results/pid_baseline.csv                  metrics per variant and profile
       results/pid/run_<variant>_<profile>.csv   per-step logs
       results/pid/pid_<profile>.png             temperature, deviation, heater and vents over the day
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.controllers.controller_pid import MODES, PIDController
from src.controllers.experiment import run_experiment
from src.emulator.weather import ALL_PROFILES
from src.utils.config import ensure_dir, load_config, resolve_path

INK_2 = "#52514e"
COLORS = {"pid_heat": "#2a78d6", "pid_heat_vent": "#eb6834"}
LABELS = {"pid_heat": "PID, heater only", "pid_heat_vent": "PID, heater + ventilation"}


# Both variants on one profile: temperature, deviation from the target, heater, vents
def plot_profile(logs: dict[str, pd.DataFrame], cfg: dict, profile: str, path: Path):
    target = cfg["target"]["temperature"]
    band = cfg["experiment"]["temperature_band"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1, 0.8, 0.8]})
    for name, log in logs.items():
        hours = log["step"].to_numpy() * cfg["emulator"]["timestep_minutes"] / 60
        color, label = COLORS[name], LABELS[name]
        axes[0].plot(hours, log["temperature"], color=color, lw=1.6, label=label)
        axes[1].plot(hours, log["temperature"] - target, color=color, lw=1.6, label=label)
        axes[2].step(hours, log["heater_power"], where="post", color=color, lw=1.6, label=label)
        axes[3].step(hours, log["ventilation_rate"], where="post", color=color, lw=1.6, label=label)
    axes[0].plot(hours, log["outdoor_temp"], color=INK_2, lw=1, ls="--", label="outdoor")
    axes[0].axhspan(target - band, target + band, color=INK_2, alpha=0.08, lw=0)
    axes[0].axhline(target, color=INK_2, lw=1, label=f"target {target:g} °C (band ± {band:g})")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(fontsize=8, ncol=4, loc="upper left")
    axes[1].axhspan(-band, band, color=INK_2, alpha=0.08, lw=0)
    axes[1].axhline(0, color=INK_2, lw=1)
    axes[1].set_ylabel("T − target (°C)")
    axes[2].set_ylabel("Heater (%)")
    axes[3].set_ylabel("Ventilation (%)")
    for ax in axes[2:]:
        ax.set_ylim(-5, 105)
    axes[-1].set_xlabel("hour of day")
    axes[-1].set_xticks(range(0, 25, 3))
    axes[-1].set_xlim(0, 24)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"PID baseline — {profile}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Run every variant on every profile, save the logs, the table and the plots
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="call the controller directly instead of over MQTT")
    parser.add_argument("--profiles", nargs="+", default=ALL_PROFILES, choices=ALL_PROFILES)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = parser.parse_args()

    cfg = load_config()
    out = ensure_dir("results/pid")
    p = cfg["pid"]
    print(f"PID gains: heater Kp {p['kp']} Ki {p['ki']} Kd {p['kd']}; ventilation Kp {p['ventilation']['kp']} "
          f"Ki {p['ventilation']['ki']} above {cfg['target']['temperature'] + p['ventilation']['setpoint_offset']} °C"
          f" — {'offline' if args.offline else 'over MQTT, lockstep'}\n")

    rows = []
    for profile in args.profiles:
        logs = {}
        for mode in args.modes:
            controller = PIDController.from_config(cfg, mode)
            try:
                log, m = run_experiment(controller, profile, cfg, mqtt=not args.offline)
            except ConnectionError as e:
                print(f"ERROR: {e}\n(use --offline to run without the broker)")
                return 1
            mqtt = log.attrs.get("mqtt")
            if mqtt and mqtt["timeouts"]:
                print(f"  WARNING: {mqtt['timeouts']} lockstep waits timed out — the run is not reproducible")
            log.to_csv(out / f"run_{controller.name}_{profile}.csv", index=False)
            logs[controller.name] = log
            rows.append({"controller": controller.name, "profile": profile, **m,
                         **({"mqtt_commands": mqtt["commands_received"], "mqtt_timeouts": mqtt["timeouts"]}
                            if mqtt else {})})
            print(f"  {profile:16s} {controller.name:14s} energy {m['total_kwh']:6.1f} kWh   "
                  f"T-RMSE {m['temp_rmse_c']:5.2f} °C   max dev {m['temp_max_dev_c']:5.2f} °C   "
                  f"in band {m['hours_in_band']:4.1f} h   compute {m['latency_mean_ms'] * 1000:5.1f} µs"
                  + (f"   MQTT {mqtt['commands_received']} commands, {mqtt['timeouts']} timeouts" if mqtt else ""))
        plot_profile(logs, cfg, profile, out / f"pid_{profile}.png")

    table = pd.DataFrame(rows)
    path = resolve_path("results/pid_baseline.csv")
    table.round(4).to_csv(path, index=False)
    cols = ["controller", "profile", "total_kwh", "heater_kwh", "vent_kwh", "temp_rmse_c", "temp_max_dev_c",
            "hours_in_band", "hours_below_band", "hours_above_band", "hours_rh_above_85"]
    print("\n" + table[cols].round(2).to_string(index=False))
    print(f"\nSaved {path}, logs and plots in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
