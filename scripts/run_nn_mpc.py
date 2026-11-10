"""
NN-MPC on every weather profile: 24 hours over MQTT, compute time per step.

    python scripts/run_nn_mpc.py              # over the Mosquitto broker (lockstep), ~2 min
    python scripts/run_nn_mpc.py --offline    # direct calls, no broker needed

Runs are warm-started like the PID baseline and run one at a time, so the measured compute times
are not disturbed by other runs. The PID with heater + vents (scripts/run_pid_baseline.py) is drawn
for comparison.

    -> results/nn_mpc_runs.csv                 metrics per profile
       results/mpc/run_nn_mpc_<profile>.csv    per-step logs (incl. compute time, iterations, bias)
       results/mpc/mpc_<profile>.png           temperature, actuators, compute time and bias over the day
       results/mpc/latency.csv                 compute time per step
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # figures go to files only, no windows (safe next to the MQTT threads)
import matplotlib.pyplot as plt
import pandas as pd

from src.controllers.controller_nn_mpc import NNMPCController
from src.controllers.experiment import run_experiment
from src.emulator.weather import ALL_PROFILES
from src.utils.config import ensure_dir, load_config, resolve_path

INK_2 = "#52514e"
COLORS = {"pid_heat_vent": INK_2, "nn_mpc": "#2a78d6"}
LABELS = {"pid_heat_vent": "PID, heater + ventilation", "nn_mpc": "NN-MPC, target 22 °C"}
LATENCY_LIMIT_MS = 500


# PID and NN-MPC on one profile: temperature, heater, vents, NN-MPC compute time and model bias
def plot_profile(logs: dict[str, pd.DataFrame], cfg: dict, profile: str, path: Path):
    target = cfg["target"]["temperature"]
    band = cfg["experiment"]["temperature_band"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 0.8, 0.8, 0.8, 0.7]})
    for name, log in logs.items():
        hours = log["step"].to_numpy() * cfg["emulator"]["timestep_minutes"] / 60
        style = dict(color=COLORS[name], lw=1.0 if name.startswith("pid") else 1.6, label=LABELS[name])
        axes[0].plot(hours, log["temperature"], **style)
        axes[1].step(hours, log["heater_power"], where="post", **style)
        axes[2].step(hours, log["ventilation_rate"], where="post", **style)
        if "mpc_bias" in log:
            axes[3].plot(hours, log["latency_ms"], **style)
            axes[4].plot(hours, log["mpc_bias"], **style)
    axes[0].plot(hours, log["outdoor_temp"], color=INK_2, lw=1, ls="--", label="outdoor")
    axes[0].axhspan(target - band, target + band, color=INK_2, alpha=0.08, lw=0)
    axes[0].axhline(target, color=INK_2, lw=0.8)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(fontsize=8, ncol=3, loc="best")
    axes[1].set_ylabel("Heater (%)")
    axes[2].set_ylabel("Ventilation (%)")
    for ax in axes[1:3]:
        ax.set_ylim(-5, 105)
    axes[3].axhline(LATENCY_LIMIT_MS, color=INK_2, lw=1)
    axes[3].text(0.2, LATENCY_LIMIT_MS, f"limit {LATENCY_LIMIT_MS} ms", va="bottom", fontsize=8, color=INK_2)
    axes[3].set_ylabel("Compute time (ms)")
    axes[3].set_ylim(0, LATENCY_LIMIT_MS * 1.15)
    axes[4].axhline(0, color=INK_2, lw=0.8)
    axes[4].set_ylabel("Model bias\n(°C per step)")
    axes[-1].set_xlabel("hour of day")
    axes[-1].set_xticks(range(0, 25, 3))
    axes[-1].set_xlim(0, 24)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"NN-MPC vs PID — {profile}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Run the NN-MPC on every profile, save logs, table, plots and compute-time statistics
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="call the controller directly instead of over MQTT")
    parser.add_argument("--profiles", nargs="+", default=ALL_PROFILES, choices=ALL_PROFILES)
    args = parser.parse_args()

    cfg = load_config()
    out = ensure_dir("results/mpc")
    m = cfg["mpc"]
    print(f"NN-MPC: horizon {m['horizon_steps']} steps, alpha {m['alpha']}, beta {m['beta']}, "
          f"gamma {m['move_weight']}, SLSQP — {'offline' if args.offline else 'over MQTT, lockstep'}\n")

    rows, latencies = [], []
    for profile in args.profiles:
        logs = {}
        pid_log = resolve_path(f"results/pid/run_pid_heat_vent_{profile}.csv")
        if pid_log.exists():
            logs["pid_heat_vent"] = pd.read_csv(pid_log)
        controller = NNMPCController.from_config(cfg)
        try:
            log, met = run_experiment(controller, profile, cfg, mqtt=not args.offline)
        except ConnectionError as e:
            print(f"ERROR: {e}\n(use --offline to run without the broker)")
            return 1
        mqtt = log.attrs.get("mqtt")
        if mqtt and mqtt["timeouts"]:
            print(f"  WARNING: {mqtt['timeouts']} lockstep waits timed out — the run is not reproducible")
        log.to_csv(out / f"run_{controller.name}_{profile}.csv", index=False)
        logs[controller.name] = log
        latencies.append(log["latency_ms"])
        rows.append({"controller": controller.name, "profile": profile, **met,
                     "mpc_iterations_mean": log["mpc_iterations"].mean(),
                     "mpc_not_converged_steps": int((~log["mpc_converged"].astype(bool)).sum()),
                     "mpc_bias_mean": log["mpc_bias"].mean(),
                     **({"mqtt_commands": mqtt["commands_received"], "mqtt_timeouts": mqtt["timeouts"]}
                        if mqtt else {})})
        print(f"  {profile:16s} energy {met['total_kwh']:6.1f} kWh   T-RMSE {met['temp_rmse_c']:5.2f} °C   "
              f"in band {met['hours_in_band']:4.1f} h   compute mean {met['latency_mean_ms']:5.1f} / "
              f"max {met['latency_max_ms']:5.1f} ms   iterations {log['mpc_iterations'].mean():4.1f}"
              + (f"   MQTT {mqtt['commands_received']} commands, {mqtt['timeouts']} timeouts" if mqtt else ""))
        plot_profile(logs, cfg, profile, out / f"mpc_{profile}.png")

    table = pd.DataFrame(rows)
    table.round(4).to_csv(resolve_path("results/nn_mpc_runs.csv"), index=False)
    lat = pd.concat(latencies).dropna()
    latency = pd.DataFrame([{"steps": len(lat), "mean_ms": lat.mean(), "median_ms": lat.median(),
                             "p95_ms": lat.quantile(0.95), "max_ms": lat.max(),
                             "over_limit": int((lat > LATENCY_LIMIT_MS).sum())}])
    latency.round(3).to_csv(out / "latency.csv", index=False)
    cols = ["profile", "total_kwh", "temp_rmse_c", "temp_max_dev_c", "hours_in_band", "cold_degree_hours",
            "warm_degree_hours", "heater_travel_pct", "hours_heat_and_vent"]
    print("\n" + table[cols].round(2).to_string(index=False))
    print("\nCompute time per control step:\n" + latency.round(1).to_string(index=False))
    print(f"\nSaved results/nn_mpc_runs.csv, logs, plots and latency.csv in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
