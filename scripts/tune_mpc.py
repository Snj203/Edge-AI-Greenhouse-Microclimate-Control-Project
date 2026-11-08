"""
Tune the NN-MPC's move penalty γ on the PID's tuning days.

    python scripts/tune_mpc.py                  # 63 runs in parallel processes (~10 min), table and plot
    python scripts/tune_mpc.py --write-config   # also store the chosen γ in configs/default.yaml
    python scripts/tune_mpc.py --replot         # table, plot and choice again from results/mpc/tuning_runs.csv

α and β stay at the plan's values (1 per kWh, 10 per °C²). The plan's cost has no penalty on actuator
changes, and without one the MPC switches the heater up and down every step. γ is chosen with the
rule used for the PID gains, on the same 9 tuning days: the smallest γ for which each actuator moves
on average at most 1 % per step.

γ is tuned without the interlock (no venting while heating), as in the original procedure: γ smooths
the optimiser's own plan. The interlock is applied afterwards; it closes the vents at once when heating
starts, a deliberate jump that γ cannot smooth. With it, the vents move more (1.15 %/step on average at
γ = 100 on the tuning days), and no γ meets the 1 % rule there.

    -> results/mpc/tuning_runs.csv, tuning.csv, tuning.png
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # figures go to files only, no windows (safe next to the MQTT threads)
import matplotlib.pyplot as plt
import pandas as pd

from src.controllers.controller_nn_mpc import NNMPCController
from src.controllers.experiment import TUNING_SEEDS, run_offline, tuning_weather, warm_emulator
from src.emulator.weather import SYNTHETIC_PROFILES
from src.utils.config import DEFAULT_CONFIG_PATH, ensure_dir, load_config
from src.utils.metrics import control_metrics

GAMMAS = [0, 1, 3, 10, 30, 100, 300]
MAX_MOVE = 1.0  # % per step, same limit as for the PID
N_DAYS = len(SYNTHETIC_PROFILES) * len(TUNING_SEEDS)
COLOR, INK_2 = "#2a78d6", "#52514e"

_days = None  # tuning days, built once per worker process


# One tuning run (in a worker process): NN-MPC with move penalty gamma on tuning day i -> metrics
def tuning_run(job: tuple[float, int]) -> dict:
    global _days
    gamma, i = job
    cfg = load_config()
    if _days is None:
        _days = tuning_weather(cfg)
    day = _days[i]
    mpc = NNMPCController.from_config(cfg, move_weight=gamma, interlock=False)
    log = run_offline(mpc, warm_emulator(day["weather"], day["cfg"]), day["weather"])
    m = control_metrics(log, cfg)
    return {"gamma": gamma, "day": f"tuning_{day['profile']}_s{day['seed']}", "profile": day["profile"], **m,
            "move_max_actuator": max(m["heater_travel_pct"], m["vent_travel_pct"]) / (len(log) - 1)}


# Mean over the tuning days for every γ, and the smallest γ within the movement limit
def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    cols = ["total_kwh", "temp_rmse_c", "cold_degree_hours", "warm_degree_hours", "move_max_actuator",
            "hours_heat_and_vent", "latency_mean_ms"]
    summary = runs.groupby("gamma")[cols].mean().reset_index()
    ok = summary[summary["move_max_actuator"] <= MAX_MOVE]
    if ok.empty:
        raise RuntimeError(f"no gamma in {GAMMAS} keeps the actuators below {MAX_MOVE} %/step:\n{summary}")
    chosen = float(ok["gamma"].min())
    summary["chosen"] = summary["gamma"] == chosen
    return summary, chosen


# Actuator movement, energy and compute time against γ
def plot(summary: pd.DataFrame, chosen: float, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = [g if g > 0 else 0.3 for g in GAMMAS]  # log axis: show γ = 0 at the left edge
    s = summary.set_index("gamma").loc[GAMMAS]
    titles = ["Movement of the busier actuator (%/step)", "Energy per day (kWh)",
              "Compute time per step (ms; runs in parallel, relative only)"]
    for ax, col, title in zip(axes, ["move_max_actuator", "total_kwh", "latency_mean_ms"], titles):
        ax.plot(x, s[col], marker="o", color=COLOR, label=f"chosen γ = {chosen:g}")
        ax.plot([chosen if chosen > 0 else 0.3], [s.loc[chosen, col]], marker="o", ms=11, mfc="none", mec=COLOR, mew=2)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{g:g}" for g in GAMMAS])
        ax.minorticks_off()
        ax.set_xlabel("move penalty γ")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].axhline(MAX_MOVE, color=INK_2, lw=1, label=f"limit {MAX_MOVE:g} %/step")
    axes[0].legend(fontsize=8)
    axes[2].set_yscale("log")
    fig.suptitle(f"NN-MPC move penalty, mean over {N_DAYS} tuning days (ringed: chosen)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Replace the value of mpc.move_weight in the config, keeping its comment
def write_config(gamma: float, path: Path = DEFAULT_CONFIG_PATH):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("  move_weight:"):
            text = f"  move_weight: {float(gamma)!r}"
            hash_pos = line.find("#")
            lines[i] = text.ljust(hash_pos) + line[hash_pos:] if hash_pos > len(text) else text + "\n"
            path.write_text("".join(lines), encoding="utf-8")
            print(f"Wrote move_weight {gamma:g} to {path}")
            return
    raise ValueError(f"mpc.move_weight not found in {path}")


# Run all γ × tuning days (or reuse the saved runs), summarise, plot, optionally write the config
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-config", action="store_true", help="store the chosen γ in configs/default.yaml")
    parser.add_argument("--replot", action="store_true", help="reuse results/mpc/tuning_runs.csv, no new runs")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()

    out = ensure_dir("results/mpc")
    if args.replot:
        runs = pd.read_csv(out / "tuning_runs.csv")
    else:
        jobs = [(float(g), i) for g in GAMMAS for i in range(N_DAYS)]
        print(f"{len(jobs)} runs on {args.workers} worker processes")
        t0, rows = time.time(), []
        with ProcessPoolExecutor(args.workers) as pool:
            for n, row in enumerate(pool.map(tuning_run, jobs), 1):
                rows.append(row)
                if n % 9 == 0:
                    print(f"  {n}/{len(jobs)} runs  ({time.time() - t0:.0f}s)", flush=True)
        runs = pd.DataFrame(rows)
        runs.round(4).to_csv(out / "tuning_runs.csv", index=False)

    summary, chosen = summarize(runs)
    summary.round(4).to_csv(out / "tuning.csv", index=False)
    plot(summary, chosen, out / "tuning.png")
    print("\n" + summary.round(3).to_string(index=False))
    print(f"\nSmallest gamma within {MAX_MOVE} %/step: {chosen:g}")
    print(f"Saved {out / 'tuning.csv'}, tuning_runs.csv and tuning.png")
    if args.write_config:
        write_config(chosen)


if __name__ == "__main__":
    main()
