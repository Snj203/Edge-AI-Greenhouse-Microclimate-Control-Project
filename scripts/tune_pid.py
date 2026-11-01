"""
Tune the PID gains by grid search on separate tuning days.

    python scripts/tune_pid.py                  # search, print, save table and plot (~1.5 min)
    python scripts/tune_pid.py --write-config   # also store the chosen gains in configs/default.yaml

Tuning days: the 3 synthetic weather profiles × 3 seeds with randomly changed weather, warm-started.
They are not the days the baseline is evaluated on.

    stage 1  heater loop, heater only:       grid over Kp, Ki, Kd
    stage 2  vent loop on top of it:         grid over Kp, Ki (heater gains from stage 1)

Criterion: the lowest mean temperature RMSE over the tuning days, among gains where each actuator
moves on average at most 1 % per step. Above that the output starts to jump back and forth
(at Kp 400 the heater jumps by ~25 % every step), which a real heater or vent motor would not tolerate.

    -> results/pid/tuning.csv, results/pid/tuning.png
"""

import argparse
import copy
import itertools
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.controllers.controller_pid import PIDController
from src.controllers.experiment import TUNING_SEEDS, run_offline, tuning_weather, warm_emulator
from src.emulator.weather import SYNTHETIC_PROFILES
from src.utils.config import DEFAULT_CONFIG_PATH, ensure_dir, load_config
from src.utils.metrics import control_metrics

MAX_MOVE = 1.0  # % per step, allowed mean actuator movement
PLAN_GAINS = {"kp": 2.0, "ki": 0.1, "kd": 0.5}
HEATER_GRID = {"kp": [10, 20, 40, 80, 120, 160, 240, 400], "ki": [0.1, 0.25, 0.5, 1, 2, 4], "kd": [0, 100]}
VENT_GRID = {"vent_kp": [10, 30, 100, 200], "vent_ki": [0, 0.5, 2, 4]}

INK, INK_2 = "#0b0b0b", "#52514e"
TUNED, PLAN = "#2a78d6", "#eb6834"


# The 9 tuning days, each with its warm-started emulator
def tuning_days(cfg: dict) -> list[dict]:
    return [day | {"emu": warm_emulator(day["weather"], day["cfg"])} for day in tuning_weather(cfg)]


# Run one set of gains on all tuning days; mean RMSE (also per profile), actuator movement, energy
def evaluate(cfg: dict, days: list[dict], mode: str, gains: dict, keep_logs: bool = False) -> dict:
    rows, logs = [], []
    for day in days:
        ctrl = PIDController.from_config(cfg, mode, **gains)
        log = run_offline(ctrl, copy.deepcopy(day["emu"]), day["weather"])
        m = control_metrics(log, cfg)
        # movement of the busier actuator, in % per step (each actuator is limited on its own)
        rows.append({"profile": day["profile"], "rmse": m["temp_rmse_c"], "kwh": m["total_kwh"],
                     "move": max(m["heater_travel_pct"], m["vent_travel_pct"]) / (len(log) - 1)})
        if keep_logs:
            logs.append(log)
    df = pd.DataFrame(rows)
    by_profile = df.groupby("profile")["rmse"].mean()
    out = {**gains, "rmse_mean": df["rmse"].mean(), **{f"rmse_{p}": by_profile[p] for p in SYNTHETIC_PROFILES},
           "move_mean": df["move"].mean(), "move_max": df["move"].max(), "kwh_mean": df["kwh"].mean()}
    if keep_logs:
        out["logs"] = logs
    return out


# Try every combination of the grid; mark the allowed ones and the chosen one
def search(cfg, days, mode, grid: dict, fixed: dict, stage: str) -> pd.DataFrame:
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    rows, t0 = [], time.time()
    for i, gains in enumerate(combos, 1):
        rows.append({"stage": stage, **fixed, **evaluate(cfg, days, mode, {**fixed, **gains})})
        if i % 16 == 0 or i == len(combos):
            print(f"  {stage}: {i}/{len(combos)}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["eligible"] = df["move_mean"] <= MAX_MOVE
    if not df["eligible"].any():
        raise RuntimeError(f"{stage}: no gains keep the actuators below {MAX_MOVE} %/step — widen the grid")
    df["chosen"] = False
    df.loc[df[df["eligible"]]["rmse_mean"].idxmin(), "chosen"] = True
    return df


# Replace kp/ki/kd in the `pid` block of the config (indent 2: heater loop, 4: vent loop), keeping comments
def write_config(heater: dict, vent: dict, path: Path = DEFAULT_CONFIG_PATH):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    inside, done = False, set()
    for i, line in enumerate(lines):
        if line.startswith("pid:"):
            inside = True
            continue
        if inside and line.strip() and not line.startswith(" "):
            break
        m = re.match(r"^( +)(kp|ki|kd):", line)
        if inside and m:
            loop = "heater" if len(m.group(1)) == 2 else "vent"
            value = (heater if loop == "heater" else vent)[m.group(2)]
            text = f"{m.group(1)}{m.group(2)}: {float(value)!r}"
            hash_pos = line.find("#")
            lines[i] = (text.ljust(hash_pos) + line[hash_pos:] if hash_pos > len(text) else text + "\n")
            done.add((loop, m.group(2)))
    if len(done) != 6:
        raise ValueError(f"pid gains not found in {path}: found {sorted(done)}")
    path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote tuned gains to {path}")


# Left: heater-loop RMSE for every Kp × Ki. Right: a winter tuning day with the plan's and the tuned gains
def plot(cfg, heater_grid: pd.DataFrame, best: dict, plan: dict, path: Path):
    target = cfg["target"]["temperature"]
    band = cfg["experiment"]["temperature_band"]
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], hspace=0.3, wspace=0.22)

    ax = fig.add_subplot(gs[:, 0])
    sub = heater_grid[heater_grid["kd"] == best["kd"]]
    rmse = sub.pivot(index="kp", columns="ki", values="rmse_mean")
    ok = sub.pivot(index="kp", columns="ki", values="eligible")
    im = ax.imshow(rmse.to_numpy(), cmap="Blues", aspect="auto", origin="lower")
    lo, hi = np.nanmin(rmse.to_numpy()), np.nanmax(rmse.to_numpy())
    for (r, c), v in np.ndenumerate(rmse.to_numpy()):
        dark = (v - lo) / (hi - lo + 1e-9) > 0.55
        label = f"{v:.2f}" if ok.to_numpy()[r, c] else f"{v:.2f}\nexcl."
        ax.text(c, r, label, ha="center", va="center", fontsize=8, color="white" if dark else INK)
    r_best = list(rmse.index).index(best["kp"])
    c_best = list(rmse.columns).index(best["ki"])
    ax.add_patch(plt.Rectangle((c_best - 0.5, r_best - 0.5), 1, 1, fill=False, ec=INK, lw=2.5))
    ax.set_xticks(range(len(rmse.columns)), [f"{v:g}" for v in rmse.columns])
    ax.set_yticks(range(len(rmse.index)), [f"{v:g}" for v in rmse.index])
    ax.set_xlabel("Ki  (%/(°C·min))")
    ax.set_ylabel("Kp  (%/°C)")
    ax.set_title(f"Heater loop: mean temperature RMSE (°C) on tuning days, Kd = {best['kd']:g}\n"
                 f"outlined = chosen; 'excl.' = excluded, heater moves > {MAX_MOVE:g} %/step", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="°C")

    ax_t = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1], sharex=ax_t)
    for res, name, color in [(best, "tuned", TUNED), (plan, "plan gains (2.0 / 0.1 / 0.5)", PLAN)]:
        log = res["logs"][SYNTHETIC_PROFILES.index("winter_cloudy") * len(TUNING_SEEDS)]
        hours = np.arange(len(log)) * cfg["emulator"]["timestep_minutes"] / 60
        ax_t.plot(hours, log["temperature"], color=color, lw=1.6, label=name)
        ax_h.plot(hours, log["heater_power"], color=color, lw=1.6, label=name)
    ax_t.axhspan(target - band, target + band, color=INK_2, alpha=0.08, lw=0)
    ax_t.axhline(target, color=INK_2, lw=1, label=f"target {target:g} °C")
    ax_t.plot(hours, log["outdoor_temp"], color=INK_2, lw=1, ls="--", label="outdoor")
    ax_t.set_ylabel("Temperature (°C)")
    ax_t.set_title("Winter tuning day, heater only", fontsize=10)
    ax_t.legend(fontsize=8, loc="center right")
    ax_h.set_ylabel("Heater (%)")
    ax_h.set_ylim(-5, 105)
    ax_h.set_xlabel("hour of day")
    ax_h.set_xticks(range(0, 25, 3))
    for a in (ax_t, ax_h):
        a.grid(alpha=0.3)
    fig.suptitle("PID tuning by grid search", fontweight="bold")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# Plan gains for reference, then stage 1 (heater) and stage 2 (vents); save, print, optionally write config
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-config", action="store_true", help="store the chosen gains in configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config()
    out = ensure_dir("results/pid")
    t0 = time.time()
    days = tuning_days(cfg)
    print(f"{len(days)} tuning days ready ({time.time() - t0:.0f}s)")

    plan = evaluate(cfg, days, "heat", PLAN_GAINS, keep_logs=True)
    heater = search(cfg, days, "heat", HEATER_GRID, {}, "heater")
    h_best = heater[heater["chosen"]].iloc[0]
    h_gains = {k: float(h_best[k]) for k in HEATER_GRID}
    vent = search(cfg, days, "heat_vent", VENT_GRID, h_gains, "ventilation")
    v_best = vent[vent["chosen"]].iloc[0]
    v_gains = {k: float(v_best[k]) for k in VENT_GRID}

    plan_row = {"stage": "plan gains", **{k: v for k, v in plan.items() if k != "logs"}, "eligible": None,
                "chosen": False}
    table = pd.concat([pd.DataFrame([plan_row]), heater, vent], ignore_index=True)
    table.round(4).to_csv(out / "tuning.csv", index=False)

    best_logs = evaluate(cfg, days, "heat", h_gains, keep_logs=True)
    plot(cfg, heater, {**h_gains, "logs": best_logs["logs"]}, plan, out / "tuning.png")

    cols = ["rmse_mean", *[f"rmse_{p}" for p in SYNTHETIC_PROFILES], "move_mean", "kwh_mean"]
    print("\nHeater loop (heater only), mean over tuning days:")
    print(pd.DataFrame([{"gains": "plan 2.0/0.1/0.5", **{c: plan[c] for c in cols}},
                        {"gains": f"tuned {h_gains['kp']:g}/{h_gains['ki']:g}/{h_gains['kd']:g}",
                         **{c: h_best[c] for c in cols}}]).round(3).to_string(index=False))
    print(f"\nVentilation loop: Kp {v_gains['vent_kp']:g}, Ki {v_gains['vent_ki']:g} -> "
          f"mean RMSE {v_best['rmse_mean']:.3f} °C (heater only {h_best['rmse_mean']:.3f} °C)")
    print(f"Saved {out / 'tuning.csv'} and tuning.png  ({time.time() - t0:.0f}s)")

    if args.write_config:
        write_config(h_gains, {"kp": v_gains["vent_kp"], "ki": v_gains["vent_ki"], "kd": 0.0})


if __name__ == "__main__":
    main()
