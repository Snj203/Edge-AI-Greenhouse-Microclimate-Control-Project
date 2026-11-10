"""
Train and evaluate the surrogate neural network (needs data from prepare_training_data.py).

    python scripts/train_surrogate.py     -> models/, results/surrogate/  (~3 min)

Variants
    mlp_plan  the plan's inputs: T, RH, VPD, lux, soil moisture, heater, vent, sun, outdoor T (9 -> 5)
    mlp       + soil temperature, outdoor humidity and wind (12 -> 6)

The variant with the lowest 1-hour temperature error on the validation days is saved as
models/greenhouse_nn_model.pth; the test days are used only for the final numbers.
"""

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # figures go to files only, no windows (safe next to the MQTT threads)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.emulator.simulation import create_emulator, weather_at
from src.emulator.weather import build_weather
from src.models.training import (Samples, evaluate, latency_ms, load_splits, rollout_predictions, train,
                                  variants)
from src.utils.config import ensure_dir, load_config, resolve_path

LABELS = {"temperature": ("Temperature", "°C"), "humidity": ("Humidity", "%RH"), "soil_moisture": ("Soil moisture", "%")}


# Training curves of all variants
def plot_training(logs: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, log in logs.items():
        axes[0].plot(log["epoch"], log["train_loss"], ls="--", lw=1, label=f"{name} train")
        axes[0].plot(log["epoch"], log["val_loss"], label=f"{name} val")
        axes[1].plot(log["epoch"], log["val_temp_rmse"], label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE (standardised change)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation temperature RMSE, 5 min (°C)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Surrogate training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Error of chained predictions vs horizon, for each variant and the "nothing changes" guess
def plot_rollout_error(metrics: dict, n_steps: int, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    minutes = np.arange(1, n_steps + 1) * 5
    for ax, name in zip(axes, LABELS):
        persist = None
        for variant, m in metrics.items():
            r = m["test"]["emulator"]["rollout"][name]
            ax.plot(minutes, r["rmse_by_horizon"], marker="o", ms=3, label=variant)
            persist = r["persistence_by_horizon"]
        ax.plot(minutes, persist, color="gray", ls="--", label="no-change baseline")
        label, unit = LABELS[name]
        ax.set_title(label)
        ax.set_xlabel("prediction horizon (min)")
        ax.set_ylabel(f"RMSE ({unit})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Error of chained predictions on emulator test days", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# One emulator test day and the real test day: actual temperature vs predicted 5 min and 1 h ahead
@torch.no_grad()
def plot_timeseries(model, data, spec, n_steps, path: Path):
    s = Samples(data["test"], spec)
    df = s.df
    emu_days = df[(df["source"] == "emulator") & (df["policy"] == "thermostat")]["episode"].unique()
    picks = [(emu_days[0], "emulator"), (df[df["source"] == "real"]["episode"].iloc[-1], "real")]
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 4 * len(picks)))
    for ax, (episode, src) in zip(axes, picks):
        mask = (df["episode"] == episode).to_numpy()
        steps = df.loc[mask, "step"].to_numpy()
        first = steps.min()  # real-data steps count from the start of the week
        ax.plot((steps - first) * 5 / 60, df.loc[mask, "temperature"], color="black", lw=1.2, label="actual")
        for horizon, color in [(1, "#2b6cb0"), (n_steps, "#c53030")]:
            r = rollout_predictions(model, s, horizon, mask, stride=1)
            if r is None:
                continue
            pred, _, _, starts = r
            hours = (df["step"].to_numpy()[starts] + horizon - first) * 5 / 60
            ax.plot(hours, pred[:, -1, 0], color=color, lw=0.9, label=f"predicted {horizon * 5} min ahead")
        profile = df.loc[mask, "profile"].iloc[0]
        policy = df.loc[mask, "policy"].iloc[0]
        ax.set_title(f"{src} test day — {profile}, control: {policy}", fontsize=10)
        ax.set_xlabel("hours since start of day")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Surrogate temperature predictions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Does the network know what the heater and vents do? From the same emulator state, hold four
# actions for 1 hour and compare the emulator with the network's chained prediction.
@torch.no_grad()
def actuator_check(model, cfg: dict, n_steps: int, path: Path) -> dict:
    cases = [("winter_cloudy", 3, "winter night 03:00"), ("spring_variable", 6, "spring morning 06:00"),
             ("spring_variable", 13, "spring noon 13:00"), ("summer_clear", 14, "summer afternoon 14:00")]
    actions = {"off": (0, 0), "heater 100 %": (100, 0), "vent 100 %": (0, 100), "heater 50 % + vent 50 %": (50, 50)}
    colors = dict(zip(actions, ["#4a5568", "#c53030", "#2b6cb0", "#805ad5"]))
    weather_key = {"solar_radiation": "solar_radiation", "outdoor_temp": "outdoor_temp",
                   "outdoor_humidity": "outdoor_rh", "wind_speed": "wind_speed"}
    errors = {}
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.2))
    minutes = np.arange(0, n_steps + 1) * 5
    for ax, (profile, hour, title) in zip(axes, cases):
        weather = build_weather(cfg, profile, seed=123)
        emu = create_emulator(weather, cfg)
        k0 = hour * 12
        for k in range(k0):
            emu.step(weather_at(weather, k), 0.0, 0.0)
        case_errors = []
        for name, (h, v) in actions.items():
            e = copy.deepcopy(emu)
            obs0 = e.observe(weather_at(weather, k0))
            truth = [obs0["temperature"]]
            for k in range(k0, k0 + n_steps):
                e.step(weather_at(weather, k), h, v)
                truth.append(e.observe(weather_at(weather, k + 1))["temperature"])
            state0 = torch.tensor([[obs0[s] for s in model.state_names]], dtype=torch.float32)
            w = [[weather_at(weather, k)[weather_key[n]] for n in model.weather_names] for k in range(k0, k0 + n_steps)]
            controls = torch.tensor([[[h, v]] * n_steps], dtype=torch.float32)
            pred = model.rollout(state0, controls, torch.tensor([w], dtype=torch.float32))[0, :, 0].numpy()
            pred = np.concatenate([[obs0["temperature"]], pred])
            ax.plot(minutes, truth, color=colors[name], lw=1.8, label=f"{name} — emulator")
            ax.plot(minutes, pred, color=colors[name], lw=1.4, ls="--")
            case_errors.append(abs(pred[-1] - truth[-1]))
            errors[f"{title} | {name}"] = {"emulator_change": round(truth[-1] - truth[0], 3),
                                           "surrogate_change": round(float(pred[-1] - pred[0]), 3),
                                           "error_after_1h": round(float(pred[-1] - truth[-1]), 3)}
        ax.set_title(f"{title} (max error {max(case_errors):.2f} °C)", fontsize=10)
        ax.set_xlabel("minutes")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Actuator effect: emulator (solid) vs surrogate (dashed), 1 hour of constant actions",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return errors


# Main numbers of each variant on the test days, plus the "nothing changes" guess
def comparison_table(metrics: dict, n_steps: int) -> pd.DataFrame:
    rows = []
    for variant, m in metrics.items():
        t = m["test"]
        rows.append({
            "variant": variant,
            "inputs": m["n_inputs"], "parameters": m["parameters"], "epochs": m["epochs"],
            "train_s": round(m["train_seconds"]),
            "T_1step_emu": round(t["emulator"]["one_step"]["temperature"]["rmse"], 3),
            "T_1step_real": round(t["real"]["one_step"]["temperature"]["rmse"], 3),
            f"T_{n_steps}step_emu": t["emulator"]["rollout"]["temperature"]["rmse_by_horizon"][-1],
            f"T_{n_steps}step_real": t["real"]["rollout"]["temperature"]["rmse_by_horizon"][-1],
            f"RH_{n_steps}step_emu": t["emulator"]["rollout"]["humidity"]["rmse_by_horizon"][-1],
            f"val_T_{n_steps}step_emu": m["val"]["emulator"]["rollout"]["temperature"]["rmse_by_horizon"][-1],
            "rollout_ms": m["latency"][f"rollout_{n_steps}_steps_ms"],
        })
    base = next(iter(metrics.values()))["test"]
    rows.append({
        "variant": "no-change baseline",
        "T_1step_emu": round(base["emulator"]["one_step"]["temperature"]["persistence_rmse"], 3),
        "T_1step_real": round(base["real"]["one_step"]["temperature"]["persistence_rmse"], 3),
        f"T_{n_steps}step_emu": base["emulator"]["rollout"]["temperature"]["persistence_by_horizon"][-1],
        f"T_{n_steps}step_real": base["real"]["rollout"]["temperature"]["persistence_by_horizon"][-1],
        f"RH_{n_steps}step_emu": base["emulator"]["rollout"]["humidity"]["persistence_by_horizon"][-1],
    })
    return pd.DataFrame(rows)


# Train both variants, pick the best, save the models, metrics and figures
def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=cfg["nn"]["epochs"], help="maximum number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the training")
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))  # results depend slightly on the thread count
    n_steps = cfg["nn"]["rollout_steps"]
    data = load_splits(cfg)
    out = ensure_dir("results/surrogate")
    models_dir = ensure_dir("models")
    print("Samples: " + ", ".join(f"{k} {int(v['valid'].sum())}" for k, v in data.items()))

    metrics, logs, trained = {}, {}, {}
    for name, spec in variants(cfg).items():
        n_in = len(spec["state"]) + len(spec["controls"]) + len(spec["weather"])
        print(f"\n== {name}: {n_in} inputs -> {len(spec['state'])} outputs")
        model, log, seconds = train(spec, data, cfg, args.epochs, args.seed)
        m = evaluate(model, data, spec, n_steps)
        m.update({"n_inputs": n_in, "parameters": model.n_parameters(), "epochs": len(log),
                  "best_epoch": int(log["val_loss"].idxmin() + 1), "train_seconds": seconds,
                  "latency": latency_ms(model, n_steps), "spec": spec})
        metrics[name], logs[name], trained[name] = m, log, model
        log.to_csv(out / f"training_log_{name}.csv", index=False)
        model.save(models_dir / f"surrogate_{name}.pth", {"variant": name, "trained": time.strftime("%Y-%m-%d %H:%M")})
        te = m["test"]["emulator"]
        print(f"   test temperature RMSE: 5 min {te['one_step']['temperature']['rmse']:.3f} °C "
              f"(no-change {te['one_step']['temperature']['persistence_rmse']:.3f}), "
              f"1 h {te['rollout']['temperature']['rmse_by_horizon'][-1]:.3f} °C "
              f"(no-change {te['rollout']['temperature']['persistence_by_horizon'][-1]:.3f})")

    best = min(metrics, key=lambda k: metrics[k]["val"]["emulator"]["rollout"]["temperature"]["rmse_by_horizon"][-1])
    model_path = resolve_path(cfg["nn"]["model_path"])
    shutil.copyfile(models_dir / f"surrogate_{best}.pth", model_path)
    threshold = cfg["nn"]["rmse_threshold"]
    t_val = metrics[best]["val"]["emulator"]["one_step"]["temperature"]["rmse"]
    t_test = metrics[best]["test"]["emulator"]["one_step"]["temperature"]["rmse"]
    summary = {"best_variant": best, "threshold_c": threshold,
               "val_temperature_rmse_1step": t_val, "test_temperature_rmse_1step": t_test,
               "passes_threshold": bool(t_test < threshold and t_val < threshold)}

    table = comparison_table(metrics, n_steps)
    table.to_csv(out / "comparison.csv", index=False)
    summary["actuator_check"] = actuator_check(trained[best], cfg, n_steps, out / "actuator_check.png")
    summary["actuator_check_max_error_c"] = max(abs(v["error_after_1h"]) for v in summary["actuator_check"].values())
    (out / "metrics.json").write_text(json.dumps({"summary": summary, "variants": metrics}, indent=2))

    plot_training(logs, out / "training_curves.png")
    plot_rollout_error(metrics, n_steps, out / "rollout_error.png")
    plot_timeseries(trained[best], data, metrics[best]["spec"], n_steps, out / "timeseries_example.png")

    print("\n" + table.to_string(index=False))
    print(f"\nBest variant (validation, 1-hour temperature error): {best} -> {model_path}")
    print(f"Temperature RMSE 5 min ahead: validation {t_val:.3f} °C, test {t_test:.3f} °C, "
          f"threshold {threshold} °C -> {'PASS' if summary['passes_threshold'] else 'FAIL'}")
    print(f"Actuator check (1 h of constant heater/vent actions): max error {summary['actuator_check_max_error_c']:.2f} °C")
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
