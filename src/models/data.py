"""
Training data for the surrogate model. One row = one 5-minute transition:
the state, actuators and weather at t, and the state at t+1 (columns "next_*").

    real      the Texcoco week (heater and vents off)
    emulator  emulator days with varied weather and four control styles

Transitions with irrigation are marked valid = False: irrigation is not a model input, so the jump
in soil moisture cannot be predicted. Data is split by whole days, never by single rows: neighbouring
5-minute rows are almost identical, so a row-wise split would leak test data into training.
"""

import copy

import numpy as np
import pandas as pd

from src.emulator.greenhouse_emulator import LUX_PER_PPFD
from src.emulator.simulation import create_emulator, run
from src.emulator.weather import SYNTHETIC_PROFILES, build_weather
from src.utils.config import load_config
from src.utils.dataset import greenhouse_data, smooth_outdoor

STATE_PLAN = ["temperature", "humidity", "vpd", "light_lux", "soil_moisture"]  # sensors named in the plan
STATE_EXT = STATE_PLAN + ["soil_temperature"]
CONTROLS = ["heater_power", "ventilation_rate"]
WEATHER_PLAN = ["solar_radiation", "outdoor_temp"]
WEATHER_EXT = WEATHER_PLAN + ["outdoor_humidity", "wind_speed"]
META = ["episode", "step", "source", "profile", "policy", "time", "valid", "split"]

# share of emulator days per control style
POLICY_MIX = {"random": 0.40, "thermostat": 0.35, "sweep": 0.15, "off": 0.10}


# Control rule for one emulator day: policy(readings, step) -> (heater %, vent %)
def make_policy(kind: str, rng: np.random.Generator, n_steps: int):
    if kind == "off":
        return lambda obs, k: (0.0, 0.0)

    if kind == "random":  # random levels, each held 15 min .. 3 h
        heater, vent = np.zeros(n_steps), np.zeros(n_steps)
        k = 0
        while k < n_steps:
            hold = int(rng.integers(3, 37))
            heater[k:k + hold] = 0.0 if rng.random() < 0.35 else rng.uniform(0, 100)
            vent[k:k + hold] = 0.0 if rng.random() < 0.35 else rng.uniform(0, 100)
            k += hold
        return lambda obs, k: (float(heater[k]), float(vent[k]))

    if kind == "sweep":  # fixed levels in four 6-hour blocks
        levels = [0.0, 25.0, 50.0, 75.0, 100.0]
        block = n_steps // 4
        h = rng.choice(levels, 4)
        v = rng.choice(levels, 4)
        return lambda obs, k: (float(h[min(k // block, 3)]), float(v[min(k // block, 3)]))

    if kind == "thermostat":  # like a real controller, around a random set-point, with some noise
        target = rng.uniform(17, 27)
        band = rng.uniform(0.5, 3.0)
        gain = rng.uniform(10, 80)
        noise = rng.uniform(0, 10)

        # heat in proportion below the set-point, vent in proportion above set-point + band
        def policy(obs, k):
            t = obs["temperature"]
            heater = gain * (target - t) + rng.normal(0, noise)
            vent = gain * (t - target - band) + rng.normal(0, noise)
            return float(np.clip(heater, 0, 100)), float(np.clip(vent, 0, 100))

        return policy

    raise ValueError(f"unknown policy kind {kind}")


# Copy of the config with the synthetic profile's weather randomly changed (more variety)
def jittered_cfg(cfg: dict, profile: str, rng: np.random.Generator) -> dict:
    out = copy.deepcopy(cfg)
    p = out["weather_profiles"][profile]
    p["temp_mean"] += rng.uniform(-3, 3)
    p["temp_amplitude"] *= rng.uniform(0.8, 1.2)
    p["rh_mean"] = float(np.clip(p["rh_mean"] + rng.uniform(-8, 8), 20, 95))
    p["solar_peak"] *= rng.uniform(0.85, 1.15)
    p["cloud_mean"] = float(np.clip(p["cloud_mean"] + rng.uniform(-0.1, 0.1), 0, 1))
    p["wind_mean"] *= rng.uniform(0.6, 1.4)
    p["ground_temp"] += rng.uniform(-2, 2)
    return out


# Run log -> transitions: add next_* columns; a step with irrigation is not valid
def _transitions(log: pd.DataFrame) -> pd.DataFrame:
    df = log.copy()
    for c in STATE_EXT:
        df[f"next_{c}"] = df[c].shift(-1)
    df["valid"] = df["irrigation_count"].shift(-1) == df["irrigation_count"]
    return df.iloc[:-1].reset_index(drop=True)


# Simulated days: each with its own weather and control style
def generate_emulator_data(cfg: dict | None = None, episodes_per_profile: int = 70,
                           dataset_episodes_per_day: int = 10, seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    dataset_days = [f"2025-09-{d}" for d in range(12, 18)]
    jobs = [(p, None) for p in SYNTHETIC_PROFILES for _ in range(episodes_per_profile)]
    jobs += [("dataset", day) for day in dataset_days for _ in range(dataset_episodes_per_day)]

    frames = []
    for i, (profile, day) in enumerate(jobs):
        if profile == "dataset":
            ep_cfg = copy.deepcopy(cfg)
            ep_cfg["weather_profiles"]["dataset"]["day"] = day
        else:
            ep_cfg = jittered_cfg(cfg, profile, rng)
        weather = build_weather(ep_cfg, profile, hours=24, seed=int(rng.integers(1_000_000)))
        kind = rng.choice(list(POLICY_MIX), p=list(POLICY_MIX.values()))
        policy = make_policy(kind, rng, len(weather))
        log = run(create_emulator(weather, ep_cfg, spin_up_days=1), weather, policy)
        tr = _transitions(log)
        tr["episode"] = f"emu{i:04d}"
        tr["source"] = "emulator"
        tr["profile"] = profile
        tr["policy"] = kind
        frames.append(tr)
        if verbose and (i + 1) % 50 == 0:
            print(f"  emulator days: {i + 1}/{len(jobs)}")
    return pd.concat(frames, ignore_index=True)


# Transitions of the real Texcoco week (heater and vents off), one episode per calendar day
def real_data(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    data = greenhouse_data(cfg)
    w = smooth_outdoor(data)
    df = pd.DataFrame({
        "time": data["time"],
        "temperature": data["temp_aire_media"],
        "humidity": data["hum_aire_media"],
        "vpd": data["vpd_media"],
        "light_lux": data["ppfd"] * LUX_PER_PPFD,
        "soil_moisture": data["hum_suelo"],
        "soil_temperature": data["temp_suelo"],
        "heater_power": 0.0,
        "ventilation_rate": 0.0,
        "solar_radiation": w["solar_radiation"],
        "outdoor_temp": w["outdoor_temp"],
        "outdoor_humidity": w["outdoor_rh"],
        "wind_speed": w["wind_speed"],
    })
    # step number on a regular 5-min grid, so a gap in the data breaks the sequence
    df["step"] = ((df["time"] - df["time"].iloc[0]).dt.total_seconds() / 300).round().astype(int)
    for c in STATE_EXT:
        df[f"next_{c}"] = df[c].shift(-1)
    consecutive = df["step"].shift(-1) == df["step"] + 1
    sm_ok = data["soil_moisture_valid"] & data["soil_moisture_valid"].shift(-1, fill_value=False)
    no_irrigation = ~data["irrigation_event"].shift(-1, fill_value=False)
    df["valid"] = consecutive & sm_ok & no_irrigation
    df["episode"] = "real_" + df["time"].dt.strftime("%m%d")
    df["source"] = "real"
    df["profile"] = "texcoco"
    df["policy"] = "off"
    return df.iloc[:-1].reset_index(drop=True)


# Put every day into train/val/test: emulator days shuffled within each weather profile,
# real days in time order (first days train, last day test)
def split_episodes(df: pd.DataFrame, fractions=(0.7, 0.15, 0.15), seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=df.index)
    for (source, profile), group in df.groupby(["source", "profile"]):
        episodes = list(dict.fromkeys(group["episode"]))
        if source != "real":
            rng.shuffle(episodes)
        n = len(episodes)
        n_train = max(1, round(fractions[0] * n))
        n_val = max(1, round(fractions[1] * n))
        labels = {e: ("train" if j < n_train else "val" if j < n_train + n_val else "test")
                  for j, e in enumerate(episodes)}
        split[group.index] = group["episode"].map(labels)
    return split


# Real + emulator transitions with a train/val/test label
def build_training_data(cfg: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    gen = cfg["nn"]["data"]
    emu = generate_emulator_data(cfg, gen["episodes_per_profile"], gen["dataset_episodes_per_day"],
                                 gen["seed"], verbose)
    df = pd.concat([real_data(cfg), emu], ignore_index=True)
    df["split"] = split_episodes(df, cfg["nn"]["train_val_test_split"], gen["seed"])
    return df[META + STATE_EXT + CONTROLS + WEATHER_EXT + [f"next_{c}" for c in STATE_EXT]]
