"""
Weather profiles — the outdoor conditions that drive the emulator.

    dataset          a real day of the Texcoco week (OpenMeteo, interpolated)
    summer_clear     synthetic: hot, dry, sunny
    winter_cloudy    synthetic: cold, humid, overcast
    spring_variable  synthetic: mild with passing clouds

Each profile is a table with one row per emulator step: time, outdoor_temp (°C), outdoor_rh (%),
solar_radiation (W/m²), wind_speed (km/h). `df.attrs` holds the profile name, an optional
ground temperature and, for the dataset profile, the measured indoor state at the start.
"""

import math

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.dataset import greenhouse_data, smooth_outdoor, to_seconds

WEATHER_COLUMNS = ["outdoor_temp", "outdoor_rh", "solar_radiation", "wind_speed"]
SYNTHETIC_PROFILES = ["summer_clear", "winter_cloudy", "spring_variable"]
ALL_PROFILES = ["dataset", *SYNTHETIC_PROFILES]


# Weather table of a profile, one row per emulator step
def build_weather(cfg: dict | None = None, profile: str | None = None, hours: float | None = None,
                  seed: int = 0) -> pd.DataFrame:
    cfg = cfg or load_config()
    profile = profile or cfg["emulator"]["weather_profile"]
    hours = hours or cfg["emulator"]["simulation_hours"]
    dt = cfg["emulator"]["timestep_minutes"]
    if profile not in cfg["weather_profiles"]:
        raise KeyError(f"Unknown weather profile '{profile}'. Options: {list(cfg['weather_profiles'])}")
    settings = cfg["weather_profiles"][profile]
    if profile == "dataset":
        return _dataset_weather(cfg, settings, hours, dt)
    return _synthetic_weather(profile, settings, hours, dt, seed)


# A real day from the dataset, plus the measured indoor state at its start
def _dataset_weather(cfg, settings, hours, dt) -> pd.DataFrame:
    data = greenhouse_data(cfg)
    outdoor = smooth_outdoor(data)
    start = pd.Timestamp(settings["day"])
    times = pd.date_range(start, start + pd.Timedelta(hours=hours), freq=f"{dt}min", inclusive="left")
    if times[0] < data["time"].iloc[0] or times[-1] > data["time"].iloc[-1]:
        raise ValueError(f"The dataset covers {data['time'].iloc[0]} → {data['time'].iloc[-1]}; "
                         f"cannot replay {hours} h from {start}")

    df = pd.DataFrame({"time": times})
    for col in WEATHER_COLUMNS:
        df[col] = np.interp(to_seconds(times), to_seconds(outdoor["time"]), outdoor[col])

    first = data.iloc[(data["time"] - start).abs().argmin()]
    df.attrs = {
        "profile": "dataset",
        "ground_temp": None,
        "initial_state": {
            "temperature": float(first["temp_aire_media"]),
            "humidity": float(first["hum_aire_media"]),
            "soil_temperature": float(first["temp_suelo"]),
            "soil_moisture": float(first["hum_suelo"]),
        },
    }
    return df


# Slowly varying random noise (AR(1)) with standard deviation `sigma`
def _ar1(rng, n, phi, sigma):
    x = np.zeros(n)
    e = rng.normal(0, sigma * math.sqrt(1 - phi ** 2), n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# Synthetic day: daily temperature cycle, sun arc, random clouds and wind
def _synthetic_weather(profile, s, hours, dt, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed + SYNTHETIC_PROFILES.index(profile) * 1000)
    start = pd.Timestamp(s["start_date"])
    times = pd.date_range(start, start + pd.Timedelta(hours=hours), freq=f"{dt}min", inclusive="left")
    n = len(times)
    h = ((times - start) / pd.Timedelta(hours=1)).to_numpy() % 24  # hour of the day

    cloud = np.clip(s["cloud_mean"] + _ar1(rng, n, 0.97, s["cloud_variability"]), 0, 1)

    # sun: half-sine between sunrise and sunset, dimmed by clouds
    sunrise = 12 - s["day_length_hours"] / 2
    phase = np.clip((h - sunrise) / s["day_length_hours"], 0, 1)
    clear_sky = s["solar_peak"] * np.sin(np.pi * phase) ** 1.2
    solar = clear_sky * (1 - 0.75 * cloud)

    # warmest at 15:00, cooler under clouds by day; humidity is lowest in the afternoon
    daily = np.cos(2 * np.pi * (h - 15) / 24)
    temp = (s["temp_mean"] + s["temp_amplitude"] * daily
            - 2.0 * (cloud - s["cloud_mean"]) * (clear_sky > 0) + _ar1(rng, n, 0.98, 0.5))
    rh = np.clip(s["rh_mean"] - s["rh_amplitude"] * daily + 10 * (cloud - s["cloud_mean"])
                 + _ar1(rng, n, 0.95, 2.0), 15, 100)
    wind = np.clip(s["wind_mean"] * (1 + 0.3 * np.cos(2 * np.pi * (h - 14) / 24))
                   + _ar1(rng, n, 0.9, 0.25 * s["wind_mean"]), 0, None)

    df = pd.DataFrame({"time": times, "outdoor_temp": temp, "outdoor_rh": rh,
                       "solar_radiation": solar, "wind_speed": wind})
    df.attrs = {"profile": profile, "ground_temp": s.get("ground_temp"), "initial_state": None}
    return df
