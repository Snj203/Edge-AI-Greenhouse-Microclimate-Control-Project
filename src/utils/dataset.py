"""Texcoco greenhouse dataset: loading, data-quality flags and smoothed outdoor weather."""

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path

# Found in the data analysis (scripts/eda_dataset.py)
SETUP_END = pd.Timestamp("2025-09-11 19:00")  # before this the sensors were not yet in the greenhouse
SOIL_FAULT_BELOW = 10.0     # % soil moisture below this is a sensor fault
SOIL_SATURATED_FROM = 99.0  # % the sensor saturates for a few minutes after irrigation
IRRIGATION_JUMP = 15.0      # % rise between two samples = irrigation

OUTDOOR_COLUMNS = {
    "outside_openmeteo_temperature_2m": "outdoor_temp",
    "outside_openmeteo_relative_humidity_2m": "outdoor_rh",
    "outside_openmeteo_shortwave_radiation": "solar_radiation",
    "outside_openmeteo_wind_speed_10m": "wind_speed",
}


# Times -> seconds as numbers (for interpolation)
def to_seconds(times) -> np.ndarray:
    return np.asarray(times, dtype="datetime64[s]").astype("int64").astype(float)


# Load the CSV: all rows sorted by time, with data-quality flags
def load_dataset(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ds = cfg["dataset"]
    df = pd.read_csv(resolve_path(ds["path"]))
    df.attrs["raw_columns"] = list(df.columns)

    df["time"] = pd.to_datetime(df[ds["timestamp_col"]], format="%d/%m/%Y %H:%M")
    df = df.sort_values("time").reset_index(drop=True)
    df["dt_min"] = df["time"].diff().dt.total_seconds().div(60)

    df["setup_period"] = df["time"] < SETUP_END
    sm = df["hum_suelo"]
    rise = sm.diff()
    df["soil_fault"] = sm < SOIL_FAULT_BELOW
    df["soil_saturated"] = sm >= SOIL_SATURATED_FROM

    jump = (rise > IRRIGATION_JUMP) & ~df["setup_period"] & ~df["setup_period"].shift(1, fill_value=True)
    # one irrigation can rise over two samples (e.g. 43 -> 81 -> 100 %), count it once
    df["irrigation_event"] = jump & ~jump.shift(1, fill_value=False)

    settling = df["soil_saturated"].shift(1, fill_value=False) & (rise < 0)
    df["soil_moisture_valid"] = ~(df["setup_period"] | df["soil_fault"] | df["soil_saturated"] | settling
                                  | df["irrigation_event"])
    return df


# Rows recorded inside the greenhouse (setup period removed)
def greenhouse_data(cfg: dict | None = None) -> pd.DataFrame:
    df = load_dataset(cfg)
    return df[~df["setup_period"]].reset_index(drop=True)


# Outdoor weather at every row, interpolated between the hourly OpenMeteo values.
# Solar radiation is an hourly mean, so it is placed in the middle of its hour (−30 min).
def smooth_outdoor(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.groupby(df["time"].dt.floor("h"))[list(OUTDOOR_COLUMNS)].first()
    t_rows = to_seconds(df["time"])
    out = pd.DataFrame({"time": df["time"]})
    for col, name in OUTDOOR_COLUMNS.items():
        shift = -1800 if name == "solar_radiation" else 0
        out[name] = np.interp(t_rows, to_seconds(hourly.index) + shift, hourly[col].to_numpy())
    out["solar_radiation"] = out["solar_radiation"].clip(lower=0)
    return out
