"""
Metrics of a run with a controller. Input: the run log, one row per 5-min step (state at the
start of the step and the actuator settings held during it).

    energy        heater, fans and total in kWh; heater_pct_hours = ∫ heater_power dt in %·h (the plan's)
    temperature   RMSE, mean and maximum deviation from the target; min / mean / max;
                  hours inside / below / above target ± band; degree-hours below / above the band
    actuators     mean level, total movement Σ|Δu|, hours with heating and venting at the same time
    other sensors humidity and VPD vs their targets, hours above 85 %RH, mean light and soil moisture
    compute time  controller time per step: mean / 95th percentile / max (if logged)
"""

import numpy as np
import pandas as pd

HUMID_RH = 85.0  # %RH above which the risk of fungal disease rises


# All metrics of one run as a dict
def control_metrics(log: pd.DataFrame, cfg: dict) -> dict:
    target = cfg["target"]
    band = cfg["experiment"]["temperature_band"]
    emu = cfg["emulator"]
    phys = emu["physics"]
    dt_h = emu["timestep_minutes"] / 60.0
    area_kw = emu["floor_area_m2"] / 1000.0

    temp = log["temperature"].to_numpy(float)
    err = temp - target["temperature"]
    heater = log["heater_power"].to_numpy(float).clip(0, 100)
    vent = log["ventilation_rate"].to_numpy(float).clip(0, 100)
    heater_kwh = float(np.sum(heater / 100.0) * float(phys["heater_max_power"]) * area_kw * dt_h)
    vent_kwh = float(np.sum(vent / 100.0) * float(phys["vent_fan_power"]) * area_kw * dt_h)
    rh = log["humidity"].to_numpy(float)
    vpd = log["vpd"].to_numpy(float)

    m = {
        "hours": len(log) * dt_h,
        "heater_kwh": heater_kwh,
        "vent_kwh": vent_kwh,
        "total_kwh": heater_kwh + vent_kwh,
        "heater_pct_hours": float(np.sum(heater) * dt_h),
        "temp_rmse_c": float(np.sqrt(np.mean(err ** 2))),
        "temp_mae_c": float(np.mean(np.abs(err))),
        "temp_max_dev_c": float(np.max(np.abs(err))),
        "temp_mean_c": float(np.mean(temp)),
        "temp_min_c": float(np.min(temp)),
        "temp_max_c": float(np.max(temp)),
        "hours_in_band": float(np.sum(np.abs(err) <= band) * dt_h),
        "hours_below_band": float(np.sum(err < -band) * dt_h),
        "hours_above_band": float(np.sum(err > band) * dt_h),
        "cold_degree_hours": float(np.sum(np.maximum(-band - err, 0.0)) * dt_h),
        "warm_degree_hours": float(np.sum(np.maximum(err - band, 0.0)) * dt_h),
        "heater_mean_pct": float(np.mean(heater)),
        "vent_mean_pct": float(np.mean(vent)),
        "heater_travel_pct": float(np.sum(np.abs(np.diff(heater)))),
        "vent_travel_pct": float(np.sum(np.abs(np.diff(vent)))),
        "hours_heat_and_vent": float(np.sum((heater > 0.5) & (vent > 0.5)) * dt_h),
        "humidity_mean_pct": float(np.mean(rh)),
        "humidity_rmse_pct": float(np.sqrt(np.mean((rh - target["humidity"]) ** 2))),
        "hours_rh_above_85": float(np.sum(rh > HUMID_RH) * dt_h),
        "vpd_mean_kpa": float(np.mean(vpd)),
        "vpd_rmse_kpa": float(np.sqrt(np.mean((vpd - target["vpd"]) ** 2))),
        "ppfd_mean": float(log["light_ppfd"].mean()),
        "soil_moisture_mean_pct": float(log["soil_moisture"].mean()),
        "soil_moisture_min_pct": float(log["soil_moisture"].min()),
    }
    if "latency_ms" in log:
        lat = log["latency_ms"].dropna().to_numpy(float)
        if len(lat):
            m.update({"latency_mean_ms": float(np.mean(lat)), "latency_p95_ms": float(np.percentile(lat, 95)),
                      "latency_max_ms": float(np.max(lat))})
    return m
