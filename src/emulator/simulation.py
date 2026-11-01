"""Set up the emulator for a weather profile and run it without MQTT."""

from typing import Callable

import pandas as pd

from src.emulator.greenhouse_emulator import GreenhouseEmulator, PhysicsParams
from src.emulator.weather import WEATHER_COLUMNS
from src.utils.config import load_config

# policy(readings, step) -> (heater %, ventilation %)
Policy = Callable[[dict, int], tuple[float, float]]


# Weather of step k as a dict
def weather_at(weather: pd.DataFrame, k: int) -> dict:
    row = weather.iloc[k]
    return {c: float(row[c]) for c in WEATHER_COLUMNS}


# Reference thermostat for the warm start: heats in proportion below `target`, vents in proportion
# above `vent_above` (gain in % per °C). It is not one of the compared controllers.
def hold_target_policy(target: float, vent_above: float, gain: float = 100.0) -> Policy:
    # heater and vent levels for the current temperature
    def policy(obs: dict, k: int) -> tuple[float, float]:
        t = obs["temperature"]
        return min(max(gain * (target - t), 0.0), 100.0), min(max(gain * (t - vent_above), 0.0), 100.0)

    return policy


# Emulator ready to start at the first weather row.
# Floor and soil are warmed up first: the first day is replayed `spin_up_days` times.
#   spin_up_policy None   actuators off (the greenhouse without control). A dataset day then starts
#                         from the indoor values measured at that moment.
#   spin_up_policy given  the greenhouse is operated by this rule during the spin-up (warm start
#                         for runs with a controller); the state reached is kept.
def create_emulator(weather: pd.DataFrame, cfg: dict | None = None, spin_up_days: int = 2,
                    params: PhysicsParams | None = None, spin_up_policy: Policy | None = None) -> GreenhouseEmulator:
    cfg = cfg or load_config()
    emu_cfg = cfg["emulator"]
    params = params or PhysicsParams.from_config(cfg)
    if weather.attrs.get("ground_temp") is not None:
        params = params.replace(ground_temp=float(weather.attrs["ground_temp"]))

    emu = GreenhouseEmulator(params, dt_minutes=emu_cfg["timestep_minutes"],
                             max_substep_s=emu_cfg["max_substep_seconds"], floor_area_m2=emu_cfg["floor_area_m2"])
    init = emu_cfg["initial_state"]
    start = weather["time"].iloc[0].to_pydatetime()
    emu.reset(temperature=float(weather["outdoor_temp"].iloc[0]) + init["temp_offset"],
              humidity=init["humidity"], soil_moisture=init["soil_moisture"], time=start)

    steps_per_day = min(len(weather), round(24 * 60 / emu_cfg["timestep_minutes"]))
    for _ in range(spin_up_days):
        for k in range(steps_per_day):
            w = weather_at(weather, k)
            heater, vent = spin_up_policy(emu.observe(w), k) if spin_up_policy else (0.0, 0.0)
            emu.step(w, heater, vent)

    measured = weather.attrs.get("initial_state")
    if measured and spin_up_policy is None:
        emu.reset(measured["temperature"], measured["humidity"], measured["soil_temperature"],
                  measured["soil_moisture"], start, floor_temperature=emu.Tf)
    else:
        emu.restart(start)
    return emu


# Simulate the whole profile without MQTT; one row per step (readings and actuators)
def run(emu: GreenhouseEmulator, weather: pd.DataFrame, policy: Policy | None = None) -> pd.DataFrame:
    rows = []
    for k in range(len(weather)):
        w = weather_at(weather, k)
        obs = emu.observe(w)
        heater, vent = policy(obs, k) if policy else (0.0, 0.0)
        rows.append({"time": emu.time, "step": k, **obs, "heater_power": heater, "ventilation_rate": vent,
                     "heater_energy_kwh": emu.heater_energy_kwh, "vent_energy_kwh": emu.vent_energy_kwh,
                     "irrigation_count": emu.irrigation_count})
        emu.step(w, heater, vent)
    return pd.DataFrame(rows)
