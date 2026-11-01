"""
Runs with a controller: one controller, one weather profile, 24 hours.

Warm start: before the evaluated day its weather is replayed `experiment.spin_up_days` times while a
reference thermostat holds the target. Without it a 24-hour run would mostly measure warming up a
cold greenhouse (in winter the floor alone adds ~20 % to the day's heating). The reference thermostat
is not one of the compared controllers, so every controller starts from the identical state.

Two ways to run, with identical results:
    offline  the controller is called directly (fast: tuning, tests)
    mqtt     emulator and controller talk through the Mosquitto broker, the emulator waiting
             for the reply to every step (lockstep)
"""

import time

import numpy as np
import pandas as pd

from src.controllers.base import Controller
from src.controllers.mqtt_node import ControllerNode
from src.emulator.greenhouse_emulator import GreenhouseEmulator
from src.emulator.mqtt_node import EmulatorNode
from src.emulator.simulation import create_emulator, hold_target_policy, weather_at
from src.emulator.weather import SYNTHETIC_PROFILES, build_weather
from src.models.data import jittered_cfg
from src.utils.metrics import control_metrics

TUNING_SEEDS = (101, 102, 103)


# Emulator at the start of the weather, after `spin_up_days` with the reference thermostat holding the target
def warm_emulator(weather: pd.DataFrame, cfg: dict) -> GreenhouseEmulator:
    exp = cfg["experiment"]
    target = cfg["target"]["temperature"]
    policy = hold_target_policy(target, vent_above=target + exp["temperature_band"])
    return create_emulator(weather, cfg, spin_up_days=exp["spin_up_days"], spin_up_policy=policy)


# Days for tuning controllers, never used for evaluation: 3 synthetic profiles × 3 seeds with
# randomly changed weather. Each day: {"profile", "seed", "cfg" (changed copy), "weather"}
def tuning_weather(cfg: dict) -> list[dict]:
    rng = np.random.default_rng(2024)
    days = []
    for profile in SYNTHETIC_PROFILES:
        for seed in TUNING_SEEDS:
            day_cfg = jittered_cfg(cfg, profile, rng)
            weather = build_weather(day_cfg, profile, hours=cfg["experiment"]["hours"], seed=seed)
            days.append({"profile": profile, "seed": seed, "cfg": day_cfg, "weather": weather})
    return days


# Weather of the evaluated day and a warm-started emulator
def prepare_run(cfg: dict, profile: str, seed: int = 0, hours: float | None = None):
    weather = build_weather(cfg, profile, hours or cfg["experiment"]["hours"], seed)
    return weather, warm_emulator(weather, cfg)


# Run with direct calls, no MQTT; the controller gets the same message as over MQTT. One row per step
def run_offline(controller: Controller, emu: GreenhouseEmulator, weather: pd.DataFrame) -> pd.DataFrame:
    controller.reset()
    rows = []
    for k in range(len(weather)):
        w = weather_at(weather, k)
        obs = emu.observe(w)
        heater, vent = controller({"timestamp": emu.time.isoformat(), "step": k, **obs})
        rows.append({"time": emu.time, "step": k, **obs, "heater_power": heater, "ventilation_rate": vent,
                     "heater_energy_kwh": emu.heater_energy_kwh, "vent_energy_kwh": emu.vent_energy_kwh,
                     "latency_ms": controller.last_latency_ms})
        emu.step(w, heater, vent)
    return pd.DataFrame(rows)


# Run with the emulator and the controller as two MQTT nodes (in this process) talking through the broker.
# log.attrs["mqtt"] holds the number of commands received and of lockstep waits that timed out
def run_mqtt(controller: Controller, emu: GreenhouseEmulator, weather: pd.DataFrame, cfg: dict,
             command_timeout: float = 5.0, verbose: bool = False) -> pd.DataFrame:
    node = ControllerNode(cfg, controller, verbose=verbose).start()
    try:
        time.sleep(0.2)  # let the controller's subscriptions settle
        emulator = EmulatorNode(cfg, emu, weather, lockstep=True, command_timeout=command_timeout, verbose=verbose)
        log = emulator.run()
        node.wait(2.0)
    finally:
        node.stop()
    log["latency_ms"] = log["step"].map(node.latency_ms)
    log.attrs["mqtt"] = {"commands_received": emulator.commands_received, "timeouts": emulator.timeouts}
    return log


# Warm-started run of a controller on a weather profile; returns the log and its metrics
def run_experiment(controller: Controller, profile: str, cfg: dict, mqtt: bool = False, seed: int = 0,
                   hours: float | None = None) -> tuple[pd.DataFrame, dict]:
    weather, emu = prepare_run(cfg, profile, seed, hours)
    log = run_mqtt(controller, emu, weather, cfg) if mqtt else run_offline(controller, emu, weather)
    return log, control_metrics(log, cfg)
