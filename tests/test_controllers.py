import numpy as np
import pandas as pd
import pytest

from src.controllers.controller_pid import PID, PIDController
from src.controllers.experiment import prepare_run, run_offline, warm_emulator
from src.emulator.simulation import create_emulator
from src.emulator.weather import build_weather
from src.utils.config import load_config
from src.utils.metrics import control_metrics


# Config shared by the tests in this file
@pytest.fixture(scope="module")
def cfg():
    return load_config()


# P, I and D follow the plan's formula (worked out by hand)
def test_pid_terms_follow_the_plan_formula():
    pid = PID(kp=2.0, ki=0.1, kd=0.5, setpoint=22.0, dt=5.0)
    assert pid.update(21.0) == pytest.approx(2.0 + 0.5)            # e = 1: P 2, I 0.5, no D on the first step
    assert pid.update(21.5) == pytest.approx(1.0 + 0.75 - 0.05)    # e = 0.5: P 1, I 0.75, D 0.5·(0.5 − 1)/5


# Output stays within 0–100 %; the vent loop reacts to "too warm" instead of "too cold"
def test_output_is_clamped_and_cooling_loop_reverses():
    heat = PID(10, 0, 0, setpoint=22, dt=5)
    assert heat.update(30) == 0.0 and heat.update(0) == 100.0
    vent = PID(10, 0, 0, setpoint=23, dt=5, cooling=True)
    assert vent.update(25) == pytest.approx(20.0)
    assert vent.update(20) == 0.0


# The integral never goes beyond its limit
def test_integral_is_bounded():
    pid = PID(kp=0, ki=1, kd=0, setpoint=22, dt=5, integral_limit=30, out_max=1000)
    for _ in range(20):
        u = pid.update(21.0)
    assert pid.integral == 30 and u == 30


# While the heater is already at 100 %, the integral does not pile up (anti-windup)
def test_no_windup_while_saturated():
    pid = PID(kp=10, ki=1, kd=0, setpoint=22, dt=5)
    for _ in range(100):                        # far too cold for hours: heater at 100 %
        assert pid.update(10.0) == 100.0
    assert pid.integral == 0.0                  # the integral did not grow while it could not help
    assert pid.update(23.0) == 0.0              # so the heater switches off as soon as it is too warm


# The two variants from the config; gains can be overridden; an unknown variant is an error
def test_modes(cfg):
    target = cfg["target"]["temperature"]
    heat = PIDController.from_config(cfg, "heat")
    both = PIDController.from_config(cfg, "heat_vent")
    assert heat.name == "pid_heat" and both.name == "pid_heat_vent"
    assert heat({"temperature": target + 5}) == (0.0, 0.0)
    heater, vent = both({"temperature": target + 5})
    assert heater == 0.0 and vent > 0
    both.reset()
    heater, vent = both({"temperature": target - 3})
    assert heater > 0 and vent == 0.0
    assert PIDController.from_config(cfg, "heat", kp=7.0).heater.kp == 7.0
    with pytest.raises(ValueError):
        PIDController.from_config(cfg, "cool")


# After hot hours, a cold reading turns the heater on, keeps the vents closed and resets the vent loop
def test_no_venting_while_heating(cfg):
    target = cfg["target"]["temperature"]
    ctrl = PIDController.from_config(cfg, "heat_vent")
    for _ in range(36):                          # 3 hot hours: the vent integral builds up
        ctrl({"temperature": target + 3})
    assert ctrl.vent.integral > 0
    heater, vent = ctrl({"temperature": target - 0.5})
    assert heater > 0 and vent == 0.0            # heating has priority
    assert ctrl.vent.integral == 0.0             # and the vent loop starts afresh


# Warm start: the greenhouse begins near the target with a warm floor, energy counters at zero
def test_warm_start_begins_in_a_heated_greenhouse(cfg):
    weather = build_weather(cfg, "winter_cloudy")
    cold = create_emulator(weather, cfg)
    warm = warm_emulator(weather, cfg)
    assert abs(warm.T - cfg["target"]["temperature"]) < 1.5
    assert warm.Tf > cold.Tf + 5
    assert warm.heater_energy_kwh == 0.0 and warm.time == weather["time"].iloc[0]


# Metrics on a small hand-made log: 30 min at the target, then 30 min 2 °C above with vents on
def test_metrics_on_a_known_log(cfg):
    target = cfg["target"]
    log = pd.DataFrame({"temperature": [target["temperature"]] * 6 + [target["temperature"] + 2.0] * 6,
                        "heater_power": [50.0] * 12, "ventilation_rate": [0.0] * 6 + [100.0] * 6,
                        "humidity": target["humidity"], "vpd": target["vpd"], "light_ppfd": 100.0,
                        "soil_moisture": 50.0})
    m = control_metrics(log, cfg)
    phys, area_kw = cfg["emulator"]["physics"], cfg["emulator"]["floor_area_m2"] / 1000
    assert m["hours"] == 1.0
    assert m["temp_rmse_c"] == pytest.approx(np.sqrt(2.0))
    assert m["temp_max_dev_c"] == pytest.approx(2.0)
    assert m["hours_in_band"] == 0.5 and m["hours_above_band"] == 0.5 and m["hours_below_band"] == 0
    assert m["warm_degree_hours"] == pytest.approx(0.5) and m["cold_degree_hours"] == 0  # 1 °C above for 0.5 h
    assert m["hours_heat_and_vent"] == pytest.approx(0.5)
    assert m["heater_kwh"] == pytest.approx(0.5 * phys["heater_max_power"] * area_kw * 1.0)
    assert m["vent_kwh"] == pytest.approx(1.0 * phys["vent_fan_power"] * area_kw * 0.5)
    assert m["heater_pct_hours"] == pytest.approx(50.0)
    assert m["vent_travel_pct"] == 100.0
    assert m["humidity_rmse_pct"] == 0.0 and "latency_mean_ms" not in m


# On a winter day the PID holds 22 °C within 0.5 °C; metrics agree with the emulator's energy meter
def test_pid_holds_the_winter_target(cfg):
    weather, emu = prepare_run(cfg, "winter_cloudy")
    log = run_offline(PIDController.from_config(cfg, "heat"), emu, weather)
    m = control_metrics(log, cfg)
    assert m["temp_rmse_c"] < 0.5
    assert (log["ventilation_rate"] == 0).all()
    assert m["heater_kwh"] == pytest.approx(emu.heater_energy_kwh)
    assert m["latency_max_ms"] < 50
