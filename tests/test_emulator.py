from datetime import datetime

import pytest

from src.emulator.greenhouse_emulator import GreenhouseEmulator, PhysicsParams, relative_humidity, vapour_density
from src.emulator.simulation import create_emulator, run
from src.emulator.weather import ALL_PROFILES, build_weather
from src.utils.config import load_config

NIGHT = {"outdoor_temp": 10.0, "outdoor_rh": 90.0, "solar_radiation": 0.0, "wind_speed": 5.0}
NOON = {"outdoor_temp": 22.0, "outdoor_rh": 50.0, "solar_radiation": 800.0, "wind_speed": 5.0}


# Physics parameters from the config (given to the tests below)
@pytest.fixture
def params():
    return PhysicsParams.from_config(load_config())


# New emulator in a fixed starting state, irrigation off
def fresh(params):
    emu = GreenhouseEmulator(params, irrigation=False)
    emu.reset(temperature=18.0, humidity=75.0, soil_temperature=18.0, soil_moisture=55.0)
    return emu


# Run for some hours with fixed weather and actuators; returns the last readings
def run_for(emu, weather, hours, heater=0.0, vent=0.0):
    for _ in range(int(hours * 12)):
        obs = emu.step(weather, heater, vent)
    return obs


# RH -> vapour density -> RH gives the same value
def test_humidity_conversion_round_trip():
    assert relative_humidity(25.0, vapour_density(25.0, 60.0)) == pytest.approx(60.0)


# Heater on is clearly warmer than heater off
def test_heater_warms(params):
    off = run_for(fresh(params), NIGHT, 3)["temperature"]
    on = run_for(fresh(params), NIGHT, 3, heater=100)["temperature"]
    assert on > off + 5


# Vents cool the greenhouse, but not below the outdoor temperature
def test_ventilation_cools_but_not_below_outside(params):
    closed = run_for(fresh(params), NOON, 3)["temperature"]
    vented = run_for(fresh(params), NOON, 3, vent=100)["temperature"]
    assert vented < closed - 2
    assert vented > NOON["outdoor_temp"]


# Sun warms the air; light readings follow the transmission factor
def test_sun_warms_and_lights(params):
    obs = run_for(fresh(params), NOON, 3)
    assert obs["temperature"] > NOON["outdoor_temp"]
    assert obs["light_ppfd"] == pytest.approx(params.light_transmission * 800, rel=1e-3)
    assert obs["light_lux"] == pytest.approx(54.05 * obs["light_ppfd"], rel=1e-3)


# Humidity stays within 0–100 % and VPD is never negative
def test_humidity_stays_physical(params):
    foggy = {"outdoor_temp": 5.0, "outdoor_rh": 100.0, "solar_radiation": 0.0, "wind_speed": 0.0}
    obs = run_for(fresh(params), foggy, 24)
    assert 0 <= obs["humidity"] <= 100
    assert obs["vpd"] >= 0


# Energy (kWh) = power × level × area × time
def test_energy_accounting(params):
    emu = fresh(params)
    run_for(emu, NIGHT, 1, heater=50, vent=100)
    area_kw = emu.floor_area_m2 / 1000
    assert emu.heater_energy_kwh == pytest.approx(params.heater_max_power * 0.5 * area_kw)
    assert emu.vent_energy_kwh == pytest.approx(params.vent_fan_power * area_kw)


# Irrigation happens only in daytime
def test_irrigation_only_in_daytime(params):
    emu = GreenhouseEmulator(params, irrigation=True)
    emu.reset(soil_moisture=params.irrigation_threshold - 1, time=datetime(2025, 1, 1, 2, 0))
    emu.step(NIGHT)
    assert emu.irrigation_count == 0
    emu.reset(soil_moisture=params.irrigation_threshold - 1, time=datetime(2025, 1, 1, 11, 0))
    emu.step(NOON)
    assert emu.irrigation_count == 1
    assert emu.SM == pytest.approx(params.irrigation_target)


# Every weather profile runs 24 h with realistic values
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_profiles_run_24h(profile):
    cfg = load_config()
    weather = build_weather(cfg, profile)
    assert len(weather) == 24 * 60 // cfg["emulator"]["timestep_minutes"]
    assert not weather.isna().any().any()
    log = run(create_emulator(weather, cfg), weather)
    assert len(log) == len(weather)
    assert log["temperature"].between(-10, 60).all()
    assert log["humidity"].between(0, 100).all()


# Same seed gives the same weather, another seed a different one
def test_synthetic_weather_is_reproducible():
    cfg = load_config()
    a = build_weather(cfg, "spring_variable", seed=3)
    b = build_weather(cfg, "spring_variable", seed=3)
    c = build_weather(cfg, "spring_variable", seed=4)
    assert a.equals(b)
    assert not a.equals(c)
