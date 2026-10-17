"""MQTT tests. They need Mosquitto on localhost:1883 and are skipped without it."""

import threading
import time

import pytest

from src.emulator.mqtt_node import EmulatorNode
from src.emulator.simulation import create_emulator
from src.emulator.weather import build_weather
from src.utils.config import load_config
from src.utils.mqtt_bridge import MQTTBridge

SENSOR_FIELDS = {"timestamp", "temperature", "humidity", "vpd", "light_lux", "light_ppfd", "soil_moisture",
                 "outdoor_temp", "solar_radiation"}


# Config with separate test topics; skips the tests if the broker is not running
@pytest.fixture(scope="module")
def cfg():
    cfg = load_config()
    try:
        MQTTBridge(cfg, client_name="probe").connect(timeout=2).disconnect()
    except ConnectionError:
        pytest.skip("Mosquitto broker is not running on localhost:1883")
    # separate topics, so the tests never mix with a real run
    cfg["mqtt"] |= {"topic_sensors": "test/sensors", "topic_actuators": "test/actuators", "topic_status": "test/status"}
    return cfg


# A message sent through the broker comes back to the subscriber unchanged
def test_bridge_round_trip(cfg):
    got = threading.Event()
    received = {}
    with MQTTBridge(cfg, client_name="t-bridge") as bridge:
        bridge.subscribe_sensors(lambda d: (received.update(d), got.set()))
        time.sleep(0.2)
        bridge.publish_sensors({"temperature": 21.5}, timestamp="2026-01-15T14:30:00")
        assert got.wait(3)
    assert received == {"timestamp": "2026-01-15T14:30:00", "temperature": 21.5}


# Emulator in lockstep gets a controller reply for every step
def test_emulator_lockstep_with_controller(cfg):
    weather = build_weather(cfg, "spring_variable", hours=2)
    emu = create_emulator(weather, cfg)
    messages = []
    controller = MQTTBridge(cfg, client_name="t-controller").connect()

    # test controller: record the readings and reply with a fixed command
    def on_sensors(data):
        messages.append(data)
        controller.publish_actuators({"heater_power": 40.0, "ventilation_rate": 10.0, "step": data["step"]})

    controller.subscribe_sensors(on_sensors)
    time.sleep(0.2)
    log = EmulatorNode(cfg, emu, weather, lockstep=True, command_timeout=2.0, verbose=False).run()
    controller.disconnect()

    assert len(messages) == len(weather) == len(log)
    assert SENSOR_FIELDS <= set(messages[0])
    assert [m["step"] for m in messages] == list(range(len(weather)))
    assert (log["heater_power"] == 40.0).all() and (log["ventilation_rate"] == 10.0).all()
    assert emu.heater_energy_kwh > 0
