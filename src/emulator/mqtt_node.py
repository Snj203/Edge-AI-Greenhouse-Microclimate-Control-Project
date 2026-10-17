"""
The emulator as an MQTT node.

Every step it publishes the sensor readings to `greenhouse/sensors/data` and applies the last
command received on `greenhouse/actuators/control`. Start and end are announced on
`greenhouse/emulator/status`.

    realtime  one step every `interval_sec` seconds, using the last command received
    lockstep  after publishing step k it waits for the controller's command for step k,
              so a run gives the same result however fast the controller is
"""

import threading
import time

import pandas as pd

from src.emulator.greenhouse_emulator import GreenhouseEmulator
from src.emulator.simulation import weather_at
from src.utils.mqtt_bridge import MQTTBridge


# Runs the emulator step by step and talks to the controller over MQTT
class EmulatorNode:
    # Settings, the last received command and the MQTT client
    def __init__(self, cfg: dict, emulator: GreenhouseEmulator, weather: pd.DataFrame,
                 interval_sec: float | None = None, lockstep: bool = False, command_timeout: float = 5.0,
                 verbose: bool = True):
        self.emu = emulator
        self.weather = weather
        self.interval = cfg["mqtt"]["publish_interval_sec"] if interval_sec is None else interval_sec
        self.lockstep = lockstep
        self.command_timeout = command_timeout
        self.verbose = verbose
        self.topic_status = cfg["mqtt"]["topic_status"]

        self._lock = threading.Lock()
        self._new_command = threading.Event()
        self._command = {"heater_power": 0.0, "ventilation_rate": 0.0, "step": None}
        self.commands_received = 0
        self.timeouts = 0
        self.bridge = MQTTBridge(cfg, client_name="emulator")

    # Called for every actuator command: check it, clamp to 0–100 %, store it
    def _on_command(self, payload: dict):
        try:
            heater = float(payload.get("heater_power", self._command["heater_power"]))
            vent = float(payload.get("ventilation_rate", self._command["ventilation_rate"]))
        except (TypeError, ValueError):
            print(f"[emulator] ignored malformed command: {payload}")
            return
        with self._lock:
            self._command = {"heater_power": min(max(heater, 0.0), 100.0),
                             "ventilation_rate": min(max(vent, 0.0), 100.0),
                             "step": payload.get("step")}
            self.commands_received += 1
        self._new_command.set()

    # Lockstep: wait for the command for step k (or one without a step number), or time out
    def _wait_for_command(self, k: int):
        deadline = time.monotonic() + self.command_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._new_command.wait(remaining):
                self.timeouts += 1
                return
            with self._lock:
                step = self._command["step"]
                self._new_command.clear()
            if step is None or int(step) >= k:
                return

    # Main loop: publish readings -> get the command -> make a step. Returns the run log
    def run(self) -> pd.DataFrame:
        self.bridge.connect()
        self.bridge.subscribe_actuators(self._on_command)
        time.sleep(0.2)  # let the subscription settle
        n = len(self.weather)
        self.bridge.publish(self.topic_status, {"event": "started", "steps": n,
                                                "profile": self.weather.attrs.get("profile"),
                                                "lockstep": self.lockstep})
        if self.verbose:
            mode = "lockstep" if self.lockstep else f"realtime, {self.interval}s/step"
            print(f"[emulator] {n} steps ({mode}) — publishing to {self.bridge.topic_sensors}")

        rows = []
        next_tick = time.monotonic()
        try:
            for k in range(n):
                w = weather_at(self.weather, k)
                obs = self.emu.observe(w)
                if self.lockstep:
                    self._new_command.clear()  # wait for the reply to this step only
                self.bridge.publish_sensors({"step": k, **obs}, timestamp=self.emu.time.isoformat())

                if self.lockstep:
                    self._wait_for_command(k)
                else:
                    next_tick += self.interval
                    time.sleep(max(0.0, next_tick - time.monotonic()))

                with self._lock:
                    heater, vent = self._command["heater_power"], self._command["ventilation_rate"]
                rows.append({"time": self.emu.time, "step": k, **obs, "heater_power": heater,
                             "ventilation_rate": vent, "heater_energy_kwh": self.emu.heater_energy_kwh,
                             "vent_energy_kwh": self.emu.vent_energy_kwh})
                self.emu.step(w, heater, vent)

                if self.verbose and (k % 12 == 0 or k == n - 1):
                    print(f"  {obs['temperature']:6.2f} °C  {obs['humidity']:5.1f} %RH  "
                          f"heater {heater:5.1f} %  vent {vent:5.1f} %   [{self.emu.time:%H:%M}] step {k + 1}/{n}")
        finally:
            self.bridge.publish(self.topic_status, {"event": "finished", "steps": len(rows)}, wait=True)
            self.bridge.disconnect()
        if self.verbose:
            print(f"[emulator] done: {len(rows)} steps, {self.commands_received} commands received, "
                  f"{self.timeouts} waits timed out")
        return pd.DataFrame(rows)
