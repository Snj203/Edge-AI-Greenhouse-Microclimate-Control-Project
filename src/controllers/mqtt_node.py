"""
A controller as an MQTT node: every sensor message in -> one actuator command out.

    subscribed to  greenhouse/sensors/data, greenhouse/emulator/status
    publishes to   greenhouse/actuators/control  {"heater_power", "ventilation_rate", "step"}

The command carries the step of the message it answers, so an emulator in lockstep knows it is
the reply to that step. The controller is reset when the emulator reports "started".
"""

import threading

from src.controllers.base import Controller
from src.utils.mqtt_bridge import MQTTBridge


# Connects a controller to the broker and answers every sensor message
class ControllerNode:
    # Settings, MQTT client and the per-step compute times
    def __init__(self, cfg: dict, controller: Controller, verbose: bool = False):
        self.controller = controller
        self.verbose = verbose
        self.topic_status = cfg["mqtt"]["topic_status"]
        self.bridge = MQTTBridge(cfg, client_name=f"controller-{controller.name}")
        self.finished = threading.Event()
        self.latency_ms: dict[int, float] = {}  # step -> compute time
        self.commands_sent = 0

    # Sensor message received: compute the command and send it back with the same step number
    def _on_sensors(self, obs: dict):
        heater, vent = self.controller(obs)
        step = obs.get("step")
        self.bridge.publish_actuators({"heater_power": heater, "ventilation_rate": vent, "step": step})
        self.commands_sent += 1
        if step is not None:
            self.latency_ms[int(step)] = self.controller.last_latency_ms
        if self.verbose and step is not None and int(step) % 12 == 0:
            print(f"  [{self.controller.name}] step {step}: {obs['temperature']:6.2f} °C -> heater {heater:5.1f} %  "
                  f"vent {vent:5.1f} %  ({self.controller.last_latency_ms:.3f} ms)")

    # Emulator status: "started" resets the controller, "finished" ends the waiting
    def _on_status(self, data: dict):
        event = data.get("event")
        if event == "started":
            self.controller.reset()
            self.latency_ms.clear()
            self.finished.clear()
        elif event == "finished":
            self.finished.set()
        if self.verbose:
            print(f"[{self.controller.name}] emulator {event}")

    # Connect and subscribe to the status and sensor topics
    def start(self) -> "ControllerNode":
        self.bridge.connect()
        self.bridge.subscribe(self.topic_status, self._on_status)
        self.bridge.subscribe_sensors(self._on_sensors)
        return self

    # Disconnect from the broker
    def stop(self):
        self.bridge.disconnect()

    # Wait until the emulator reports "finished" (True) or the timeout passes (False)
    def wait(self, timeout: float | None = None) -> bool:
        return self.finished.wait(timeout)
