"""
A simple on/off thermostat over MQTT, to test the emulator loop end to end (not the PID).
Heater on below target − band, vents on above target + band.

    terminal 1:  python scripts/demo_controller.py
    terminal 2:  python scripts/run_emulator.py --lockstep --interval 0

Stops when the emulator reports "finished".
"""

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.mqtt_bridge import MQTTBridge


# On/off thermostat: answer every sensor message with a command until the emulator finishes
def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=float, default=cfg["target"]["temperature"])
    parser.add_argument("--band", type=float, default=1.0, help="°C dead band around the target")
    args = parser.parse_args()

    finished = threading.Event()
    bridge = MQTTBridge(cfg, client_name="demo-controller")

    # heater on below target − band, vents on above target + band
    def on_sensors(data: dict):
        t = data["temperature"]
        heater = 100.0 if t < args.target - args.band else 0.0
        vent = 100.0 if t > args.target + args.band else 0.0
        bridge.publish_actuators({"heater_power": heater, "ventilation_rate": vent, "step": data.get("step")})

    # stop when the emulator reports "finished"
    def on_status(data: dict):
        print(f"[controller] emulator {data.get('event')}")
        if data.get("event") == "finished":
            finished.set()

    bridge.connect()
    bridge.subscribe_sensors(on_sensors)
    bridge.subscribe(cfg["mqtt"]["topic_status"], on_status)
    print(f"[controller] on/off thermostat, target {args.target} ± {args.band} °C — waiting for the emulator "
          "(Ctrl+C to stop)")
    try:
        while not finished.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    bridge.disconnect()


if __name__ == "__main__":
    main()
