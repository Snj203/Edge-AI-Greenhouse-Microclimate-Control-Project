"""
Run a controller as its own MQTT process (two terminals).

    terminal 1:  python scripts/run_controller.py                       # PID, heater + ventilation
                 python scripts/run_controller.py --mode heat           # PID, heater only
                 python scripts/run_controller.py --controller nn_mpc   # NN-MPC (needs the forecast)
    terminal 2:  python scripts/run_emulator.py --lockstep --interval 0 --warm               # PID
                 python scripts/run_emulator.py --lockstep --interval 0 --warm --forecast 12  # NN-MPC

Answers every sensor message with a command. Stops when the emulator reports "finished" and
prints the compute time per step. Without --forecast the NN-MPC assumes the current weather stays.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.controllers.controller_nn_mpc import NNMPCController
from src.controllers.controller_pid import MODES, PIDController
from src.controllers.mqtt_node import ControllerNode
from src.utils.config import load_config


# Start the controller node and wait until the emulator finishes (or Ctrl+C)
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", choices=["pid", "nn_mpc"], default="pid")
    parser.add_argument("--mode", choices=MODES, default="heat_vent", help="PID variant")
    parser.add_argument("--horizon", type=int, help="NN-MPC horizon in steps (default: from the config)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if args.controller == "pid":
        controller = PIDController.from_config(cfg, args.mode)
    else:
        controller = NNMPCController.from_config(cfg, **({"horizon": args.horizon} if args.horizon else {}))
    node = ControllerNode(cfg, controller, verbose=not args.quiet)
    try:
        node.start()
    except ConnectionError as e:
        print(f"ERROR: {e}")
        return 1
    print(f"[{controller.name}] waiting for sensor data on {node.bridge.topic_sensors} (Ctrl+C to stop)")
    try:
        while not node.wait(0.5):
            pass
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        node.stop()
    lat = np.array(list(node.latency_ms.values()))
    if len(lat):
        print(f"[{controller.name}] {node.commands_sent} commands; compute time per step: mean {lat.mean():.3f} ms, "
              f"max {lat.max():.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
