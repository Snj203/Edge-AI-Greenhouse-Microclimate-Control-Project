"""
Run the greenhouse emulator as an MQTT node.

    python scripts/run_emulator.py                                # real day, 1 s per step
    python scripts/run_emulator.py --profile winter_cloudy --interval 0.05
    python scripts/run_emulator.py --lockstep --interval 0        # wait for a controller every step
    python scripts/run_emulator.py --lockstep --interval 0 --warm # with scripts/run_controller.py (PID)

Watch the messages:  mosquitto_sub -t "greenhouse/#" -v
Saves the run to results/emulator/run_<profile>.csv and a plot next to it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controllers.experiment import warm_emulator
from src.emulator.mqtt_node import EmulatorNode
from src.emulator.simulation import create_emulator
from src.emulator.weather import ALL_PROFILES, build_weather
from src.utils.config import ensure_dir, load_config
from src.utils.plotting import plot_run


# Read the options, run the emulator over MQTT, save the log and the plot
def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=cfg["emulator"]["weather_profile"], choices=ALL_PROFILES)
    parser.add_argument("--hours", type=float, default=cfg["emulator"]["simulation_hours"])
    parser.add_argument("--interval", type=float, default=cfg["mqtt"]["publish_interval_sec"],
                        help="real seconds per simulated step (realtime mode)")
    parser.add_argument("--lockstep", action="store_true", help="wait for a controller command every step")
    parser.add_argument("--timeout", type=float, default=5.0, help="lockstep: max seconds to wait per step")
    parser.add_argument("--seed", type=int, default=0, help="random seed of the synthetic weather")
    parser.add_argument("--warm", action="store_true",
                        help="start from a greenhouse already held at the target (as in the controller runs)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    weather = build_weather(cfg, args.profile, args.hours, args.seed)
    emu = warm_emulator(weather, cfg) if args.warm else create_emulator(weather, cfg)
    node = EmulatorNode(cfg, emu, weather, interval_sec=args.interval, lockstep=args.lockstep,
                        command_timeout=args.timeout, verbose=not args.quiet)
    try:
        log = node.run()
    except ConnectionError as e:
        print(f"ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nstopped")
        return 1

    out = ensure_dir("results/emulator")
    log.to_csv(out / f"run_{args.profile}.csv", index=False)
    plot_run(log, out / f"run_{args.profile}.png", f"Emulator run over MQTT — {args.profile}", cfg["target"])
    print(f"Saved {out / f'run_{args.profile}.csv'} and the plot")
    print(f"Heater energy {emu.heater_energy_kwh:.1f} kWh, ventilation {emu.vent_energy_kwh:.1f} kWh, "
          f"irrigations {emu.irrigation_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
