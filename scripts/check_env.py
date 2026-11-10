"""
Check the environment: Python, packages, PyTorch device, dataset and MQTT broker.

    python scripts/check_env.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGES = ["torch", "numpy", "pandas", "scipy", "casadi", "paho.mqtt", "matplotlib", "seaborn", "yaml"]


# Import every package and print its version
def packages_ok() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"  ok  {name:<11} {getattr(module, '__version__', '')}")
        except ImportError as e:
            print(f"  !!  {name:<11} missing ({e})")
            ok = False
    return ok


# Check packages, PyTorch device, dataset and MQTT broker
def main() -> int:
    print(f"Python      {sys.version.split()[0]}  ({sys.executable})")
    if not packages_ok():
        print("\nSome packages are missing: pip install -r requirements.txt")
        return 1

    import torch
    from src.utils.config import load_config, resolve_path
    from src.utils.mqtt_bridge import MQTTBridge

    device = f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch     training on {device}")

    cfg = load_config()
    dataset = resolve_path(cfg["dataset"]["path"])
    ok = dataset.exists()
    print(f"Dataset     {'found' if ok else 'MISSING'}: {dataset.name}")

    bridge = MQTTBridge(cfg, client_name="check-env")
    try:
        bridge.connect(timeout=3)
        bridge.disconnect()
        print(f"MQTT        broker reachable at {bridge.broker_host}:{bridge.broker_port}")
    except ConnectionError as e:
        print(f"MQTT        !! {e}")
        ok = False

    print("\nEnvironment OK" if ok else "\nEnvironment has problems (see !! above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
