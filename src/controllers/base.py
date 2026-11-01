"""
Common interface of the controllers. A controller turns one sensor message (the same dict that is
sent over MQTT) into actuator settings: obs -> (heater_power %, ventilation_rate %).
"""

import time


# Base class: every controller has a name, can be reset and measures its compute time per step
class Controller:
    name = "controller"

    # No compute time measured yet
    def __init__(self):
        self.last_latency_ms = 0.0

    # Forget the internal state (integrals, previous errors) before a new run
    def reset(self):
        pass

    # The control rule itself; each controller defines its own
    def compute(self, obs: dict) -> tuple[float, float]:
        raise NotImplementedError

    # controller(obs) = compute + time measurement. `step` lets a controller be used as policy(obs, k)
    def __call__(self, obs: dict, step: int | None = None) -> tuple[float, float]:
        t0 = time.perf_counter()
        heater, vent = self.compute(obs)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return float(heater), float(vent)
