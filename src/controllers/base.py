"""
Common interface of the controllers. A controller turns one sensor message (the same dict that is
sent over MQTT) into actuator settings: obs -> (heater_power %, ventilation_rate %).

A controller that needs a weather forecast sets `forecast_steps` > 0; the message then carries
"forecast". Diagnostics of the last step (e.g. optimiser iterations) go into `last_info` and are
written to the run log next to the actuator settings.
"""

import time


# Base class: every controller has a name, can be reset and measures its compute time per step
class Controller:
    name = "controller"
    forecast_steps = 0

    # No compute time measured yet, no diagnostics
    def __init__(self):
        self.last_latency_ms = 0.0
        self.last_info: dict = {}

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
