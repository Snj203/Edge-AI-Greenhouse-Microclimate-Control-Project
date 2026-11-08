"""
NN-MPC controller: model predictive control with the trained network as the greenhouse model.

Every control step (5 min):
  1. Read the sensors and the weather forecast for the horizon (without a forecast in the message
     the current weather is assumed to stay).
  2. Update the temperature bias (model correction, below).
  3. Find the actuator plan u_0 … u_{N−1} (heater and vents, 0–100 %) that minimises

         J = Σ_k  α·E(u_k) + β·(T_{k+1} − T_target)² + γ·|Δu_k|²

     T_{k+1}  temperature predicted by chaining the network over the plan
     E(u_k)   heater + fan energy of step k in kWh for the whole greenhouse
     Δu_k     change of the actuator levels (fractions 0–1) from the previous step. The plan's
              formula has no such term; without it the MPC switches the heater up and down every
              step. γ is tuned like the PID gains (scripts/tune_mpc.py).
  4. Apply only u_0 (receding horizon). The next step starts from the plan shifted by one step.
     Like the PID, the vents stay closed while the heater is on.

Optimiser: SciPy SLSQP with bounds 0–100 %; the exact gradient dJ/du comes from the network rebuilt
in CasADi (src/models/casadi_surrogate.py).

Model correction (offset-free MPC). In closed loop the network has a small steady error: it was
trained on days that start from a greenhouse without control, while the runs start from a heated
one, and the warm floor is not among its inputs. Without a correction the MPC holds 22.2 °C instead
of 22.0 in winter. So at every step the controller compares the new reading with its own one-step
prediction from the previous step, moves a temperature bias d by `bias_gain` × that error and adds
d to every predicted step. This works like the I part of a PID.
"""

from pathlib import Path

import casadi as ca
import numpy as np
from scipy.optimize import minimize

from src.controllers.base import Controller
from src.models.architecture import SurrogateModel
from src.models.casadi_surrogate import casadi_step
from src.utils.config import resolve_path

BIAS_LIMIT = 0.5         # °C per step; limits the correction if a reading is ever wildly off
INTERLOCK_LEVEL = 0.005  # heater fraction (0.5 %) above which the vents stay closed


# The NN-MPC controller
class NNMPCController(Controller):
    # Store the settings and build the cost function J with its gradient in CasADi
    def __init__(self, model: SurrogateModel, target: float, horizon: int = 12, alpha: float = 1.0,
                 beta: float = 10.0, kwh_per_step: tuple[float, float] = (2.5, 0.0833),
                 u_max: tuple[float, float] = (100.0, 100.0), move_weight: float = 0.0, max_iterations: int = 100,
                 bias_gain: float = 0.3, interlock: bool = True, name: str = "nn_mpc"):
        super().__init__()
        if list(model.control_names) != ["heater_power", "ventilation_rate"]:
            raise ValueError("the network must take heater_power and ventilation_rate as actuators")
        self.model = model
        self.target, self.N = float(target), int(horizon)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(move_weight)
        self.kwh = np.asarray(kwh_per_step, float)  # kWh per step at 100 % (heater, fans)
        self.u_max = np.asarray(u_max, float)
        self.max_iterations = int(max_iterations)
        self.bias_gain, self.interlock = float(bias_gain), bool(interlock)
        self.forecast_steps = self.N
        self.name = name
        self.state_names, self.weather_names = model.state_names, model.weather_names
        self.n_s, self.n_w = model.n_state, len(model.weather_names)
        self._step = casadi_step(model)
        self._fg = self._build_cost()
        self.reset()

    # Build from the config (`mpc` section); keyword arguments override it (e.g. horizon, move_weight)
    @classmethod
    def from_config(cls, cfg: dict, model_path: str | Path | None = None, **overrides) -> "NNMPCController":
        m = cfg["mpc"]
        emu = cfg["emulator"]
        phys = emu["physics"]
        area_kw, dt_h = emu["floor_area_m2"] / 1000.0, emu["timestep_minutes"] / 60.0
        model = SurrogateModel.load(resolve_path(model_path or cfg["nn"]["model_path"]))
        kwargs = dict(
            target=cfg["target"]["temperature"], horizon=m["horizon_steps"], alpha=m["alpha"], beta=m["beta"],
            kwh_per_step=(float(phys["heater_max_power"]) * area_kw * dt_h,
                          float(phys["vent_fan_power"]) * area_kw * dt_h),
            u_max=(m["max_heater_power"], m["max_vent_rate"]), move_weight=m["move_weight"],
            max_iterations=m["max_iterations"], bias_gain=m["bias_gain"], interlock=m["interlock"])
        kwargs.update(overrides)
        return cls(model, **kwargs)

    # J(plan) and dJ/dplan as one CasADi function of: plan x (fractions h0, v0, h1, v1, ...), current
    # state s0, forecast w, temperature bias d, last applied command u_prev, first (0 = nothing applied yet)
    def _build_cost(self) -> ca.Function:
        N, n_s, n_w = self.N, self.n_s, self.n_w
        x = ca.SX.sym("x", 2 * N)
        s0 = ca.SX.sym("s0", n_s)
        w = ca.SX.sym("w", N * n_w)
        d = ca.SX.sym("d")
        u_prev = ca.SX.sym("u_prev", 2)
        first = ca.SX.sym("first")
        s, J, last = s0, 0, u_prev
        for k in range(N):
            uk = x[2 * k:2 * k + 2]
            s = self._step(ca.vertcat(s, uk * self.u_max, w[k * n_w:(k + 1) * n_w]))
            s = ca.vertcat(s[0] + d, s[1:])
            move = ca.sumsqr(uk - last) * (first if k == 0 else 1)
            J += self.alpha * ca.dot(uk, self.kwh) + self.beta * (s[0] - self.target) ** 2 + self.gamma * move
            last = uk
        return ca.Function("mpc_cost", [x, s0, w, d, u_prev, first], [J, ca.gradient(J, x)])

    # J and its gradient for a plan x, state s0, forecast w (flat), bias d, last command u_prev (None = none yet)
    def cost(self, x: np.ndarray, s0: np.ndarray, w: np.ndarray, d: float = 0.0,
             u_prev: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        first = 0.0 if u_prev is None else 1.0
        u_prev = np.zeros(2) if u_prev is None else np.asarray(u_prev, float)
        J, g = self._fg(x, s0, w, d, u_prev, first)
        return float(J), np.asarray(g).ravel()

    # Start a new run: empty plan, no bias, nothing applied yet
    def reset(self):
        self.plan = np.zeros(2 * self.N)
        self.bias = 0.0
        self._last_input: np.ndarray | None = None  # [state, actuators, weather] of the previous step
        self._last_u: np.ndarray | None = None      # last applied command, fractions 0–1

    # Weather for the N steps of the horizon (N × n_weather); the current weather if there is no forecast
    def forecast(self, obs: dict) -> np.ndarray:
        rows = [[f[n] for n in self.weather_names] for f in obs.get("forecast", [])[:self.N]]
        if not rows:
            rows = [[obs[n] for n in self.weather_names]]
        rows += [rows[-1]] * (self.N - len(rows))
        return np.asarray(rows, float)

    # One network step (without the bias) for an input vector [state, actuators %, weather]
    def predict_next(self, x_input: np.ndarray) -> np.ndarray:
        return np.asarray(self._step(x_input)).ravel()

    # One control step: update the bias, optimise the plan, apply its first step
    def compute(self, obs: dict) -> tuple[float, float]:
        s0 = np.array([obs[n] for n in self.state_names], float)
        w = self.forecast(obs)
        if self._last_input is not None:
            predicted = self.predict_next(self._last_input)[0] + self.bias
            self.bias = float(np.clip(self.bias + self.bias_gain * (s0[0] - predicted), -BIAS_LIMIT, BIAS_LIMIT))

        res = minimize(self.cost, self.plan, args=(s0, w.ravel(), self.bias, self._last_u), jac=True,
                       method="SLSQP", bounds=[(0.0, 1.0)] * (2 * self.N), options={"maxiter": self.max_iterations})
        plan = np.clip(res.x, 0.0, 1.0).reshape(self.N, 2)
        if self.interlock:  # no ventilation while the heater is on
            plan[plan[:, 0] > INTERLOCK_LEVEL, 1] = 0.0
        self.plan = np.vstack([plan[1:], plan[-1:]]).ravel()  # starting point for the next step
        self._last_u = plan[0].copy()
        heater, vent = plan[0] * self.u_max
        self._last_input = np.concatenate([s0, [heater, vent], w[0]])
        self.last_info = {"mpc_iterations": int(res.nit), "mpc_converged": bool(res.success),
                          "mpc_cost": float(res.fun), "mpc_bias": self.bias}
        return float(heater), float(vent)
