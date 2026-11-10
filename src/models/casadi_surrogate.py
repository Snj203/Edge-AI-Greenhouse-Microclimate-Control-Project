"""
The trained network rebuilt as a CasADi function, for the NN-MPC.

The NN-MPC evaluates the network's 12-step chained prediction and its gradient ~30 times per
control step. Through PyTorch one such evaluation takes ~5 ms, almost all of it overhead, because
the network is tiny. CasADi computes the same thing with exact derivatives in ~0.4 ms. The PyTorch
weights are copied, so both give the same outputs up to float32 rounding (checked in tests).
"""

import math

import casadi as ca
import torch

from src.models.architecture import SurrogateModel


# One step of the network as a CasADi function, exactly like model.forward:
# x = [state, actuators, weather] in physical units -> next state in physical units
def casadi_step(model: SurrogateModel) -> ca.Function:
    # tensor -> numpy array in double precision
    def np_(t):
        return t.detach().double().numpy()

    n_state, n_in = model.n_state, len(model.input_names)
    x = ca.SX.sym("x", n_in)
    h = (x - np_(model.in_mean)) / np_(model.in_std)
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    for i, layer in enumerate(linears):
        h = ca.mtimes(np_(layer.weight), h) + np_(layer.bias)
        if i < len(linears) - 1:
            h = ca.fmax(h, 0)  # ReLU
    nxt = x[:n_state] + h * np_(model.delta_std) + np_(model.delta_mean)

    # physical limits (humidity 0–100 %, light ≥ 0, ...), only where a limit exists
    lo, hi = model.out_min.tolist(), model.out_max.tolist()
    parts = []
    for i in range(n_state):
        v = nxt[i]
        if math.isfinite(lo[i]):
            v = ca.fmax(v, lo[i])
        if math.isfinite(hi[i]):
            v = ca.fmin(v, hi[i])
        parts.append(v)
    return ca.Function("surrogate_step", [x], [ca.vertcat(*parts)], ["x"], ["next_state"])
