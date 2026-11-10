"""
Surrogate model of the greenhouse: state + actuators + weather now -> state 5 minutes later.

    inputs -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(n_state)

Inside, the inputs are standardised and the network predicts the standardised *change* of the
state; the model adds it back to the current state. The normalisation is stored in the model,
so one .pth file is all that is needed to use it.
"""

from pathlib import Path

import torch
from torch import nn

# Physical limits of the outputs, so chained predictions cannot drift into impossible values
BOUNDS = {"humidity": (0.0, 100.0), "vpd": (0.0, None), "light_lux": (0.0, None), "soil_moisture": (0.0, 100.0)}


# The neural network (MLP) with its input/output names and normalisation
class SurrogateModel(nn.Module):
    # Build the layers for the given inputs and outputs
    def __init__(self, state: list[str], controls: list[str], weather: list[str], hidden=(64, 32)):
        super().__init__()
        self.spec = {"state": list(state), "controls": list(controls), "weather": list(weather), "hidden": list(hidden)}
        self.state_names, self.control_names, self.weather_names = list(state), list(controls), list(weather)
        n_in, n_out = len(self.input_names), len(state)

        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

        self.register_buffer("in_mean", torch.zeros(n_in))
        self.register_buffer("in_std", torch.ones(n_in))
        self.register_buffer("delta_mean", torch.zeros(n_out))
        self.register_buffer("delta_std", torch.ones(n_out))
        lo = [BOUNDS.get(s, (None, None))[0] for s in state]
        hi = [BOUNDS.get(s, (None, None))[1] for s in state]
        self.register_buffer("out_min", torch.tensor([-torch.inf if v is None else v for v in lo]), persistent=False)
        self.register_buffer("out_max", torch.tensor([torch.inf if v is None else v for v in hi]), persistent=False)

    # All input names in order: state, then actuators, then weather
    @property
    def input_names(self) -> list[str]:
        return self.state_names + self.control_names + self.weather_names

    # Number of state values the model predicts
    @property
    def n_state(self) -> int:
        return len(self.state_names)

    # Store the mean and spread of inputs and changes (computed on the training data)
    def set_normalization(self, in_mean, in_std, delta_mean, delta_std):
        for name, v in [("in_mean", in_mean), ("in_std", in_std), ("delta_mean", delta_mean), ("delta_std", delta_std)]:
            v = torch.as_tensor(v, dtype=torch.float32)
            getattr(self, name).copy_(torch.where(v.abs() < 1e-8, torch.ones_like(v), v) if "std" in name else v)

    # Inputs (B, n_in) in physical units -> standardised predicted change (what the network is trained on)
    def normalized_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.in_mean) / self.in_std)

    # Inputs (B, n_in) -> next state (B, n_state) in physical units, within the physical limits
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.normalized_delta(x) * self.delta_std + self.delta_mean
        return torch.clamp(x[:, : self.n_state] + delta, self.out_min, self.out_max)

    # Predict N steps ahead by feeding each prediction back in (this is what the MPC does).
    # state0 (B, n_state), controls (B, N, 2), weather (B, N, n_weather) -> states at t+1..t+N (B, N, n_state).
    # Differentiable with respect to the controls.
    def rollout(self, state0: torch.Tensor, controls: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        state, out = state0, []
        for k in range(controls.shape[1]):
            state = self.forward(torch.cat([state, controls[:, k], weather[:, k]], dim=-1))
            out.append(state)
        return torch.stack(out, dim=1)

    # Save weights, normalisation and input/output names to one .pth file
    def save(self, path: str | Path, info: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"spec": self.spec, "state_dict": self.state_dict(), "info": info or {}}, path)

    # Load a saved model, ready for predictions
    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["spec"])
        model.load_state_dict(ckpt["state_dict"])
        model.info = ckpt.get("info", {})
        return model.eval()

    # Number of trainable weights
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
