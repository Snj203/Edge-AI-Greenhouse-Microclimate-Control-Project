"""Training and evaluation of the surrogate model on the prepared data (data/processed)."""

import copy
import time

import numpy as np
import pandas as pd
import torch

from src.models.architecture import SurrogateModel
from src.models.data import CONTROLS, STATE_EXT, STATE_PLAN, WEATHER_EXT, WEATHER_PLAN
from src.utils.config import resolve_path

# A sample is used only if the 4 steps before it belong to the same day. The MLP does not need them,
# but this keeps the same samples for sequence models (LSTM, in the roadmap), so they compare fairly.
HISTORY = 5


# The two networks that are compared: the plan's inputs and the extended inputs
def variants(cfg: dict) -> dict:
    hidden = cfg["nn"]["hidden_layers"]
    return {
        "mlp_plan": dict(state=STATE_PLAN, controls=CONTROLS, weather=WEATHER_PLAN, hidden=hidden),  # 9 inputs
        "mlp": dict(state=STATE_EXT, controls=CONTROLS, weather=WEATHER_EXT, hidden=hidden),         # 12 inputs
    }


# Read train/val/test made by scripts/prepare_training_data.py
def load_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    folder = resolve_path(cfg["nn"]["data"]["output_dir"])
    if not (folder / "train.csv").exists():
        raise FileNotFoundError(f"No training data in {folder}. Run: python scripts/prepare_training_data.py")
    return {s: pd.read_csv(folder / f"{s}.csv").sort_values(["episode", "step"]).reset_index(drop=True)
            for s in ("train", "val", "test")}


# True for rows that have the previous n-1 steps of the same day right before them
def history_ok(df: pd.DataFrame, n: int) -> np.ndarray:
    ok = np.ones(len(df), bool)
    for j in range(1, n):
        ok &= ((df["episode"].shift(j) == df["episode"]) & (df["step"].shift(j) == df["step"] - j)).to_numpy()
    return ok


# True for rows that start n consecutive valid steps of the same day (needed for chained predictions)
def chain_ok(df: pd.DataFrame, n: int) -> np.ndarray:
    ok = df["valid"].to_numpy().copy()
    for j in range(1, n):
        ok &= ((df["episode"].shift(-j) == df["episode"]) & (df["step"].shift(-j) == df["step"] + j)
               & df["valid"].shift(-j, fill_value=False)).to_numpy()
    return ok


# Inputs and next-state targets of one split, for one model variant
class Samples:
    # Turn the table into arrays and find the usable rows
    def __init__(self, df: pd.DataFrame, spec: dict):
        self.df = df
        self.state = spec["state"]
        self.A = df[spec["state"] + spec["controls"] + spec["weather"]].to_numpy(np.float32)
        self.next = df[[f"next_{s}" for s in self.state]].to_numpy(np.float32)
        self.hist = history_ok(df, HISTORY)
        self.idx = np.flatnonzero(self.hist & df["valid"].to_numpy())

    # Inputs and targets of the usable rows (optionally only rows where mask is True)
    def one_step(self, mask: np.ndarray | None = None):
        rows = self.idx if mask is None else self.idx[mask[self.idx]]
        return self.A[rows], self.next[rows]


# Train one variant: MSE on the standardised 5-min change, Adam, early stopping on validation loss.
# Returns the model with the best validation loss, the training log and the time taken.
def train(spec: dict, data: dict[str, pd.DataFrame], cfg: dict, epochs: int, seed: int = 0, verbose=True):
    torch.manual_seed(seed)
    model = SurrogateModel(**spec)
    Xtr, Ytr = Samples(data["train"], spec).one_step()
    Xva, Yva = Samples(data["val"], spec).one_step()
    n_state = len(spec["state"])

    delta = Ytr - Xtr[:, :n_state]
    model.set_normalization(Xtr.mean(0), Xtr.std(0), delta.mean(0), delta.std(0))

    # inputs as a tensor and the target: standardised change of the state
    def to_tensors(X, Y):
        X = torch.from_numpy(X)
        return X, (torch.from_numpy(Y) - X[:, :n_state] - model.delta_mean) / model.delta_std

    Xtr_t, Ttr = to_tensors(Xtr, Ytr)
    Xva_t, Tva = to_tensors(Xva, Yva)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["nn"]["learning_rate"])
    loss_fn = torch.nn.MSELoss()
    batch = cfg["nn"]["batch_size"]
    patience = cfg["nn"]["early_stopping_patience"]

    best, best_state, wait, log = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model.normalized_delta(Xtr_t[b]), Ttr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model.normalized_delta(Xva_t), Tva).item()
            pred = model(Xva_t).numpy()
        val_t_rmse = float(np.sqrt(np.mean((pred[:, 0] - Yva[:, 0]) ** 2)))
        log.append({"epoch": epoch, "train_loss": total / len(perm), "val_loss": val, "val_temp_rmse": val_t_rmse})
        if val < best - 1e-6:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d}  train {log[-1]['train_loss']:.4f}  val {val:.4f}  "
                  f"val T-RMSE {val_t_rmse:.3f} °C  ({time.time() - t0:.0f}s)")
        if wait >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} (best val loss {best:.4f})")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(log), time.time() - t0


# Root mean square error
def rmse(a, b, axis=0):
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))


# Error 5 minutes ahead for every output, and of the "nothing changes" guess
@torch.no_grad()
def eval_one_step(model, s: Samples, mask=None) -> dict:
    X, Y = s.one_step(mask)
    if len(X) == 0:
        return {}
    pred = model(torch.from_numpy(X)).numpy()
    persist = X[:, : len(s.state)]
    return {name: {"rmse": float(rmse(pred[:, i], Y[:, i])), "persistence_rmse": float(rmse(persist[:, i], Y[:, i])),
                   "n": int(len(Y))} for i, name in enumerate(s.state)}


# Chained predictions n steps ahead, started every `stride` steps, with the real actuators and weather.
# Returns predictions, true values, the "nothing changes" guess and the start rows.
@torch.no_grad()
def rollout_predictions(model, s: Samples, n_steps: int, mask=None, stride: int = 3):
    ok = chain_ok(s.df, n_steps) & s.hist
    if mask is not None:
        ok &= mask
    starts = np.flatnonzero(ok)
    starts = starts[(s.df["step"].to_numpy()[starts] % stride) == 0]
    if len(starts) == 0:
        return None
    n_s, n_c = len(s.state), len(CONTROLS)
    rows = starts[:, None] + np.arange(n_steps)
    steps = s.A[rows]  # (B, N, n_in): rows t .. t+N-1
    pred = model.rollout(torch.from_numpy(s.A[starts, :n_s]), torch.from_numpy(steps[:, :, n_s:n_s + n_c]),
                         torch.from_numpy(steps[:, :, n_s + n_c:])).numpy()
    truth = s.next[rows]
    persist = np.repeat(s.A[starts, None, :n_s], n_steps, axis=1)
    return pred, truth, persist, starts


# Error of chained predictions for every output and every horizon (5 min .. n_steps × 5 min)
def eval_rollout(model, s: Samples, n_steps: int, mask=None) -> dict:
    r = rollout_predictions(model, s, n_steps, mask)
    if r is None:
        return {}
    pred, truth, persist, starts = r
    return {name: {"rmse_by_horizon": rmse(pred[:, :, i], truth[:, :, i]).round(4).tolist(),
                   "persistence_by_horizon": rmse(persist[:, :, i], truth[:, :, i]).round(4).tolist(),
                   "n_rollouts": int(len(starts))} for i, name in enumerate(s.state)}


# Errors on validation and test days, separately for emulator and real days
def evaluate(model, data: dict, spec: dict, n_steps: int) -> dict:
    out = {}
    for split in ("val", "test"):
        s = Samples(data[split], spec)
        emu = (s.df["source"] == "emulator").to_numpy()
        out[split] = {
            "emulator": {"one_step": eval_one_step(model, s, emu), "rollout": eval_rollout(model, s, n_steps, emu)},
            "real": {"one_step": eval_one_step(model, s, ~emu), "rollout": eval_rollout(model, s, n_steps, ~emu)},
        }
    return out


# Time of one prediction and of a chained prediction over n_steps, in ms
@torch.no_grad()
def latency_ms(model, n_steps: int, repeats: int = 300) -> dict:
    state0 = torch.zeros(1, model.n_state)
    controls = torch.zeros(1, n_steps, len(model.control_names))
    weather = torch.zeros(1, n_steps, len(model.weather_names))
    x = torch.zeros(1, len(model.input_names))
    for _ in range(20):  # warm-up
        model.rollout(state0, controls, weather)
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    one = (time.perf_counter() - t0) / repeats * 1000
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.rollout(state0, controls, weather)
    roll = (time.perf_counter() - t0) / repeats * 1000
    return {"one_step_ms": round(one, 4), f"rollout_{n_steps}_steps_ms": round(roll, 4)}
