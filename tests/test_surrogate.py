import numpy as np
import pandas as pd
import pytest
import torch

from src.models.architecture import SurrogateModel
from src.models.data import (CONTROLS, STATE_EXT, STATE_PLAN, WEATHER_EXT, WEATHER_PLAN, generate_emulator_data,
                             real_data, split_episodes)
from src.models.training import Samples, chain_ok, history_ok, train, variants
from src.utils.config import load_config


# Config shared by the tests in this file
@pytest.fixture(scope="module")
def cfg():
    return load_config()


# A small data set: 12 emulator days split 50/25/25
@pytest.fixture(scope="module")
def small_data(cfg):
    df = generate_emulator_data(cfg, episodes_per_profile=4, dataset_episodes_per_day=0, seed=1, verbose=False)
    df["split"] = split_episodes(df, (0.5, 0.25, 0.25), seed=1)
    return {s: df[df["split"] == s].sort_values(["episode", "step"]).reset_index(drop=True)
            for s in ("train", "val", "test")}


# The plan variant has the plan's 9 inputs and 5 outputs
def test_plan_variant_matches_plan_spec(cfg):
    model = SurrogateModel(**variants(cfg)["mlp_plan"])
    assert len(model.input_names) == 9 and model.n_state == 5
    assert model.state_names == ["temperature", "humidity", "vpd", "light_lux", "soil_moisture"]


# Real data: 7 days, actuators off, valid rows have a next state and no saturated soil sensor
def test_real_data_transitions(cfg):
    df = real_data(cfg)
    assert df["episode"].nunique() == 7
    assert (df[CONTROLS] == 0).all().all()
    ok = df[df["valid"]]
    assert ok["next_temperature"].notna().all()
    assert (ok["soil_moisture"] < 99).all() and (ok["next_soil_moisture"] < 99).all()


# Irrigation jumps are never among the valid transitions; actuators stay within 0–100 %
def test_emulator_data_marks_irrigation_invalid(small_data):
    df = pd.concat(small_data.values())
    jump = df["next_soil_moisture"] - df["soil_moisture"]
    assert (jump[df["valid"]] < 1.0).all()
    assert set(df["policy"]) <= {"random", "thermostat", "sweep", "off"}
    assert df[CONTROLS].min().min() >= 0 and df[CONTROLS].max().max() <= 100


# No day appears in two splits
def test_splits_do_not_share_days(small_data):
    days = {s: set(d["episode"]) for s, d in small_data.items()}
    assert not (days["train"] & days["val"]) and not (days["train"] & days["test"]) and not (days["val"] & days["test"])


# History and chain masks on a tiny hand-made table (with a gap between steps 3 and 5)
def test_history_and_chain_masks():
    df = pd.DataFrame({"episode": ["a"] * 6 + ["b"] * 3, "step": [0, 1, 2, 3, 5, 6, 0, 1, 2],
                       "valid": [True] * 9})
    assert history_ok(df, 3).tolist() == [False, False, True, True, False, False, False, False, True]
    assert chain_ok(df, 2).tolist() == [True, True, True, False, True, False, True, True, False]


# Output shapes, save/load gives the same predictions, gradients reach the actuator plan (needed by MPC)
def test_model_shapes_save_load_and_gradients(tmp_path):
    model = SurrogateModel(STATE_EXT, CONTROLS, WEATHER_EXT)
    n_in = len(model.input_names)
    model.set_normalization(np.random.rand(n_in) + 1, np.random.rand(n_in) + 1, np.zeros(6), np.ones(6))
    x = torch.randn(8, n_in)
    assert model(x).shape == (8, 6)

    path = tmp_path / "m.pth"
    model.save(path, {"note": "test"})
    loaded = SurrogateModel.load(path)
    assert torch.allclose(model(x), loaded(x))
    assert loaded.info["note"] == "test"

    controls = (torch.rand(2, 12, 2) * 100).requires_grad_()
    out = loaded.rollout(torch.randn(2, 6), controls, torch.randn(2, 12, 4))
    assert out.shape == (2, 12, 6)
    out[:, :, 0].sum().backward()
    assert controls.grad is not None and torch.isfinite(controls.grad).all()


# A network with all weights zero predicts "nothing changes"
def test_zero_network_predicts_no_change():
    model = SurrogateModel(STATE_PLAN, CONTROLS, WEATHER_PLAN)
    for p in model.net.parameters():
        torch.nn.init.zeros_(p)
    x = torch.rand(4, 9) * 50 + 1  # inside the physical limits
    assert torch.allclose(model(x), x[:, :5])


# Humidity and light never go below zero; temperature has no artificial limit
def test_predictions_respect_physical_limits():
    model = SurrogateModel(STATE_PLAN, CONTROLS, WEATHER_PLAN)
    model.set_normalization(np.zeros(9), np.ones(9), np.full(5, -1000.0), np.ones(5))  # pushes everything down
    out = model(torch.full((3, 9), 10.0))
    names = model.state_names
    assert (out[:, names.index("humidity")] >= 0).all()
    assert (out[:, names.index("light_lux")] >= 0).all()
    assert (out[:, names.index("temperature")] < 0).all()


# A few epochs of training already beat the "nothing changes" guess
def test_short_training_beats_no_change_baseline(cfg, small_data):
    spec = variants(cfg)["mlp"]
    model, log, _ = train(spec, small_data, cfg, epochs=8, verbose=False)
    assert log["val_loss"].iloc[-1] < log["val_loss"].iloc[0]
    X, Y = Samples(small_data["test"], spec).one_step()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy()
    rmse_model = np.sqrt(np.mean((pred[:, 0] - Y[:, 0]) ** 2))
    rmse_persist = np.sqrt(np.mean((X[:, 0] - Y[:, 0]) ** 2))
    assert rmse_model < rmse_persist
