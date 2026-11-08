import numpy as np
import pytest
import torch

from src.controllers.controller_nn_mpc import BIAS_LIMIT, NNMPCController
from src.controllers.experiment import prepare_run, run_offline
from src.emulator.simulation import forecast_at
from src.emulator.weather import build_weather
from src.models.architecture import SurrogateModel
from src.models.casadi_surrogate import casadi_step
from src.utils.config import load_config, resolve_path
from src.utils.metrics import control_metrics

NIGHT = {"outdoor_temp": 5.0, "outdoor_humidity": 85.0, "solar_radiation": 0.0, "wind_speed": 8.0}
HOT_NOON = {"outdoor_temp": 28.0, "outdoor_humidity": 40.0, "solar_radiation": 900.0, "wind_speed": 5.0}


# Config shared by the tests in this file
@pytest.fixture(scope="module")
def cfg():
    return load_config()


# The trained network
@pytest.fixture(scope="module")
def model(cfg):
    return SurrogateModel.load(resolve_path(cfg["nn"]["model_path"]))


# A sensor message with the given temperature and weather
def obs_for(temperature, weather, step=0, **extra):
    return {"step": step, "temperature": temperature, "humidity": 70.0, "vpd": 0.8,
            "light_lux": 54.05 * 0.22 * weather["solar_radiation"], "soil_moisture": 50.0, "soil_temperature": 18.0,
            **weather, **extra}


# The network rebuilt in CasADi gives the same outputs as PyTorch, also at the physical limits
def test_casadi_network_equals_pytorch(model):
    step = casadi_step(model)
    rng = np.random.default_rng(0)
    x = model.in_mean.numpy() + rng.normal(0, 1.5, (50, len(model.input_names))) * model.in_std.numpy()
    x[:5, model.state_names.index("humidity")] = 99.9
    with torch.no_grad():
        expected = model(torch.tensor(x, dtype=torch.float32)).numpy()
    got = np.stack([np.asarray(step(row)).ravel() for row in x])
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-3)


# The cost J equals the formula computed with the PyTorch network; its gradient equals finite differences
def test_cost_matches_formula_and_gradient_is_exact(cfg, model):
    mpc = NNMPCController.from_config(cfg, move_weight=10.0)
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 2 * mpc.N)
    s0 = np.array([20.5, 70.0, 0.8, 5000.0, 50.0, 18.0])
    w = np.tile([200.0, 9.0, 70.0, 6.0], (mpc.N, 1))
    u_prev = np.array([0.4, 0.1])
    J, g = mpc.cost(x, s0, w.ravel(), 0.0, u_prev)

    u = x.reshape(mpc.N, 2)
    with torch.no_grad():
        states = model.rollout(torch.tensor(s0[None], dtype=torch.float32),
                               torch.tensor((u * mpc.u_max)[None], dtype=torch.float32),
                               torch.tensor(w[None], dtype=torch.float32))[0].numpy()
    moves = np.sum(np.diff(np.vstack([u_prev, u]), axis=0) ** 2)
    expected = mpc.alpha * np.sum(u @ mpc.kwh) + mpc.beta * np.sum((states[:, 0] - mpc.target) ** 2) + mpc.gamma * moves
    assert J == pytest.approx(expected, rel=1e-4)

    eps = 1e-6
    fd = [(mpc.cost(x + eps * e, s0, w.ravel(), 0.0, u_prev)[0] - J) / eps for e in np.eye(len(x))[:6]]
    assert np.allclose(g[:6], fd, rtol=1e-3, atol=1e-3)


# Cold night: heats, does not vent. Hot noon: vents, does not heat. Within 500 ms, within 0–100 %
def test_heats_when_cold_and_vents_when_hot(cfg):
    mpc = NNMPCController.from_config(cfg)
    heater, vent = mpc(obs_for(17.0, NIGHT))
    assert heater > 50 and vent < 1
    mpc.reset()
    heater, vent = mpc(obs_for(32.0, HOT_NOON))
    assert heater == pytest.approx(0, abs=1e-6) and vent > 50
    assert mpc.last_info["mpc_converged"] and mpc.last_latency_ms < 500
    for u in (heater, vent):
        assert 0.0 <= u <= 100.0


# The forecast is taken from the message; without one the current weather is repeated
def test_forecast_from_message_or_current_weather(cfg):
    mpc = NNMPCController.from_config(cfg)
    persist = mpc.forecast(obs_for(20.0, NIGHT))
    assert persist.shape == (mpc.N, 4) and (persist == persist[0]).all()
    short = mpc.forecast(obs_for(20.0, NIGHT, forecast=[NIGHT, HOT_NOON]))
    assert short[1].tolist() == short[-1].tolist() == [HOT_NOON[n] for n in mpc.weather_names]


# forecast_at takes the profile's weather and repeats the last row past the end
def test_forecast_at_matches_weather_and_pads(cfg):
    weather = build_weather(cfg, "spring_variable", hours=2)
    f = forecast_at(weather, len(weather) - 3, 12)
    assert len(f) == 12 and f[2] == f[11]
    assert f[0]["outdoor_temp"] == round(weather["outdoor_temp"].iloc[-3], 2)


# The bias learns a steady model error (0.1 °C per step) and stays within its limit for wild readings
def test_bias_learns_a_steady_model_error(cfg):
    mpc = NNMPCController.from_config(cfg)
    obs = obs_for(20.0, NIGHT)
    for k in range(30):
        mpc(obs)
        true_next = mpc.predict_next(mpc._last_input)[0] + 0.1  # the "greenhouse" warms 0.1 °C/step more
        obs = obs_for(true_next, NIGHT, step=k + 1)
    mpc(obs)
    assert mpc.bias == pytest.approx(0.1, abs=1e-3)
    for k in range(10):
        mpc(obs_for(40.0, NIGHT, step=31 + k))
    assert mpc.bias == BIAS_LIMIT


# 3 winter hours: holds 22 °C, logs its diagnostics, every step within 500 ms
def test_short_winter_run_holds_the_target(cfg):
    weather, emu = prepare_run(cfg, "winter_cloudy", hours=3)
    log = run_offline(NNMPCController.from_config(cfg), emu, weather)
    m = control_metrics(log, cfg)
    assert m["temp_rmse_c"] < 0.6 and m["hours_in_band"] >= 2.5
    assert {"mpc_iterations", "mpc_bias", "mpc_converged"} <= set(log.columns)
    assert m["latency_max_ms"] < 500
