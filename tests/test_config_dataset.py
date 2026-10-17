import numpy as np

from src.emulator.greenhouse_emulator import PhysicsParams, saturation_vapour_pressure
from src.utils.config import load_config, resolve_path
from src.utils.dataset import SETUP_END, greenhouse_data, load_dataset, smooth_outdoor


# Config loads, the dataset file exists, all physics parameters are present
def test_config_loads():
    cfg = load_config()
    assert cfg["emulator"]["timestep_minutes"] == 5
    assert resolve_path(cfg["dataset"]["path"]).exists()
    PhysicsParams.from_config(cfg)  # every physics parameter is present and numeric


# Dataset size, time order, setup-period flag, 9 irrigations
def test_dataset_shape_and_flags():
    df = load_dataset()
    assert len(df) == 1926
    assert len(df.attrs["raw_columns"]) == 33
    assert df["time"].is_monotonic_increasing
    assert df.loc[df["time"] < SETUP_END, "setup_period"].all()
    assert df["irrigation_event"].sum() == 9


# VPD in the data equals the formula from temperature and humidity
def test_vpd_is_derived_from_t_and_rh():
    df = greenhouse_data()
    vpd = df["temp_aire_media"].map(saturation_vapour_pressure) * (1 - df["hum_aire_media"] / 100)
    assert np.allclose(vpd, df["vpd_media"], atol=1e-6)


# Smoothed outdoor weather: no gaps, no hourly steps
def test_outdoor_interpolation_is_smooth_and_complete():
    w = smooth_outdoor(greenhouse_data())
    assert not w.isna().any().any()
    assert w["solar_radiation"].max() > 800  # the sun is really there (catches time-unit bugs)
    assert w["outdoor_temp"].diff().abs().max() < 1.0  # no hourly steps left
