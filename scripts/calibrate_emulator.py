"""
Fit the emulator's physical parameters to the Texcoco week.

The emulator is driven by the real outdoor weather with heater and vents off (like the real
greenhouse), and the parameters are chosen so that the simulated indoor climate follows the
measured one (least squares).

    python scripts/calibrate_emulator.py                 # fit, report, plot (~1 min)
    python scripts/calibrate_emulator.py --write-config  # also save the values in configs/default.yaml
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from src.emulator.greenhouse_emulator import GreenhouseEmulator, PhysicsParams
from src.utils.config import DEFAULT_CONFIG_PATH, ensure_dir, load_config
from src.utils.dataset import greenhouse_data, smooth_outdoor

# parameter -> (lower, upper) bound
FITTED = {
    "thermal_capacity": (5e3, 3e5),
    "heat_loss_coeff": (0.5, 20.0),
    "solar_gain_factor": (0.02, 0.9),
    "floor_capacity": (2e4, 3e6),
    "floor_coupling": (0.5, 40.0),
    "floor_ground_coupling": (0.0, 20.0),
    "ground_temp": (10.0, 26.0),
    "leak_ach": (0.1, 20.0),
    "wind_ach": (0.0, 2.0),
    "stack_ach": (0.0, 10.0),
    "transp_rad_coeff": (0.0, 3e-4),
    "transp_vpd_coeff": (0.0, 0.1),
    "substrate_capacity": (1e4, 5e6),
    "substrate_coupling": (0.05, 30.0),
    "substrate_ground_coupling": (0.0, 30.0),
}
FITTED_SOIL = {  # fitted separately: they only affect soil moisture
    "soil_drain_rate": (0.0, 1.0),
    "field_capacity": (40.0, 75.0),
    "excess_drain_rate": (0.01, 5.0),
    "soil_water_capacity_mm": (1.0, 300.0),
}
# error that counts as "1" for each variable, so that °C and %RH are weighted fairly
SCALE = {"temperature": 1.0, "humidity": 4.0, "soil_temperature": 1.0, "soil_moisture": 1.0}
OUTPUTS = ("temperature", "humidity", "vpd", "soil_temperature", "soil_moisture", "light_ppfd", "transpiration")
MEASURED = {"temperature": "temp_aire_media", "humidity": "hum_aire_media", "vpd": "vpd_media",
            "soil_temperature": "temp_suelo", "soil_moisture": "hum_suelo", "light_ppfd": "ppfd"}


# Run the emulator through the dataset week with heater and vents off.
# The floor temperature (not measured) is warmed up by simulating the first day once.
# Real irrigations are replayed: soil moisture is set to the first valid reading after each one.
def simulate(params: PhysicsParams, data, weather) -> dict[str, np.ndarray]:
    w_rows = weather[["outdoor_temp", "outdoor_rh", "solar_radiation", "wind_speed"]].to_dict("records")
    dts = data["dt_min"].fillna(5.0).to_numpy()
    sm = data["hum_suelo"].to_numpy()
    valid_sm = data["soil_moisture_valid"].to_numpy()
    events = data["irrigation_event"].to_numpy()
    first = data.iloc[0]
    # the soil sensor is still recovering at the start, so take its value 30 min in
    sm0 = float(sm[np.argmax((data["time"] >= first["time"] + np.timedelta64(30, "m")).to_numpy())])

    emu = GreenhouseEmulator(params, irrigation=False)

    # start from the first measured state
    def start(floor_temperature):
        emu.reset(first["temp_aire_media"], first["hum_aire_media"], first["temp_suelo"], sm0,
                  first["time"].to_pydatetime(), floor_temperature=floor_temperature)

    # one step to row i
    def advance(i):
        w = {k: 0.5 * (w_rows[i - 1][k] + w_rows[i][k]) for k in w_rows[i]}  # mean over the interval
        emu.step(w, 0.0, 0.0, dt_minutes=dts[i])

    start(first["temp_suelo"])
    day = int(np.argmax((data["time"] >= first["time"] + np.timedelta64(24, "h")).to_numpy()))
    for i in range(1, day + 1):
        advance(i)
    start(emu.Tf)

    n = len(data)
    out = {k: np.empty(n) for k in OUTPUTS}
    pending_reset = False
    for i in range(n):
        if i > 0:
            advance(i)
            if pending_reset and valid_sm[i]:
                emu.SM = float(sm[i])
                pending_reset = False
            pending_reset |= bool(events[i])
        obs = emu.observe(w_rows[i])
        for k in OUTPUTS:
            out[k][i] = emu.last_transpiration if k == "transpiration" else obs[k]
    return out


# Least-squares fit of the parameters in `bounds` so that the simulated `targets` match the data.
# The optimiser works on a 0..1 scale per parameter (log scale for those spanning orders of magnitude).
def fit(base: PhysicsParams, data, weather, bounds: dict, targets: list[str], label: str) -> PhysicsParams:
    names = list(bounds)
    lo = np.array([bounds[k][0] for k in names], float)
    hi = np.array([bounds[k][1] for k in names], float)
    log = (lo > 0) & (hi / np.where(lo > 0, lo, 1) > 50)
    a = np.where(log, np.log(np.where(lo > 0, lo, 1)), lo)
    b = np.where(log, np.log(hi), hi)

    # parameter values -> 0..1 scale
    def to_unit(x):
        return (np.where(log, np.log(np.maximum(x, 1e-300)), x) - a) / (b - a)

    # 0..1 scale -> parameters
    def to_params(z):
        v = a + np.asarray(z) * (b - a)
        return base.replace(**dict(zip(names, np.where(log, np.exp(v), v).tolist())))

    masks = {k: (data["soil_moisture_valid"] if k == "soil_moisture" else data[MEASURED[k]].notna()).to_numpy()
             for k in targets}

    # errors (simulated − measured) that least squares makes small
    def residuals(z):
        sim = simulate(to_params(z), data, weather)
        return np.concatenate([((sim[k] - data[MEASURED[k]].to_numpy()) / SCALE[k])[masks[k]] for k in targets])

    z0 = np.clip(to_unit(np.array([getattr(base, k) for k in names], float)), 0, 1)
    t0 = time.time()
    # diff_step 1e-3: the model has clamps (RH <= 100 %, sqrt), so very small steps give noisy gradients
    result = least_squares(residuals, z0, bounds=(0, 1), diff_step=1e-3, max_nfev=150)
    print(f"  {label} fit: {result.nfev} iterations, {time.time() - t0:.0f}s, cost {result.cost:.1f}")
    at_bound = [k for k, z in zip(names, result.x) if z < 1e-3 or z > 1 - 1e-3]
    if at_bound:
        print(f"    at a bound: {at_bound}")
    return to_params(result.x)


# Light transmission: PPFD = τ × solar radiation (linear fit)
def fit_light(params: PhysicsParams, data, weather) -> PhysicsParams:
    sun = weather["solar_radiation"].to_numpy()
    ppfd = data["ppfd"].to_numpy()
    tau = float(np.sum(ppfd * sun) / np.sum(sun * sun))
    print(f"  light: PPFD = {tau:.4f} × solar radiation")
    return params.replace(light_transmission=tau)


# Print and return RMSE, bias and R² for each variable
def report(sim, data) -> dict:
    metrics = {}
    print("\n  variable            RMSE     bias    R²")
    for k, col in MEASURED.items():
        y = data[col].to_numpy()
        mask = data["soil_moisture_valid"].to_numpy() if k == "soil_moisture" else np.ones(len(y), bool)
        err = sim[k][mask] - y[mask]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        r2 = float(1 - np.sum(err ** 2) / np.sum((y[mask] - y[mask].mean()) ** 2))
        metrics[k] = {"rmse": round(rmse, 3), "bias": round(float(err.mean()), 3), "r2": round(r2, 3)}
        print(f"  {k:<17} {rmse:7.3f} {err.mean():8.3f} {r2:6.2f}")
    return metrics


# Measured vs simulated for the whole week
def plot(sim, data, weather, path: Path):
    rows = [("temperature", "Air temperature (°C)"), ("humidity", "Air humidity (%RH)"), ("vpd", "VPD (kPa)"),
            ("light_ppfd", "PPFD (µmol/m²/s)"), ("soil_moisture", "Soil moisture (%)"),
            ("soil_temperature", "Soil temperature (°C)")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 16), sharex=True)
    t = data["time"]
    for ax, (k, label) in zip(axes, rows):
        col = MEASURED[k]
        y = data[col].where(data["soil_moisture_valid"]) if k == "soil_moisture" else data[col]
        ax.plot(t, y, color="#718096", lw=0.8, label="measured")
        ax.plot(t, sim[k], color="#c53030", lw=1.0, label="emulator")
        ax.set_ylabel(label)
    axes[0].plot(t, weather["outdoor_temp"], color="#2f855a", lw=0.8, ls="--", label="outdoor")
    axes[0].legend(loc="upper right", ncol=3)
    fig.suptitle("Emulator calibration — open loop, real outdoor weather (Texcoco, Sep 2025)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Replace the values in the `emulator.physics` block of the YAML file, keeping the comments
def write_config(params: PhysicsParams, path: Path = DEFAULT_CONFIG_PATH):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    values = {k: (f"{v:.6g}" if isinstance(v, float) else str(v)) for k, v in params.to_dict().items()}
    inside, done = False, set()
    for i, line in enumerate(lines):
        if line.startswith("  physics:"):
            inside = True
            continue
        if inside and line.strip() and not line.startswith("    "):
            break
        m = re.match(r"^    (\w+):[^#]*(#.*)?$", line.rstrip("\n"))
        if inside and m and m.group(1) in values:
            key, comment = m.group(1), m.group(2)
            text = f"    {key}: {values[key]}"
            lines[i] = (text.ljust(36) + " " + comment if comment else text) + "\n"
            done.add(key)
    missing = set(values) - done
    if missing:
        raise ValueError(f"missing from emulator.physics in the config: {sorted(missing)}")
    path.write_text("".join(lines), encoding="utf-8")
    print(f"  wrote the calibrated values to {path}")


# Fit heat/humidity, then soil, then light; save the report and plot (and the config if asked)
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-config", action="store_true", help="save the fitted values in configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config()
    out = ensure_dir("results/calibration")
    data = greenhouse_data(cfg)
    weather = smooth_outdoor(data)
    print(f"Calibrating on {len(data)} samples")

    params = PhysicsParams.from_config(cfg)
    params = fit(params, data, weather, FITTED, ["temperature", "humidity", "soil_temperature"],
                 "temperature/humidity")
    params = fit(params, data, weather, FITTED_SOIL, ["soil_moisture"], "soil moisture")
    params = fit_light(params, data, weather)
    sim = simulate(params, data, weather)

    metrics = report(sim, data)
    plot(sim, data, weather, out / "calibration_fit.png")
    (out / "calibration.json").write_text(json.dumps({"params": params.to_dict(), "metrics": metrics}, indent=2))
    print(f"\n  saved {out / 'calibration_fit.png'} and calibration.json")
    if args.write_config:
        write_config(params)


if __name__ == "__main__":
    main()
