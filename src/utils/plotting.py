"""Standard figure of one emulator run."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# Figure of one run: temperature, humidity, VPD, light, soil moisture, actuators
def plot_run(log: pd.DataFrame, path: Path, title: str, target: dict | None = None):
    t = pd.to_datetime(log["time"])
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(t, log["temperature"], color="#c53030", label="indoor")
    axes[0].plot(t, log["outdoor_temp"], color="#2f855a", ls="--", lw=1, label="outdoor")
    if target:
        axes[0].axhline(target["temperature"], color="#2d3748", ls=":", lw=1,
                        label=f"target {target['temperature']} °C")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)

    axes[1].plot(t, log["humidity"], color="#2b6cb0", label="indoor")
    axes[1].plot(t, log["outdoor_humidity"], color="#2f855a", ls="--", lw=1, label="outdoor")
    axes[1].set_ylabel("Humidity (%RH)")
    axes[1].legend(loc="upper left", ncol=2, fontsize=8)

    axes[2].plot(t, log["vpd"], color="#6b46c1")
    if target:
        axes[2].axhline(target["vpd"], color="#2d3748", ls=":", lw=1)
    axes[2].set_ylabel("VPD (kPa)")

    axes[3].plot(t, log["light_ppfd"], color="#d69e2e")
    axes[3].set_ylabel("PPFD (µmol/m²/s)")

    axes[4].plot(t, log["soil_moisture"], color="#8b5a2b")
    axes[4].set_ylabel("Soil moisture (%)")

    axes[5].step(t, log["heater_power"], where="post", color="#c53030", label="heater")
    axes[5].step(t, log["ventilation_rate"], where="post", color="#2b6cb0", label="ventilation")
    axes[5].set_ylabel("Actuators (%)")
    axes[5].set_ylim(-5, 105)
    axes[5].legend(loc="upper left", ncol=2, fontsize=8)
    axes[5].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
