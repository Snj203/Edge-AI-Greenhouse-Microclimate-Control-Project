"""
Exploratory analysis of the Texcoco greenhouse dataset: data quality, statistics and plots.

    python scripts/eda_dataset.py      -> results/eda/ (plots and eda_summary.md)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.emulator.greenhouse_emulator import LUX_PER_PPFD, saturation_vapour_pressure
from src.utils.config import ensure_dir, load_config
from src.utils.dataset import SETUP_END, load_dataset

INDOOR = [
    ("temp_aire_media", "Air temperature", "°C"),
    ("hum_aire_media", "Air humidity", "%RH"),
    ("vpd_media", "VPD", "kPa"),
    ("ppfd", "PPFD", "µmol/m²/s"),
    ("hum_suelo", "Soil moisture", "%"),
    ("temp_suelo", "Soil temperature", "°C"),
]
OUTDOOR = [
    ("outside_openmeteo_temperature_2m", "Outdoor temperature", "°C"),
    ("outside_openmeteo_relative_humidity_2m", "Outdoor humidity", "%RH"),
    ("outside_openmeteo_shortwave_radiation", "Solar radiation", "W/m²"),
    ("outside_openmeteo_wind_speed_10m", "Wind speed", "km/h"),
]


# Text summary: size, gaps, setup period, statistics, data-quality findings
def summary(df, cfg) -> list[str]:
    lines = ["# EDA — Texcoco greenhouse dataset\n"]
    raw = df.attrs["raw_columns"]
    lines.append(f"- Rows × columns: **{len(df)} × {len(raw)}**, period "
                 f"{df['time'].min():%Y-%m-%d %H:%M} → {df['time'].max():%Y-%m-%d %H:%M}")
    missing = df[raw].isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("- Missing values: **none**")
    for col, n in missing.items():
        lines.append(f"- Missing values: **{n}** in `{col}` (not used by the project)")

    dt = df["dt_min"].dropna()
    lines.append(f"- Time step: median **{dt.median():.1f} min**, mean {dt.mean():.2f}, max {dt.max():.0f}")
    for _, gap in df.loc[df["dt_min"] > 2 * dt.median()].iterrows():
        lines.append(f"  - gap of {gap['dt_min']:.0f} min ending at {gap['time']:%Y-%m-%d %H:%M}")

    g = df[~df["setup_period"]]
    lines.append(f"- **Setup period excluded:** {int(df['setup_period'].sum())} rows before "
                 f"{SETUP_END:%Y-%m-%d %H:%M} — the station was not yet in the greenhouse (≈32 °C flat, "
                 "PPFD ≈ 0 while outdoor sun was 400–900 W/m², soil moisture 0 %). "
                 f"Remaining: **{len(g)} rows**")

    lines.append("\n## Statistics (setup period excluded)\n")
    cols = [c for c, _, _ in INDOOR + OUTDOOR]
    stats = g[cols].describe().T[["mean", "std", "min", "max"]].round(2)
    lines.append("| column | mean | std | min | max |\n|---|---|---|---|---|")
    for c, r in stats.iterrows():
        lines.append(f"| {c} | {r['mean']} | {r['std']} | {r['min']} | {r['max']} |")

    lines.append("\n## Data quality\n")
    vpd = df["temp_aire_media"].map(saturation_vapour_pressure) * (1 - df["hum_aire_media"] / 100)
    lines.append(f"- `vpd_media` = Tetens(T)·(1−RH/100): max error {np.abs(vpd - df['vpd_media']).max():.1e} kPa "
                 "→ VPD is derived from T and RH, not a separate sensor")
    ratio = (df["lux"] / df["ppfd"]).where(df["ppfd"] > 1)
    lines.append(f"- `lux / ppfd` = {ratio.median():.2f} (std {ratio.std():.1e}) → lux is derived from PPFD "
                 f"(factor {LUX_PER_PPFD})")
    lines.append(f"- Soil-moisture saturation spikes (≥ 99 %): **{int(g['soil_saturated'].sum())} rows** "
                 "(5–15 min after each irrigation) → excluded via `soil_moisture_valid`")
    events = g[g["irrigation_event"]]
    lines.append(f"- Irrigation events: **{len(events)}** — " + ", ".join(f"{t:%m-%d %H:%M}" for t in events["time"]))
    before = df["hum_suelo"].shift(1)[df["irrigation_event"]]
    after = df["hum_suelo"].shift(-3)[df["irrigation_event"]]
    lines.append(f"  - soil moisture {before.min():.0f}–{before.max():.0f} % before → "
                 f"{after.min():.0f}–{after.max():.0f} % 15 min after; always between "
                 f"{events['time'].dt.hour.min()}:00 and {events['time'].dt.hour.max() + 1}:00")
    lines.append("- Outdoor OpenMeteo values are **hourly** (constant within each hour) → interpolated for the emulator")
    t = g["temp_aire_media"]
    target = cfg["target"]["temperature"]
    lines.append(f"- Indoor temperature vs target {target} °C: {np.mean(t < target - 2):.0%} of time below "
                 f"{target - 2:.0f} °C, {np.mean(t > target + 2):.0%} above {target + 2:.0f} °C "
                 "→ the greenhouse needs heating at night and cooling (ventilation) by day")
    night = g["outside_openmeteo_shortwave_radiation"] == 0
    delta = (t - g["outside_openmeteo_temperature_2m"])[night]
    lines.append(f"- At night the greenhouse is on average **{delta.mean():.1f} °C warmer** than outside")
    return lines


# Save and close a figure
def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# The 5 EDA figures
def plots(df, cfg, out):
    sns.set_theme(style="whitegrid", font_scale=0.9)
    g = df[~df["setup_period"]]
    target = cfg["target"]["temperature"]

    fig, axes = plt.subplots(len(INDOOR), 1, figsize=(14, 15), sharex=True)
    for ax, (col, label, unit) in zip(axes, INDOOR):
        ax.plot(df["time"], df[col], lw=0.7, color="#2b6cb0")
        ax.set_ylabel(f"{label}\n({unit})")
        ax.axvspan(df["time"].min(), SETUP_END, color="#a0aec0", alpha=0.35, lw=0)
    axes[1].text(df["time"].min(), axes[1].get_ylim()[1], " setup period (excluded)", va="top", fontsize=8)
    bad = g["soil_saturated"]
    axes[4].scatter(g.loc[bad, "time"], g.loc[bad, "hum_suelo"], s=10, color="#c53030",
                    label="saturation spike (excluded)", zorder=3)
    axes[4].legend(loc="lower right")
    axes[0].axhline(target, color="#c53030", ls="--", lw=1, label=f"target {target} °C")
    axes[0].legend(loc="upper right")
    fig.suptitle("Indoor sensors — 1 week", fontweight="bold")
    save(fig, out / "timeseries_indoor.png")

    fig, axes = plt.subplots(len(OUTDOOR), 1, figsize=(14, 10), sharex=True)
    for ax, (col, label, unit) in zip(axes, OUTDOOR):
        ax.plot(df["time"], df[col], lw=0.8, color="#2f855a")
        ax.set_ylabel(f"{label}\n({unit})")
    axes[0].plot(df["time"], df["temp_aire_media"], lw=0.8, color="#c53030", label="indoor")
    axes[0].legend(loc="upper right")
    fig.suptitle("Outdoor weather (OpenMeteo) — what drives the emulator", fontweight="bold")
    save(fig, out / "timeseries_outdoor.png")

    hourly = g.assign(hour=g["time"].dt.hour).groupby("hour")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, label, unit) in zip(axes.flat, INDOOR):
        mean, std = hourly[col].mean(), hourly[col].std()
        ax.plot(mean.index, mean, color="#2b6cb0")
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color="#2b6cb0")
        ax.set_title(f"{label} ({unit})")
        ax.set_xlabel("hour of day")
    fig.suptitle("Average day (mean ± std)", fontweight="bold")
    save(fig, out / "daily_profile.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, label, unit) in zip(axes.flat, INDOOR):
        data = g.loc[g["soil_moisture_valid"], col] if col == "hum_suelo" else g[col]
        ax.hist(data, bins=40, color="#2b6cb0", edgecolor="white")
        ax.set_title(f"{label} ({unit})" + (" — valid only" if col == "hum_suelo" else ""))
    fig.suptitle("Distributions", fontweight="bold")
    save(fig, out / "distributions.png")

    cols = [c for c, _, _ in INDOOR + OUTDOOR]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(g[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation", fontweight="bold")
    save(fig, out / "correlation.png")


# Load the dataset, write eda_summary.md and the figures
def main():
    cfg = load_config()
    out = ensure_dir("results/eda")
    df = load_dataset(cfg)
    lines = summary(df, cfg)
    plots(df, cfg, out)
    lines.append(f"\nPlots: {', '.join(sorted(p.name for p in out.glob('*.png')))}")
    (out / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
