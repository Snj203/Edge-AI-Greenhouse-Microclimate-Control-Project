# EDA — Texcoco greenhouse dataset

- Rows × columns: **1926 × 33**, period 2025-09-11 06:37 → 2025-09-17 23:56
- Missing values: **6** in `outside_openmeteo_delta_temp_inside_outside` (not used by the project)
- Time step: median **5.0 min**, mean 5.03, max 40
  - gap of 40 min ending at 2025-09-11 13:26
- **Setup period excluded:** 137 rows before 2025-09-11 19:00 — the station was not yet in the greenhouse (≈32 °C flat, PPFD ≈ 0 while outdoor sun was 400–900 W/m², soil moisture 0 %). Remaining: **1789 rows**

## Statistics (setup period excluded)

| column | mean | std | min | max |
|---|---|---|---|---|
| temp_aire_media | 21.86 | 6.65 | 13.6 | 40.4 |
| hum_aire_media | 71.35 | 15.48 | 31.7 | 92.6 |
| vpd_media | 1.01 | 1.05 | 0.15 | 4.69 |
| ppfd | 48.74 | 80.63 | 0.0 | 341.76 |
| hum_suelo | 49.25 | 7.29 | 37.3 | 100.0 |
| temp_suelo | 18.28 | 1.47 | 15.8 | 21.9 |
| outside_openmeteo_temperature_2m | 15.64 | 3.36 | 10.8 | 22.8 |
| outside_openmeteo_relative_humidity_2m | 83.13 | 14.43 | 46.0 | 100.0 |
| outside_openmeteo_shortwave_radiation | 223.15 | 297.46 | 0.0 | 923.0 |
| outside_openmeteo_wind_speed_10m | 5.4 | 4.39 | 0.0 | 18.1 |

## Data quality

- `vpd_media` = Tetens(T)·(1−RH/100): max error 5.0e-10 kPa → VPD is derived from T and RH, not a separate sensor
- `lux / ppfd` = 54.05 (std 6.5e-15) → lux is derived from PPFD (factor 54.05)
- Soil-moisture saturation spikes (≥ 99 %): **13 rows** (5–15 min after each irrigation) → excluded via `soil_moisture_valid`
- Irrigation events: **9** — 09-12 09:21, 09-12 14:41, 09-13 11:01, 09-14 14:06, 09-15 13:16, 09-15 15:51, 09-16 13:01, 09-16 15:26, 09-17 15:46
  - soil moisture 42–54 % before → 60–79 % 15 min after; always between 9:00 and 16:00
- Outdoor OpenMeteo values are **hourly** (constant within each hour) → interpolated for the emulator
- Indoor temperature vs target 22.0 °C: 55% of time below 20 °C, 32% above 24 °C → the greenhouse needs heating at night and cooling (ventilation) by day
- At night the greenhouse is on average **4.0 °C warmer** than outside

Plots: correlation.png, daily_profile.png, distributions.png, timeseries_indoor.png, timeseries_outdoor.png
