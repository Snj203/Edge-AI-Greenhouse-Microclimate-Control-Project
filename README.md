# Edge-AI Greenhouse Microclimate Control

Neural-network model predictive control (NN-MPC) compared with PID control for a greenhouse.
The controllers are tested on a physics-based emulator calibrated to one week of real data from a
greenhouse in Texcoco, Mexico. Everything runs on a PC: the emulator, the controllers and an MQTT
broker (Eclipse Mosquitto) between them, as in a real sensor → controller → actuator setup.

**Research question:** how does NN-MPC compare to PID in energy use and climate stability?

## Status

| Part | Content | Status |
|---|---|---|
| 1 | Environment and MQTT, dataset analysis, greenhouse emulator and its calibration | done |
| 2 | Surrogate neural network that predicts the greenhouse 5 minutes ahead | planned |
| 3 | PID baseline and the experiment setup | planned |
| 4 | NN-MPC controller and a first comparison with PID | planned |

## Setup (Windows)

Python **3.14** (PyTorch has no 3.15 build yet) and Eclipse Mosquitto running on `localhost:1883`.

```powershell
py -V:3.14 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_env.py        # packages, PyTorch, dataset, MQTT broker
python -m pytest                   # 18 tests
```

Run all commands from the project root.

## Commands

| What | Command | Output |
|---|---|---|
| Dataset analysis | `python scripts/eda_dataset.py` | `results/eda/` |
| Calibrate the emulator to the dataset (~1 min) | `python scripts/calibrate_emulator.py [--write-config]` | `results/calibration/` |
| Emulator without control, all weather profiles | `python scripts/plot_open_loop.py` | `results/emulator/` |
| Emulator over MQTT | `python scripts/run_emulator.py --profile winter_cloudy` | `results/emulator/run_*.csv` |
| Closed-loop demo (two terminals) | `python scripts/demo_controller.py` and `python scripts/run_emulator.py --lockstep --interval 0` | `results/emulator/run_dataset.*` |

Watch the MQTT traffic: `mosquitto_sub -t "greenhouse/#" -v`

## Dataset

Texcoco greenhouse, 11–17 September 2025: 1926 records every 5 minutes, indoor sensors plus outdoor
weather from OpenMeteo. Findings that shaped the project ([results/eda/eda_summary.md](results/eda/eda_summary.md)):

- The first 137 rows were recorded before the sensors were installed in the greenhouse and are excluded.
- VPD and lux are calculated from other sensors (temperature/humidity and PPFD), not measured separately.
- Irrigation happened 9 times, always between 09:00 and 16:00, from 42–54 % up to about 60–66 % soil moisture.
- The real greenhouse has no heating or ventilation records: indoor temperature is below 20 °C 55 % of the
  time and above 24 °C 32 % of the time.

## Emulator

A simplified physical model per m² of floor ([greenhouse_emulator.py](src/emulator/greenhouse_emulator.py)):

- **States:** air temperature, water vapour, floor heat store, soil temperature, soil moisture.
- **Actuators:** heater (300 W/m² at 100 %) and ventilation, 0–100 %. Energy is counted in kWh for a 100 m² greenhouse.
- **Weather:** `dataset` (a real day, 14 Sep), `summer_clear`, `winter_cloudy`, `spring_variable`.
- **Irrigation:** automatic, like in the real greenhouse.
- **Calibration:** heater and vents off, driven by the real outdoor weather of the whole week:

  | Variable | RMSE | R² |
  |---|---|---|
  | Air temperature | 2.14 °C | 0.90 |
  | Air humidity | 4.97 %RH | 0.90 |
  | VPD | 0.42 kPa | 0.84 |
  | Soil temperature | 0.36 °C | 0.94 |
  | Soil moisture | 2.74 % | 0.70 |

MQTT topics: `greenhouse/sensors/data` (emulator → controller), `greenhouse/actuators/control`
(controller → emulator), `greenhouse/emulator/status` (started / finished). In lockstep mode the emulator
waits for the controller's reply at every step, so runs are reproducible.

## Limitations

- CO₂ is not modelled; VPD is reported as VPD.
- No cooling: ventilation cannot bring the greenhouse below the outdoor temperature.
- Light and soil moisture are monitored, not controlled (no lamps or irrigation actuator).

## Roadmap

After part 4:
- Full PID vs NN-MPC comparison on more days, energy at equal comfort, prediction horizon study, final report.
- Comfort band (e.g. 21–23 °C) instead of a fixed set-point; other network types (LSTM).
- Swappable greenhouse model: one interface for this emulator and an established open model (e.g. GreenLight).
- Weather profiles from real weather in Kyrgyzstan (OpenMeteo history: Bishkek, Osh, Naryn).
- Day/night electricity prices, sensor noise and actuator delays, humidity/VPD in the control goal.
- Measure the controller on a Raspberry Pi 4.
