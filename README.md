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
| 2 | Surrogate neural network that predicts the greenhouse 5 minutes ahead | done |
| 3 | PID baseline and the experiment setup | done |
| 4 | NN-MPC controller and a first comparison with PID | planned |

## Setup (Windows)

Python **3.14** (PyTorch has no 3.15 build yet) and Eclipse Mosquitto running on `localhost:1883`.

```powershell
py -V:3.14 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_env.py        # packages, PyTorch, dataset, MQTT broker
python -m pytest                   # 37 tests
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
| Training data for the network (~40 s) | `python scripts/prepare_training_data.py` | `data/processed/` |
| Train and evaluate the network (~3 min) | `python scripts/train_surrogate.py` | `models/`, `results/surrogate/` |
| Tune the PID gains (~1.5 min) | `python scripts/tune_pid.py [--write-config]` | `results/pid/tuning.*` |
| PID on all weather profiles over MQTT (~1 min) | `python scripts/run_pid_baseline.py [--offline]` | `results/pid_baseline.csv`, `results/pid/` |
| PID as its own process (two terminals) | `python scripts/run_controller.py` and `python scripts/run_emulator.py --lockstep --interval 0 --warm` | `results/emulator/run_dataset.*` |

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

## Surrogate network

An MLP (64 → 32, ReLU) predicts the greenhouse state 5 minutes ahead from the current state, the
actuators and the weather ([architecture.py](src/models/architecture.py)). It predicts the change of the
state, and its outputs are kept within physical limits. Chained predictions over 12 steps (1 hour, the
MPC horizon) are differentiable with respect to the actuator plan.

- **Data:** the real Texcoco week (heater and vents off) plus 270 emulator days with randomly varied
  weather and four control styles (random levels, thermostat, fixed-level blocks, off). Steps with
  irrigation are excluded. Split by whole days: 70 % train, 15 % validation, 15 % test.
- **Training:** MSE on the standardised change, Adam (lr 0.001, batch 64), early stopping (patience 20).
- **Variants:** `mlp_plan` with the plan's 9 inputs, and `mlp` with 12 inputs (+ soil temperature,
  outdoor humidity, wind). `mlp` has the lower 1-hour error on the validation days and is saved as
  `models/greenhouse_nn_model.pth`.

Temperature and humidity errors (RMSE) on the emulator test days:

| | T, 5 min | T, 1 h | RH, 1 h |
|---|---|---|---|
| `mlp_plan` (9 inputs) | 0.039 °C | 0.39 °C | 3.17 % |
| **`mlp`** (12 inputs) | **0.025 °C** | **0.27 °C** | **0.80 %** |
| "nothing changes" guess | 0.19 °C | 1.98 °C | 5.35 % |

- The plan's criterion (temperature error 5 min ahead < 0.5 °C) is met.
- Heater and vent check: from the same state, four actions (off, heater, vents, both at 50 %) held for
  1 hour in four situations. In all 12 cases with an actuator on, the network moves the temperature the
  same way as the emulator compared with "off" (e.g. winter night, heater 100 %: emulator +6.2 °C,
  network +5.9 °C). The largest error after 1 hour is 1.4 °C (full heating at a sunny spring noon, rare in the data).
- A 1-hour chained prediction takes about 1–2 ms on the PC.

## PID baseline

A discrete PID runs on every sensor message (every 5 min) with the plan's formula
([controller_pid.py](src/controllers/controller_pid.py)); the integral is limited and does not grow while
the output is at 0 or 100 % (anti-windup).

- **Variants:** `pid_heat` — heater only (the plan's baseline); `pid_heat_vent` — plus a second loop that
  opens the vents above 23 °C (target + 1 °C). While the heater is on, the vents stay closed.
- **Tuning:** grid search on 9 separate tuning days, lowest temperature RMSE among gains where each
  actuator moves on average at most 1 % per step. Heater Kp 160 %/°C, Ki 2 %/(°C·min), Kd 0; vents Kp 30,
  Ki 0.5. The plan's gains (2 / 0.1 / 0.5) were far too weak: 1 °C of error gave only 2 % of heating.
- **Warm start:** before every 24-hour run the day's weather is replayed for 7 days with a simple
  reference thermostat at the target, so every controller starts from the same heated greenhouse.
- **Runs over MQTT** in lockstep give exactly the same result as direct calls (checked by a test).

24 hours per weather profile, 100 m² greenhouse:

| Profile | Variant | Energy | Temperature RMSE | Hours in 21–23 °C |
|---|---|---|---|---|
| winter_cloudy | heater only | 502.2 kWh | 0.13 °C | 24.0 |
| winter_cloudy | heater + vents | 502.2 kWh | 0.13 °C | 24.0 |
| spring_variable | heater only | 174.7 kWh | 5.06 °C | 14.3 |
| spring_variable | heater + vents | 192.1 kWh | 2.10 °C | 16.1 |
| dataset (14 Sep) | heater only | 132.6 kWh | 6.72 °C | 13.1 |
| dataset (14 Sep) | heater + vents | 154.4 kWh | 2.64 °C | 14.7 |
| summer_clear | heater only | 14.0 kWh | 11.60 °C | 6.5 |
| summer_clear | heater + vents | 30.2 kWh | 7.65 °C | 7.8 |

- The temperature never falls below 21 °C on any profile; in winter the PID holds 22 °C within 0.13 °C.
- All remaining error is daytime overheating. Vents cannot cool below the outdoor temperature, so on
  sunny days no controller can reach the plan's ≤ 1 °C RMSE without a cooling actuator.
- Venting costs heating energy as well: the heat vented in the afternoon is missing in the evening.
  This is where a controller that plans ahead can do better.
- Compute time: microseconds per step.

## Limitations

- CO₂ is not modelled; VPD is reported as VPD.
- No cooling: ventilation cannot bring the greenhouse below the outdoor temperature.
- Light and soil moisture are monitored, not controlled (no lamps or irrigation actuator).
- The network mostly learns the emulator: on the real test day its 1-hour temperature error is 2.3 °C
  (the "nothing changes" guess: 2.7 °C), mainly because of the emulator's own error against the real greenhouse.
- The emulator's sensors have no noise. This favours the aggressive PID gain; with noisy sensors a real
  installation would need lower gains or filtering.

## Roadmap

After part 4:
- Full PID vs NN-MPC comparison on more days, energy at equal comfort, prediction horizon study, final report.
- Comfort band (e.g. 21–23 °C) instead of a fixed set-point; other networks (LSTM, wider MLP).
- Swappable greenhouse model: one interface for this emulator and an established open model (e.g. GreenLight).
- Weather profiles from real weather in Kyrgyzstan (OpenMeteo history: Bishkek, Osh, Naryn).
- Day/night electricity prices, sensor noise and actuator delays, humidity/VPD in the control goal.
- Measure the controller on a Raspberry Pi 4.
