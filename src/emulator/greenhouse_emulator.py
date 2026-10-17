"""
Physical model of the greenhouse (energy and water balances), per m² of floor.

States: air temperature T, water vapour C (g/m³), floor temperature Tf (heat store, not measured),
soil temperature Ts, soil moisture SM (%). Inputs: heater u_h and ventilation u_v (0..1).
Weather: outdoor temperature, humidity, solar radiation I, wind speed w.

    air      Cap·dT/dt   = η·I + P·u_h − (U + ρc·g)·(T − T_out) − h_f·(T − Tf) − λ·E
    floor    Cap_f·dTf/dt = h_f·(T − Tf) + h_fg·(T_ground − Tf)
    soil     Cap_s·dTs/dt = h_s·(T − Ts) + h_sg·(T_ground − Ts)
    vapour   H·dC/dt     = E − g·(C − C_out)                    (above 100 % RH it condenses)
    water    dSM/dt      = −drainage − uptake by plants,        plus automatic irrigation

    air exchange   g = H·ACH/3600,  ACH = leak + wind·w + stack·√(T − T_out) + vent_max·u_v
    transpiration  E = (a_rad·I + a_vpd·VPD)·f(SM)
    light          PPFD = τ·I,  lux = 54.05·PPFD
"""

import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta

RHO_CP = 1200.0       # J/(m³·K) heat capacity of air
LATENT_HEAT = 2450.0  # J/g evaporation of water
LUX_PER_PPFD = 54.05  # sunlight; the same factor is used in the dataset


# Saturation vapour pressure in kPa (Tetens formula, the dataset's VPD uses it too)
def saturation_vapour_pressure(temp_c: float) -> float:
    return 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))


# Water vapour density (g/m³) from temperature and relative humidity (%)
def vapour_density(temp_c: float, rh: float) -> float:
    return 2166.8 * saturation_vapour_pressure(temp_c) * rh / 100.0 / (temp_c + 273.15)


# Relative humidity (%) from temperature and vapour density
def relative_humidity(temp_c: float, density: float) -> float:
    return 100.0 * density * (temp_c + 273.15) / (2166.8 * saturation_vapour_pressure(temp_c))


# Physical parameters of the model (meanings and units: configs/default.yaml, emulator.physics)
@dataclass
class PhysicsParams:
    thermal_capacity: float
    heat_loss_coeff: float
    solar_gain_factor: float
    heater_max_power: float
    floor_capacity: float
    floor_coupling: float
    floor_ground_coupling: float
    ground_temp: float
    greenhouse_height: float
    leak_ach: float
    wind_ach: float
    stack_ach: float
    vent_max_ach: float
    vent_fan_power: float
    transp_rad_coeff: float
    transp_vpd_coeff: float
    substrate_capacity: float
    substrate_coupling: float
    substrate_ground_coupling: float
    soil_drain_rate: float
    field_capacity: float
    excess_drain_rate: float
    soil_water_capacity_mm: float
    wilting_point: float
    stress_point: float
    irrigation_threshold: float
    irrigation_target: float
    irrigation_start_hour: int
    irrigation_end_hour: int
    light_transmission: float

    # Read the parameters from the config
    @classmethod
    def from_config(cls, cfg: dict) -> "PhysicsParams":
        physics = cfg["emulator"]["physics"]
        # YAML may read numbers like 5e-05 as text, so convert to the declared type
        return cls(**{f.name: f.type(physics[f.name]) for f in fields(cls)})

    # Copy with some parameters changed
    def replace(self, **changes) -> "PhysicsParams":
        return replace(self, **changes)

    # Parameters as a dict (to save as JSON)
    def to_dict(self) -> dict:
        return asdict(self)


# The greenhouse model: current state, one physics step, sensor readings
class GreenhouseEmulator:
    # Store the settings and set a default starting state
    def __init__(self, params: PhysicsParams, dt_minutes: float = 5.0, max_substep_s: float = 60.0,
                 floor_area_m2: float = 100.0, irrigation: bool = True):
        self.p = params
        self.dt_minutes = dt_minutes
        self.max_substep_s = max_substep_s
        self.floor_area_m2 = floor_area_m2
        self.irrigation = irrigation
        self.reset()

    # Set the starting state (air, humidity, floor, soil) and the clock
    def reset(self, temperature: float = 20.0, humidity: float = 75.0, soil_temperature: float | None = None,
              soil_moisture: float = 60.0, time: datetime | None = None, floor_temperature: float | None = None):
        self.T = temperature
        self.C = vapour_density(temperature, humidity)
        self.Ts = self.p.ground_temp if soil_temperature is None else soil_temperature
        self.Tf = self.Ts if floor_temperature is None else floor_temperature
        self.SM = soil_moisture
        self.restart(time or datetime(2025, 1, 1))

    # Keep the physical state; reset the clock, energy counters and irrigation log
    def restart(self, time: datetime):
        self.time = time
        self.heater_energy_kwh = 0.0
        self.vent_energy_kwh = 0.0
        self.irrigation_count = 0
        self._last_irrigation: datetime | None = None
        self.last_transpiration = 0.0

    # Rates of change of the 5 states (the equations at the top of the file) and transpiration
    def _derivatives(self, x, w, u_heat, u_vent, c_out):
        T, C, Tf, Ts, SM = x
        p = self.p
        T_out, solar = w["outdoor_temp"], w["solar_radiation"]
        ach = (p.leak_ach + p.wind_ach * w["wind_speed"] + p.stack_ach * math.sqrt(max(T - T_out, 0.0))
               + p.vent_max_ach * u_vent)
        g = p.greenhouse_height * ach / 3600.0  # m/s
        vpd = max(saturation_vapour_pressure(T) - C * (T + 273.15) / 2166.8, 0.0)
        water = min(max((SM - p.wilting_point) / (p.stress_point - p.wilting_point), 0.0), 1.0)
        E = (p.transp_rad_coeff * solar + p.transp_vpd_coeff * vpd) * water  # g/(m²·s)

        q_air = (p.solar_gain_factor * solar
                 + p.heater_max_power * u_heat
                 - (p.heat_loss_coeff + RHO_CP * g) * (T - T_out)
                 - p.floor_coupling * (T - Tf)
                 - LATENT_HEAT * E)
        q_floor = p.floor_coupling * (T - Tf) + p.floor_ground_coupling * (p.ground_temp - Tf)
        q_soil = p.substrate_coupling * (T - Ts) + p.substrate_ground_coupling * (p.ground_temp - Ts)
        sm_rate = (p.soil_drain_rate + p.excess_drain_rate * max(SM - p.field_capacity, 0.0)
                   + E * 3.6 / p.soil_water_capacity_mm * 100.0)  # %/h
        dx = (q_air / p.thermal_capacity,
              (E - g * (C - c_out)) / p.greenhouse_height,
              q_floor / p.floor_capacity,
              q_soil / p.substrate_capacity,
              -sm_rate / 3600.0)
        return dx, E

    # One time step with the heater and vents (0–100 %) held constant; returns the new readings
    def step(self, weather: dict, heater_power: float = 0.0, ventilation_rate: float = 0.0,
             dt_minutes: float | None = None) -> dict:
        dt = (dt_minutes or self.dt_minutes) * 60.0
        u_h = min(max(heater_power, 0.0), 100.0) / 100.0
        u_v = min(max(ventilation_rate, 0.0), 100.0) / 100.0
        c_out = vapour_density(weather["outdoor_temp"], min(weather["outdoor_rh"], 100.0))

        # rates of change at a given state
        def f(state):
            return self._derivatives(state, weather, u_h, u_v, c_out)

        # state moved along a slope for h seconds
        def moved(state, slope, h):
            return tuple(a + h * b for a, b in zip(state, slope))

        n = max(1, math.ceil(dt / self.max_substep_s))
        h = dt / n
        x = (self.T, self.C, self.Tf, self.Ts, self.SM)
        E = 0.0
        for _ in range(n):  # Runge-Kutta 4
            k1, E = f(x)
            k2, _ = f(moved(x, k1, h / 2))
            k3, _ = f(moved(x, k2, h / 2))
            k4, _ = f(moved(x, k3, h))
            x = tuple(a + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(x, k1, k2, k3, k4))
            x = (x[0], min(max(x[1], 0.0), vapour_density(x[0], 100.0)), *x[2:])  # condensation

        self.T, self.C, self.Tf, self.Ts = x[0], x[1], x[2], x[3]
        self.SM = min(max(x[4], 0.0), 100.0)
        self.time += timedelta(seconds=dt)
        self.last_transpiration = E

        hours = dt / 3600.0
        area_kw = self.floor_area_m2 / 1000.0
        self.heater_energy_kwh += self.p.heater_max_power * u_h * area_kw * hours
        self.vent_energy_kwh += self.p.vent_fan_power * u_v * area_kw * hours
        self._maybe_irrigate()
        return self.observe(weather)

    # Automatic irrigation: soil too dry, daytime, at most once an hour
    def _maybe_irrigate(self):
        p = self.p
        if not self.irrigation or self.SM >= p.irrigation_threshold:
            return
        if not p.irrigation_start_hour <= self.time.hour < p.irrigation_end_hour:
            return
        if self._last_irrigation and self.time - self._last_irrigation < timedelta(hours=1):
            return
        self.SM = p.irrigation_target
        self.irrigation_count += 1
        self._last_irrigation = self.time

    # Current sensor readings (the dict that is sent over MQTT)
    def observe(self, weather: dict) -> dict:
        es = saturation_vapour_pressure(self.T)
        rh = min(relative_humidity(self.T, self.C), 100.0)
        ppfd = self.p.light_transmission * weather["solar_radiation"]
        return {
            "temperature": round(self.T, 3),
            "humidity": round(rh, 2),
            "vpd": round(es * (1 - rh / 100.0), 4),
            "light_lux": round(LUX_PER_PPFD * ppfd, 1),
            "light_ppfd": round(ppfd, 2),
            "soil_moisture": round(self.SM, 2),
            "soil_temperature": round(self.Ts, 3),
            "outdoor_temp": round(weather["outdoor_temp"], 2),
            "outdoor_humidity": round(weather["outdoor_rh"], 1),
            "solar_radiation": round(weather["solar_radiation"], 1),
            "wind_speed": round(weather["wind_speed"], 2),
        }
