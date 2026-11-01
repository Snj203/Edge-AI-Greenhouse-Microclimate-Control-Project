"""
PID controller: one discrete PID loop per actuator, run on every sensor message (every 5 min).

    e = T_target − T              (heater loop; the vent loop uses T − T_vent)
    P = Kp·e    I = I_prev + Ki·e·dt (limited to ±anti_windup_limit)    D = Kd·(e − e_prev)/dt
    u = clamp(P + I + D, 0, 100)

The gains use minutes as the time unit (dt = 5).

Anti-windup has two parts: the integral is limited, and it does not grow while the output is
already at 0 or 100 % and the error pushes it further. Without the second part the integral piles
up, for example while the heater is off all afternoon, and the heater reacts late in the evening.

    heat       heater only, vents closed (the plan's baseline)
    heat_vent  + a second loop that opens the vents above target + setpoint_offset.
               While the heater is on the vents stay closed and the vent loop is reset: otherwise in
               the evening the vent integral from the day keeps the vents open while the heater is
               already heating again, and the two work against each other for about an hour.
"""

from src.controllers.base import Controller

MODES = ("heat", "heat_vent")


# One PID loop
class PID:
    # cooling=False: output grows when the measurement is below the set-point (heater);
    # cooling=True: output grows when the measurement is above the set-point (ventilation)
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float, dt: float,
                 cooling: bool = False, integral_limit: float = 100.0, out_min: float = 0.0, out_max: float = 100.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.dt = dt
        self.cooling = cooling
        self.integral_limit = integral_limit
        self.out_min, self.out_max = out_min, out_max
        self.reset()

    # Start from zero: no integral, no previous error
    def reset(self):
        self.integral = 0.0
        self.prev_error: float | None = None

    # Error with the sign that makes the output grow
    def error(self, measurement: float) -> float:
        return measurement - self.setpoint if self.cooling else self.setpoint - measurement

    # One step: P + I + D for the new measurement, limited to out_min..out_max
    def update(self, measurement: float) -> float:
        e = self.error(measurement)
        p = self.kp * e
        d = 0.0 if self.prev_error is None else self.kd * (e - self.prev_error) / self.dt
        self.prev_error = e
        integral = min(max(self.integral + self.ki * e * self.dt, -self.integral_limit), self.integral_limit)
        u = p + integral + d
        winding_up = (u > self.out_max and e > 0) or (u < self.out_min and e < 0)
        if not winding_up:
            self.integral = integral
        return min(max(p + self.integral + d, self.out_min), self.out_max)


# Greenhouse PID controller: heater loop, plus a vent loop in the heat_vent variant
class PIDController(Controller):
    # Store the loops; the name tells the variant
    def __init__(self, heater: PID, vent: PID | None = None, name: str | None = None):
        super().__init__()
        self.heater = heater
        self.vent = vent
        self.name = name or ("pid_heat_vent" if vent else "pid_heat")

    # Build from the config. `gains` override it: kp, ki, kd (heater loop),
    # vent_kp, vent_ki, vent_kd, vent_offset (vent loop)
    @classmethod
    def from_config(cls, cfg: dict, mode: str = "heat_vent", **gains) -> "PIDController":
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        p = cfg["pid"]
        v = p["ventilation"]
        dt = cfg["emulator"]["timestep_minutes"]
        target = cfg["target"]["temperature"]
        limit = p["anti_windup_limit"]
        heater = PID(gains.get("kp", p["kp"]), gains.get("ki", p["ki"]), gains.get("kd", p["kd"]),
                     target, dt, integral_limit=limit)
        vent = None
        if mode == "heat_vent":
            vent = PID(gains.get("vent_kp", v["kp"]), gains.get("vent_ki", v["ki"]), gains.get("vent_kd", v["kd"]),
                       target + gains.get("vent_offset", v["setpoint_offset"]), dt, cooling=True,
                       integral_limit=limit)
        return cls(heater, vent, name=f"pid_{mode}")

    # Reset both loops before a new run
    def reset(self):
        self.heater.reset()
        if self.vent:
            self.vent.reset()

    # Heater from the heater loop; vents only while the heater is off
    def compute(self, obs: dict) -> tuple[float, float]:
        t = obs["temperature"]
        heater = self.heater.update(t)
        if self.vent is None:
            return heater, 0.0
        if heater > 0.0:
            self.vent.reset()  # no venting while heating; the vent loop starts afresh next time
            return heater, 0.0
        return heater, self.vent.update(t)
