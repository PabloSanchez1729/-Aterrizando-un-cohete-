"""
Controlador PID clásico para el aterrizaje del cohete.

Por qué empezar aquí antes de RL: si no puedes explicar por qué un PID
funciona (o falla), no vas a poder diagnosticar por qué un agente de RL
funciona o falla — vas a estar debuggeando una caja negra con otra caja
negra. El PID te da intuición física primero.

Estrategia de control (3 lazos independientes, "cascada"):
1. Lazo de altitud -> controla el throttle (velocidad de descenso deseada)
2. Lazo de posición horizontal -> genera un ángulo de inclinación deseado
3. Lazo de actitud -> controla el gimbal para alcanzar ese ángulo deseado
"""

import numpy as np
from src.utils.quaternion import quat_to_euler


class PID:
    """PID genérico de un solo canal (SISO)."""

    def __init__(self, kp: float, ki: float, kd: float, output_limits=(-1.0, 1.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = None
        self.output_limits = output_limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        # Anti-windup simple: si el output ya está saturado, no sigas acumulando integral.
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        low, high = self.output_limits
        return float(np.clip(output, low, high))


class RocketPIDController:
    """
    Combina 3 PIDs en cascada para producir (throttle, gimbal_pitch, gimbal_yaw)
    a partir del estado completo del cohete.

    Ganancias por defecto: NO están optimizadas, son un punto de partida razonable.
    Tunearlas a mano (o con un pequeño grid search) es parte del ejercicio de la Fase 2.
    """

    def __init__(self, params):
        self.params = params

        # Lazo de velocidad vertical -> throttle
        self.pid_altitude = PID(kp=0.05, ki=0.001, kd=0.15, output_limits=(0.0, 1.0))

        # Lazo de posición horizontal -> ángulo de inclinación deseado (rad)
        max_tilt = np.radians(20.0)
        self.pid_pos_x = PID(kp=0.02, ki=0.0, kd=0.05, output_limits=(-max_tilt, max_tilt))
        self.pid_pos_y = PID(kp=0.02, ki=0.0, kd=0.05, output_limits=(-max_tilt, max_tilt))

        # Lazo de actitud -> gimbal
        self.pid_attitude_pitch = PID(kp=1.5, ki=0.0, kd=0.6, output_limits=(-params.gimbal_max, params.gimbal_max))
        self.pid_attitude_yaw = PID(kp=1.5, ki=0.0, kd=0.6, output_limits=(-params.gimbal_max, params.gimbal_max))

        self.target_descent_rate = -3.0  # m/s, velocidad de descenso deseada al tocar suelo

    def reset(self):
        for pid in [self.pid_altitude, self.pid_pos_x, self.pid_pos_y,
                    self.pid_attitude_pitch, self.pid_attitude_yaw]:
            pid.reset()

    def compute(self, state: dict, dt: float) -> dict:
        r, v, q = state["r"], state["v"], state["q"]
        altitude = r[2]

        # --- Lazo 1: velocidad vertical -> throttle ---
        # Reducimos la velocidad de descenso deseada a medida que nos acercamos al suelo
        # (perfil de frenado suave), para no llegar a v=0 muy alto ni muy tarde.
        desired_vz = self.target_descent_rate * min(1.0, altitude / 200.0 + 0.1)
        vz_error = desired_vz - v[2]
        throttle = self.pid_altitude.update(vz_error, dt)
        throttle = max(throttle, self.params.min_throttle) if altitude > 5.0 else throttle

        # --- Lazo 2: posición horizontal -> ángulo deseado ---
        pos_error_x = 0.0 - r[0]
        pos_error_y = 0.0 - r[1]
        desired_pitch = -self.pid_pos_x.update(pos_error_x, dt)  # inclinar hacia el target
        desired_roll = self.pid_pos_y.update(pos_error_y, dt)

        # --- Lazo 3: actitud actual -> comando de gimbal ---
        current_roll, current_pitch, _ = quat_to_euler(q)
        pitch_error = desired_pitch - current_pitch
        roll_error = desired_roll - current_roll

        gimbal_pitch = self.pid_attitude_pitch.update(pitch_error, dt)
        gimbal_yaw = self.pid_attitude_yaw.update(roll_error, dt)

        return {
            "throttle": throttle,
            "gimbal_pitch": gimbal_pitch,
            "gimbal_yaw": gimbal_yaw,
        }
