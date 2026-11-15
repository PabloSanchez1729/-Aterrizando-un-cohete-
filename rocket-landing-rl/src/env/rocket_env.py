"""
Wrapper Gymnasium sobre la dinámica del cohete, para poder entrenar agentes
de RL (PPO, SAC, etc. de Stable-Baselines3) y compararlos contra PID/LQR
en igualdad de condiciones (mismo modelo físico, mismas condiciones iniciales).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.dynamics.rocket_dynamics import (
    RocketParams, rk4_step, make_initial_state,
)
from src.utils.quaternion import quat_to_euler


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt: float = 0.05, max_steps: int = 800, wind_std: float = 0.0):
        super().__init__()
        self.params = RocketParams()
        self.dt = dt
        self.max_steps = max_steps
        self.wind_std = wind_std  # desviación estándar de ráfagas de viento, m/s

        # Acción: [throttle, gimbal_pitch, gimbal_yaw] normalizados a [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observación: posición(3) + velocidad(3) + euler(3) + omega(3) + masa(1) = 13
        obs_high = np.array([np.inf] * 13, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.state = None
        self.steps = 0

    def _get_obs(self) -> np.ndarray:
        euler = quat_to_euler(self.state["q"])
        return np.concatenate([
            self.state["r"], self.state["v"], euler, self.state["omega"], self.state["m"]
        ]).astype(np.float32)

    def _action_to_control(self, action: np.ndarray) -> dict:
        throttle = (action[0] + 1.0) / 2.0  # de [-1,1] a [0,1]
        gimbal_pitch = action[1] * self.params.gimbal_max
        gimbal_yaw = action[2] * self.params.gimbal_max
        return {"throttle": throttle, "gimbal_pitch": gimbal_pitch, "gimbal_yaw": gimbal_yaw}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        altitude = rng.uniform(1000, 2000)
        offset = rng.uniform(-150, 150)
        vz0 = rng.uniform(-100, -60)

        self.state = make_initial_state(
            altitude=altitude, horizontal_offset=offset, vertical_speed=vz0, params=self.params
        )
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        control = self._action_to_control(action)

        # Ráfagas de viento como perturbación aleatoria en velocidad horizontal.
        if self.wind_std > 0:
            self.state["v"][:2] += np.random.normal(0, self.wind_std, size=2) * self.dt

        self.state = rk4_step(self.state, control, self.params, self.dt)
        self.steps += 1

        altitude = self.state["r"][2]
        vz = self.state["v"][2]
        horiz_dist = np.linalg.norm(self.state["r"][:2])
        speed = np.linalg.norm(self.state["v"])

        terminated = False
        reward = 0.0

        # --- Reward shaping ---
        # Penaliza distancia al objetivo (0,0,0) y velocidad excesiva, premia
        # mantenerse estable y usar poco combustible.
        reward -= 0.01 * horiz_dist
        reward -= 0.01 * abs(vz)
        reward -= 0.001 * control["throttle"]  # penaliza gasto de combustible
        reward -= 0.1 * np.linalg.norm(self.state["omega"])  # penaliza tumbling

        if altitude <= 0.0:
            terminated = True
            soft_landing = abs(vz) < 5.0 and horiz_dist < 10.0 and speed < 8.0
            if soft_landing:
                reward += 100.0
            else:
                reward -= 50.0  # "crash"

        truncated = self.steps >= self.max_steps
        if self.state["m"][0] <= self.params.dry_mass and altitude > 5.0:
            # Se quedó sin combustible en el aire -> caída libre, mal resultado
            terminated = True
            reward -= 50.0

        return self._get_obs(), reward, terminated, truncated, {"altitude": altitude, "vz": vz}
