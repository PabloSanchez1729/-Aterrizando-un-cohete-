"""
Tests de sanidad física. El objetivo NO es cobertura de código, es verificar
que la física implementada coincide con soluciones analíticas conocidas.
Si estos tests fallan, no tiene sentido seguir a PID/LQR/RL porque el modelo
base está mal.
"""

import numpy as np
import pytest

from src.dynamics.rocket_dynamics import (
    RocketParams, rk4_step, make_initial_state, state_derivative,
)
from src.utils.quaternion import quat_normalize, quat_mult, quat_conjugate


def test_free_fall_matches_analytic_solution():
    """Sin empuje ni arrastre, el cohete debe caer como caída libre: z(t) = z0 - 0.5*g*t^2
    Usamos un arrastre despreciable poniendo Cd muy bajo para aislar la gravedad."""
    params = RocketParams(cd=0.0)
    state = make_initial_state(altitude=1000.0, horizontal_offset=0.0, vertical_speed=0.0, params=params)
    control = {"throttle": 0.0, "gimbal_pitch": 0.0, "gimbal_yaw": 0.0}

    dt = 0.01
    t = 0.0
    while t < 2.0:
        state = rk4_step(state, control, params, dt)
        t += dt

    z_expected = 1000.0 - 0.5 * 9.80665 * t**2
    assert abs(state["r"][2] - z_expected) < 1.0, "La caída libre no coincide con la solución analítica"


def test_quaternion_stays_normalized_after_integration():
    """Tras muchos pasos de integración con rotación activa, ||q|| debe seguir siendo ~1."""
    params = RocketParams()
    state = make_initial_state(params=params)
    state["omega"] = np.array([0.1, 0.05, 0.02])
    control = {"throttle": 0.5, "gimbal_pitch": 0.05, "gimbal_yaw": -0.03}

    for _ in range(500):
        state = rk4_step(state, control, params, dt=0.02)

    norm = np.linalg.norm(state["q"])
    assert abs(norm - 1.0) < 1e-6, f"El cuaternión se desnormalizó: ||q|| = {norm}"


def test_quaternion_identity_rotation():
    """q_identity * v debe devolver v sin cambios (rotación nula)."""
    from src.utils.quaternion import quat_rotate_vector
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([1.0, 2.0, 3.0])
    v_rotated = quat_rotate_vector(q_identity, v)
    assert np.allclose(v, v_rotated)


def test_quaternion_conjugate_inverts_rotation():
    """q * q_conjugado debe ser la identidad (aplicar una rotación y luego su inversa)."""
    q = quat_normalize(np.array([0.9, 0.1, 0.2, 0.05]))
    result = quat_mult(q, quat_conjugate(q))
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(result, identity, atol=1e-6)


def test_mass_decreases_with_thrust():
    """A mayor throttle, mayor consumo de combustible (m debe disminuir monotónicamente)."""
    params = RocketParams()
    state = make_initial_state(params=params)
    control = {"throttle": 1.0, "gimbal_pitch": 0.0, "gimbal_yaw": 0.0}

    m0 = state["m"][0]
    state = rk4_step(state, control, params, dt=1.0)
    m1 = state["m"][0]

    assert m1 < m0, "La masa debería disminuir al quemar combustible"


def test_mass_never_below_dry_mass():
    """La masa no puede bajar del peso estructural (sin combustible)."""
    params = RocketParams(fuel_mass=10.0)  # muy poco combustible para forzar el límite
    state = make_initial_state(params=params)
    control = {"throttle": 1.0, "gimbal_pitch": 0.0, "gimbal_yaw": 0.0}

    for _ in range(200):
        state = rk4_step(state, control, params, dt=0.5)

    assert state["m"][0] >= params.dry_mass - 1e-6
