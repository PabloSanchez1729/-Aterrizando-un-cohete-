"""
Demo: simula el aterrizaje del cohete con PID y con LQR sobre el MISMO
escenario inicial, y compara las trayectorias.

Uso:
    python run_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from src.dynamics.rocket_dynamics import RocketParams, rk4_step, make_initial_state
from src.control.pid_controller import RocketPIDController
from src.control.lqr_controller import LQRController


def simulate(controller_factory, params, dt=0.05, max_steps=1400, seed=0,
             altitude=500.0, horizontal_offset=50.0, vertical_speed=-20.0):
    rng = np.random.default_rng(seed)
    state = make_initial_state(
        altitude=altitude, horizontal_offset=horizontal_offset, vertical_speed=vertical_speed, params=params
    )
    controller = controller_factory()

    history = {"t": [], "altitude": [], "vz": [], "horiz": [], "throttle": []}
    t = 0.0
    for _ in range(max_steps):
        control = controller.compute(state, dt)
        state = rk4_step(state, control, params, dt)
        t += dt

        history["t"].append(t)
        history["altitude"].append(state["r"][2])
        history["vz"].append(state["v"][2])
        history["horiz"].append(np.linalg.norm(state["r"][:2]))
        history["throttle"].append(control["throttle"])

        if state["r"][2] <= 0:
            break

    return history


def make_lqr(params):
    # Punto de equilibrio: hover perfecto en el origen, throttle que compense gravedad.
    from src.dynamics.rocket_dynamics import make_initial_state
    state_eq = make_initial_state(altitude=0.0, horizontal_offset=0.0, vertical_speed=0.0, params=params)
    hover_throttle = (params.dry_mass + params.fuel_mass) * 9.80665 / params.max_thrust
    control_eq = {"throttle": np.clip(hover_throttle, 0, 1), "gimbal_pitch": 0.0, "gimbal_yaw": 0.0}
    return LQRController(state_eq, control_eq, params)


if __name__ == "__main__":
    params = RocketParams()

    print("Simulando con PID...")
    hist_pid = simulate(lambda: RocketPIDController(params), params)

    print("Simulando con LQR...")
    hist_lqr = simulate(lambda: make_lqr(params), params)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(hist_pid["t"], hist_pid["altitude"], label="PID")
    axes[0].plot(hist_lqr["t"], hist_lqr["altitude"], label="LQR")
    axes[0].set_ylabel("Altitud [m]")
    axes[0].legend()

    axes[1].plot(hist_pid["t"], hist_pid["vz"], label="PID")
    axes[1].plot(hist_lqr["t"], hist_lqr["vz"], label="LQR")
    axes[1].set_ylabel("Velocidad vertical [m/s]")
    axes[1].legend()

    axes[2].plot(hist_pid["t"], hist_pid["horiz"], label="PID")
    axes[2].plot(hist_lqr["t"], hist_lqr["horiz"], label="LQR")
    axes[2].set_ylabel("Distancia horizontal al target [m]")
    axes[2].set_xlabel("Tiempo [s]")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("landing_comparison.png", dpi=150)
    print("Gráfico guardado en landing_comparison.png")

    print(f"\nPID  -> vz final: {hist_pid['vz'][-1]:.2f} m/s, offset final: {hist_pid['horiz'][-1]:.2f} m")
    print(f"LQR  -> vz final: {hist_lqr['vz'][-1]:.2f} m/s, offset final: {hist_lqr['horiz'][-1]:.2f} m")
