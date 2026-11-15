"""
Modelo de dinámica de cuerpo rígido en 6 grados de libertad (6DOF) para un
cohete en fase de descenso/aterrizaje, estilo Falcon 9 booster.

Marco de referencia:
- Mundo (inercial): Z apunta hacia arriba, X-Y es el plano del suelo.
- Cuerpo: Z_body apunta desde la base hacia la punta del cohete (dirección
  nominal del empuje es -Z_body, es decir el motor empuja "hacia atrás").

Estado (vector de 14 componentes):
    r        (3,)  posición en marco mundo [m]
    v        (3,)  velocidad en marco mundo [m/s]
    q        (4,)  cuaternión orientación cuerpo->mundo
    omega    (3,)  velocidad angular en marco cuerpo [rad/s]
    m        (1,)  masa actual [kg] (disminuye por consumo de combustible)

Por qué separamos masa como estado: el cohete quema combustible, así que la masa
NO es constante. Esto es clave — muchos proyectos de juguete asumen masa constante
y eso hace que el control aprendido no sea transferible a un cohete real.
"""

import numpy as np
from src.utils.quaternion import quat_derivative, quat_normalize, quat_rotate_vector

# --- Constantes físicas ---
G0 = 9.80665          # gravedad estándar [m/s^2]
RHO0 = 1.225          # densidad del aire a nivel del mar [kg/m^3]
SCALE_HEIGHT = 8500.0  # altura de escala atmosférica [m] (modelo exponencial simplificado)


class RocketParams:
    """Parámetros físicos del cohete. Los valores default son aproximaciones
    de orden de magnitud de un booster tipo Falcon 9 en fase de aterrizaje
    (mucho más liviano que en despegue porque ya gastó la mayoría del combustible)."""

    def __init__(
        self,
        dry_mass: float = 25000.0,       # kg, masa estructural sin combustible
        fuel_mass: float = 5000.0,       # kg, combustible restante para el aterrizaje
        isp: float = 300.0,              # s, impulso específico del motor
        max_thrust: float = 800000.0,    # N, empuje máximo de un motor Merlin (aprox)
        min_throttle: float = 0.4,       # fracción mínima de empuje (los motores no apagan a cero)
        length: float = 40.0,            # m, largo del cohete
        radius: float = 1.83,            # m, radio del cohete
        cd: float = 0.5,                 # coeficiente de arrastre aproximado
        gimbal_max_deg: float = 8.0,     # grados, deflexión máxima del gimbal del motor
    ):
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.isp = isp
        self.max_thrust = max_thrust
        self.min_throttle = min_throttle
        self.length = length
        self.radius = radius
        self.cd = cd
        self.gimbal_max = np.radians(gimbal_max_deg)
        self.ref_area = np.pi * radius**2

        # Tensor de inercia aproximando el cohete como un cilindro uniforme.
        # I_axial: eje longitudinal (fácil de rotar - "roll" del cohete)
        # I_transverse: ejes perpendiculares (rotar de "pitch/yaw" - mucho más inercia)
        total_mass = dry_mass + fuel_mass
        self.I_axial = 0.5 * total_mass * radius**2
        self.I_transverse = (1/12) * total_mass * (3*radius**2 + length**2)
        self.inertia = np.array([self.I_transverse, self.I_transverse, self.I_axial])

        # Distancia del centro de masa al motor (para el torque del gimbal).
        # Aproximación: motor en la base, CoM cerca del centro geométrico.
        self.thrust_arm = length / 2.0


def atmospheric_density(altitude: float) -> float:
    """Modelo exponencial simple de densidad atmosférica. No es exacto (la atmósfera
    real tiene capas con distintos gradientes térmicos) pero captura lo esencial:
    el arrastre cae fuertemente con la altitud."""
    return RHO0 * np.exp(-max(altitude, 0.0) / SCALE_HEIGHT)


def aerodynamic_drag_force(velocity_world: np.ndarray, altitude: float, params: RocketParams) -> np.ndarray:
    """Fuerza de arrastre aerodinámico en marco mundo.
    F_drag = -0.5 * rho * v^2 * Cd * A * v_hat
    El signo negativo porque el arrastre siempre se opone a la velocidad relativa al aire."""
    speed = np.linalg.norm(velocity_world)
    if speed < 1e-6:
        return np.zeros(3)
    rho = atmospheric_density(altitude)
    drag_magnitude = 0.5 * rho * speed**2 * params.cd * params.ref_area
    return -drag_magnitude * (velocity_world / speed)


def thrust_force_and_torque(
    throttle: float,
    gimbal_pitch: float,
    gimbal_yaw: float,
    params: RocketParams,
):
    """
    Dado el throttle (0 a 1) y los ángulos de gimbal del motor, calcula:
    - la fuerza de empuje en marco CUERPO
    - el torque que ese empuje desalineado genera sobre el centro de masa

    El motor está en la base del cohete y expulsa gases hacia -Z_body (hacia
    "atrás" respecto a la nariz). Por la tercera ley de Newton, la FUERZA que
    ese escape ejerce sobre el cohete es en +Z_body (empuja el cohete "hacia
    adelante", es decir, hacia la nariz). Si el gimbal inclina el motor, parte
    del empuje se redirige a X/Y del cuerpo, creando fuerza lateral y un
    torque de corrección de actitud (así es como un cohete real se endereza).
    """
    throttle = np.clip(throttle, 0.0, 1.0)
    thrust_mag = throttle * params.max_thrust

    gp = np.clip(gimbal_pitch, -params.gimbal_max, params.gimbal_max)
    gy = np.clip(gimbal_yaw, -params.gimbal_max, params.gimbal_max)

    # Dirección del empuje en marco cuerpo tras aplicar el gimbal.
    thrust_dir_body = np.array([
        np.sin(gp),
        np.sin(gy),
        np.cos(gp) * np.cos(gy),
    ])
    force_body = thrust_mag * thrust_dir_body

    # Torque = brazo_de_palanca x fuerza. El motor está a -thrust_arm en Z_body
    # (en la base), así que el brazo de palanca es [0, 0, -thrust_arm].
    lever_arm = np.array([0.0, 0.0, -params.thrust_arm])
    torque_body = np.cross(lever_arm, force_body)

    return force_body, torque_body


def state_derivative(state: dict, control: dict, params: RocketParams) -> dict:
    """
    Calcula d(estado)/dt dado el estado actual y los comandos de control.

    control = {
        "throttle": float en [0, 1],
        "gimbal_pitch": float en radianes,
        "gimbal_yaw": float en radianes,
    }
    """
    r, v, q, omega, m_vec = state["r"], state["v"], state["q"], state["omega"], state["m"]
    m = float(m_vec[0])  # masa como escalar; se guarda como array de 1 elemento en el estado
    q = quat_normalize(q)
    altitude = r[2]

    # --- Fuerzas ---
    force_body, torque_body = thrust_force_and_torque(
        control["throttle"], control["gimbal_pitch"], control["gimbal_yaw"], params
    )
    thrust_world = quat_rotate_vector(q, force_body)
    gravity_world = np.array([0.0, 0.0, -G0 * m])
    drag_world = aerodynamic_drag_force(v, altitude, params)

    total_force = thrust_world + gravity_world + drag_world

    # --- Traslación: F = m*a  =>  a = F/m ---
    r_dot = v.copy()
    v_dot = total_force / m

    # --- Rotación: ecuaciones de Euler para cuerpo rígido en marco cuerpo ---
    # I*omega_dot + omega x (I*omega) = torque
    # despejando: omega_dot = I^-1 * (torque - omega x (I*omega))
    I = params.inertia
    gyroscopic = np.cross(omega, I * omega)
    omega_dot = (torque_body - gyroscopic) / I

    q_dot = quat_derivative(q, omega)

    # --- Consumo de combustible: m_dot = -thrust / (Isp * g0) ---
    thrust_mag = control["throttle"] * params.max_thrust
    m_dot = -thrust_mag / (params.isp * G0) if m > params.dry_mass else 0.0

    return {"r": r_dot, "v": v_dot, "q": q_dot, "omega": omega_dot, "m": np.array([m_dot])}


def rk4_step(state: dict, control: dict, params: RocketParams, dt: float) -> dict:
    """Integrador Runge-Kutta de 4to orden. Se usa RK4 en vez de Euler simple
    porque Euler acumula error rápido en sistemas rotacionales/oscilatorios
    como este, especialmente con dt no infinitesimal."""

    def add(s, ds, scale):
        return {k: s[k] + scale * ds[k] for k in s}

    k1 = state_derivative(state, control, params)
    k2 = state_derivative(add(state, k1, dt/2), control, params)
    k3 = state_derivative(add(state, k2, dt/2), control, params)
    k4 = state_derivative(add(state, k3, dt), control, params)

    new_state = {}
    for key in state:
        new_state[key] = state[key] + (dt/6.0) * (k1[key] + 2*k2[key] + 2*k3[key] + k4[key])

    new_state["q"] = quat_normalize(new_state["q"])
    new_state["m"] = np.clip(new_state["m"], params.dry_mass, params.dry_mass + params.fuel_mass)
    return new_state


def make_initial_state(
    altitude: float = 1500.0,
    horizontal_offset: float = 100.0,
    vertical_speed: float = -80.0,
    params: RocketParams = None,
) -> dict:
    """Estado inicial típico: cohete cayendo desde 1500m con velocidad vertical
    negativa (descendiendo) y algo de offset horizontal que debe corregir."""
    if params is None:
        params = RocketParams()
    return {
        "r": np.array([horizontal_offset, 0.0, altitude]),
        "v": np.array([0.0, 0.0, vertical_speed]),
        "q": np.array([1.0, 0.0, 0.0, 0.0]),  # orientación vertical, sin rotación
        "omega": np.array([0.0, 0.0, 0.0]),
        "m": np.array([params.dry_mass + params.fuel_mass]),
    }
