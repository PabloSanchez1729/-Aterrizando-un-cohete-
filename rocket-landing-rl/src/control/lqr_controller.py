"""
Controlador LQR (Linear Quadratic Regulator).

Idea central: el sistema completo (dinámica del cohete) es no lineal, pero
CERCA de un punto de equilibrio (hover: velocidad y aceleración angular ~0),
se puede aproximar como lineal: x_dot = A*x + B*u

LQR encuentra la matriz de ganancia K tal que u = -K*x minimiza el costo
J = integral(x^T Q x + u^T R u dt), es decir, un balance óptimo entre
"qué tan rápido corrijo el error" (Q) y "cuánto esfuerzo de control gasto" (R).

Por qué esto importa más allá de este proyecto: LQR es la base matemática
detrás de la mayoría de los sistemas de control industrial y aeroespacial
reales. Entender esto te da la intuición de "por qué" antes de saltar a RL,
que es esencialmente aprender una política sin garantías analíticas de
estabilidad (a diferencia de LQR, que sí las tiene).
"""

import numpy as np
from scipy.linalg import solve_continuous_are

from src.dynamics.rocket_dynamics import state_derivative


STATE_KEYS = ["r", "v", "q", "omega", "m"]

# Dos simplificaciones necesarias para que el par (A,B) sea "estabilizable"
# (requisito de scipy.linalg.solve_continuous_are) en vez de fallar con
# LinAlgError:
#
# 1. El cuaternión completo (4 componentes) tiene la restricción ||q||=1,
#    que introduce una dirección redundante. Se usa solo la parte VECTORIAL
#    del cuaternión, y el escalar se reconstruye por la restricción de norma.
#
# 2. El gimbal del motor SOLO puede producir torque de pitch/yaw, nunca de
#    "roll" (rotación sobre el eje longitudinal del cohete) — es física
#    correcta: un motor gimbal empuja hacia un lado, pero no puede torcer
#    el cuerpo sobre su propio eje. Esto significa que roll (componente z
#    de la parte vectorial del cuaternión) y velocidad de roll (omega_z)
#    son modos NO controlables por este actuador, con autovalor exactamente
#    en 0 (ni siquiera se amortiguan solos) -> viola estabilizabilidad y
#    Riccati no tiene solución.
#    Como el roll no afecta la trayectoria de aterrizaje (posición, altitud,
#    pitch/yaw), se excluye del estado usado por el LQR. Un cohete real
#    resuelve el control de roll con un actuador aparte (vernier thrusters
#    o motores diferenciales), fuera del alcance de este proyecto.
def _state_to_vector(state: dict) -> np.ndarray:
    q_pitch_yaw = state["q"][1:3]       # solo [qx, qy], se excluye qz (roll)
    omega_pitch_yaw = state["omega"][:2]  # solo [omega_x, omega_y], se excluye omega_z
    return np.concatenate([state["r"], state["v"], q_pitch_yaw, omega_pitch_yaw, state["m"]])


def _vector_to_state(vec: np.ndarray) -> dict:
    idx = 0
    state = {}
    sizes = {"r": 3, "v": 3, "q": 2, "omega": 2, "m": 1}
    for k in STATE_KEYS:
        state[k] = vec[idx: idx + sizes[k]]
        idx += sizes[k]

    qx, qy = state["q"]
    norm_sq = min(qx**2 + qy**2, 1.0)
    qw = np.sqrt(1.0 - norm_sq)
    state["q"] = np.array([qw, qx, qy, 0.0])       # roll (qz) fijo en 0
    state["omega"] = np.array([state["omega"][0], state["omega"][1], 0.0])  # omega_z fijo en 0
    return state


def _control_to_vector(control: dict) -> np.ndarray:
    return np.array([control["throttle"], control["gimbal_pitch"], control["gimbal_yaw"]])


def _vector_to_control(vec: np.ndarray) -> dict:
    return {"throttle": vec[0], "gimbal_pitch": vec[1], "gimbal_yaw": vec[2]}


def linearize(state_eq: dict, control_eq: dict, params, eps: float = 1e-5):
    """
    Linealización numérica (diferencias finitas) alrededor de un punto de
    equilibrio (state_eq, control_eq). Devuelve matrices A y B tales que:
        delta_x_dot ~= A @ delta_x + B @ delta_u

    Se usa diferencias finitas en vez de derivar el Jacobiano a mano porque
    la dinámica tiene cuaterniones y productos cruzados — derivarlo
    analíticamente es propenso a errores. Para producción real sí conviene
    el Jacobiano simbólico, pero para aprender el concepto esto es más claro.
    """
    x0 = _state_to_vector(state_eq)
    u0 = _control_to_vector(control_eq)
    n, m = len(x0), len(u0)

    def f(x_vec, u_vec):
        s = _vector_to_state(x_vec)
        c = _vector_to_control(u_vec)
        d = state_derivative(s, c, params)
        return _state_to_vector(d)

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2 * eps)

    for i in range(m):
        du = np.zeros(m)
        du[i] = eps
        B[:, i] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2 * eps)

    return A, B


def design_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Resuelve la ecuación algebraica de Riccati continua (CARE) para obtener P,
    y de ahí la ganancia óptima K = R^-1 * B^T * P.

    Q penaliza error de estado (posición, velocidad, actitud lejos del target).
    R penaliza esfuerzo de control (gastar mucho combustible / gimbal agresivo).
    Subir Q relativo a R = controlador más agresivo. Subir R = más conservador.
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


class LQRController:
    """Controlador LQR envuelto para operar directo sobre el estado del cohete."""

    def __init__(self, state_eq: dict, control_eq: dict, params, Q=None, R=None):
        self.state_eq = state_eq
        self.control_eq = control_eq
        self.params = params

        n = len(_state_to_vector(state_eq))
        m = len(_control_to_vector(control_eq))

        # Pesos default via regla de Bryson: Q_ii = 1/(rango_esperado_i)^2,
        # R_ii = 1/(rango_esperado_control_i)^2. Esto es CRÍTICO en LQR:
        # como el estado mezcla unidades muy distintas (metros, m/s, radianes,
        # rad/s, kg) y los controles también (throttle en [0,1] vs gimbal en
        # radianes ~0.14 max), usar Q=I y R=I sin escalar produce ganancias
        # absurdamente altas que saturan el actuador para errores minúsculos
        # (ej. 0.06 m/s de error ya saturaba el throttle). Con el estado
        # reducido, el orden de índices es:
        # [0:3]=r, [3:6]=v, [6:8]=q_pitch_yaw, [8:10]=omega_pitch_yaw, [10]=m
        if Q is None:
            Q = np.eye(n)
            Q[0, 0] = Q[1, 1] = 1 / 150**2   # posición horizontal, rango ~150m
            Q[2, 2] = 1 / 150**2              # altitud, rango ~150m
            Q[3, 3] = Q[4, 4] = 1 / 50**2      # velocidad horizontal, rango ~50 m/s
            Q[5, 5] = 1 / 50**2                # velocidad vertical, rango ~50 m/s
            Q[6, 6] = Q[7, 7] = 1 / 0.3**2      # actitud pitch/yaw, rango ~0.3 rad
            Q[8, 8] = Q[9, 9] = 1 / 0.5**2      # velocidad angular, rango ~0.5 rad/s
            Q[10, 10] = 1e-10                   # masa, prácticamente irrelevante para el costo
        if R is None:
            R = np.diag([1/0.5**2, 1/params.gimbal_max**2, 1/params.gimbal_max**2]) * 0.02

        self.A, self.B = linearize(state_eq, control_eq, params)
        self.K = design_lqr(self.A, self.B, Q, R)
        self.x_eq = _state_to_vector(state_eq)
        self.u_eq = _control_to_vector(control_eq)

    def compute(self, state: dict, dt: float = None) -> dict:
        x = _state_to_vector(state)
        delta_x = x - self.x_eq
        delta_u = -self.K @ delta_x
        u = self.u_eq + delta_u

        u[0] = np.clip(u[0], 0.0, 1.0)  # throttle
        u[1] = np.clip(u[1], -self.params.gimbal_max, self.params.gimbal_max)
        u[2] = np.clip(u[2], -self.params.gimbal_max, self.params.gimbal_max)

        return _vector_to_control(u)
