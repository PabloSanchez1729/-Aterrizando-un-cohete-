"""
Utilidades de cuaterniones (convención escalar-primero: q = [w, x, y, z]).

Por qué cuaterniones y no ángulos de Euler:
- Euler tiene singularidades (gimbal lock) cuando el pitch se acerca a +-90 grados.
  Un cohete aterrizando puede pasar por orientaciones extremas durante maniobras
  de corrección, así que Euler puede romperse justo cuando más lo necesitas.
- Cuaterniones no tienen singularidades y son más baratos de integrar numéricamente
  que matrices de rotación (4 numeros vs 9).
"""

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normaliza el cuaternión. Es obligatorio hacerlo tras cada paso de integración
    porque la integración numérica introduce error que hace ||q|| != 1 con el tiempo,
    y un cuaternión no unitario ya no representa una rotación válida."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Producto de Hamilton q1 * q2. Representa componer dos rotaciones:
    primero aplicar la rotación q2, luego la rotación q1."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugado = inverso para un cuaternión unitario. Rota en sentido contrario."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rota el vector v (marco cuerpo -> marco mundo) usando q.
    Formula: v_world = q * [0, v] * q_conjugado, tomando la parte vectorial."""
    q = quat_normalize(q)
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    result = quat_mult(quat_mult(q, v_quat), quat_conjugate(q))
    return result[1:]


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """dq/dt dado la velocidad angular en el marco cuerpo.
    Esta es la ecuación cinemática que conecta 'cómo rota el cuerpo ahora'
    con 'cómo cambia su orientación'."""
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * quat_mult(q, omega_quat)


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convierte a angulos de Euler (roll, pitch, yaw) SOLO para visualización/logging.
    Nunca uses esto dentro del loop de integración física (por eso existe este archivo)."""
    w, x, y, z = quat_normalize(q)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([roll, pitch, yaw])
