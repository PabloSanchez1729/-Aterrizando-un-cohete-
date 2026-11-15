# Rocket Landing: PID vs LQR vs RL

Simulación 6DOF de un cohete reutilizable en fase de aterrizaje (estilo
Falcon 9 booster), comparando tres estrategias de control:

1. **PID** clásico en cascada (altitud, posición, actitud)
2. **LQR** (Linear Quadratic Regulator) vía linealización numérica
3. **RL** (PPO/SAC) entrenado sobre un entorno Gymnasium

El objetivo del proyecto es aprender **dinámica de cuerpo rígido y teoría de
control** de verdad, usando RL como punto de comparación — no como reemplazo
de entender la física.

## Por qué este proyecto

La mayoría de proyectos "ML aplicado a ingeniería" son una red neuronal
entrenada sobre un dataset genérico. Acá el ML es una pieza más dentro de
un modelo físico completo: las ecuaciones de movimiento se derivan e
implementan desde cero (nada de `scipy.integrate.odeint` de caja negra sin
entender qué hay adentro).

## Física implementada

**Estado del cohete** (14 componentes): posición, velocidad, orientación
(cuaternión), velocidad angular, masa.

**Por qué cuaterniones**: los ángulos de Euler tienen singularidades
(gimbal lock) en orientaciones extremas. Un cohete corrigiendo actitud
agresivamente puede pasar por esas orientaciones, así que Euler no es seguro
para la integración numérica (sí lo usamos para logging/visualización).

**Ecuaciones**:
- Traslación: `F = m*a` con fuerzas de empuje vectorizado, gravedad y arrastre
  aerodinámico dependiente de densidad atmosférica (modelo exponencial).
- Rotación: ecuaciones de Euler para cuerpo rígido,
  `I*omega_dot + omega x (I*omega) = torque`
- Masa variable: `m_dot = -thrust / (Isp * g0)` (ecuación del cohete de
  Tsiolkovsky en forma diferencial).
- Integración: Runge-Kutta 4to orden (Euler simple acumula error rápido en
  sistemas rotacionales).

## Estructura del repo

```
rocket-landing-rl/
├── src/
│   ├── dynamics/
│   │   └── rocket_dynamics.py   # ecuaciones de movimiento + integrador RK4
│   ├── control/
│   │   ├── pid_controller.py    # PID en cascada
│   │   └── lqr_controller.py    # linealización + Riccati + LQR
│   ├── env/
│   │   └── rocket_env.py        # wrapper Gymnasium para RL
│   └── utils/
│       └── quaternion.py        # álgebra de cuaterniones
├── tests/
│   └── test_dynamics.py         # verificación contra soluciones analíticas
├── run_demo.py                  # compara PID vs LQR y grafica
├── requirements.txt
└── README.md
```

## Cómo correrlo

```bash
pip install -r requirements.txt
pytest tests/                     # verifica que la física es correcta
python run_demo.py                # compara PID vs LQR, genera landing_comparison.png
```

Para entrenar RL (fase 4, ver roadmap):
```python
from stable_baselines3 import PPO
from src.env.rocket_env import RocketLandingEnv

env = RocketLandingEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

## Roadmap

- [x] **Fase 1**: derivar e implementar dinámica 6DOF, verificar contra
      caída libre analítica (`tests/test_dynamics.py`)
- [x] **Fase 2**: controlador PID en cascada
- [x] **Fase 3**: controlador LQR, comparar contra PID
- [ ] **Fase 4**: entorno Gymnasium + entrenar PPO/SAC
- [ ] **Fase 5**: comparar robustez PID/LQR/RL ante viento y fallas de sensor
- [ ] **Fase 6**: documentar resultados con gráficos de trayectoria y métricas
      (combustible usado, error de aterrizaje, tiempo de convergencia)

## Costos

Todo el proyecto corre con software gratuito (NumPy, SciPy, Gymnasium,
Stable-Baselines3). Las fases 1-3 corren en CPU sin problema. Para entrenar
RL más rápido, Google Colab o Kaggle Notebooks ofrecen GPU gratis con
límites de uso suficientes para este tamaño de problema. No se requiere
hardware físico — todo es simulación.

## Estado actual del demo (`run_demo.py`) y qué esperar

Con las condiciones default del demo (altitud 500m, offset horizontal 50m,
vz inicial -20 m/s):

- **LQR aterriza perfectamente** (vz final ≈ 0, offset final ≈ 0). Los pesos
  Q/R están escalados con la **regla de Bryson** (`Q_ii = 1/rango_esperado²`,
  igual para R) — sin este escalamiento, con Q=I y R=I ingenuos, el
  controlador satura el throttle para errores de 0.06 m/s y nunca converge.
  Este es un error clásico en LQR y vale la pena que lo entiendas: **el
  escalamiento de Q/R importa tanto como el diseño mismo del controlador**,
  porque el estado mezcla unidades muy distintas (metros, m/s, radianes, kg).

- **PID se estrella** con las ganancias default. Esto es intencional y es
  el ejercicio de la Fase 2: el lazo de altitud funciona bien aislado (sin
  offset horizontal), pero el acoplamiento entre el lazo de posición
  horizontal y el de actitud genera oscilaciones que terminan destabilizando
  también la velocidad vertical. Tunear las ganancias a mano — o rediseñar
  el esquema de cascada — es el trabajo que te toca a ti.

**Limitación importante del LQR que sí es física/matemática, no un bug**: si
alejas mucho las condiciones iniciales del punto de equilibrio (por ejemplo,
altitud 1500m con vz=-80 m/s, las condiciones originales que se probaron
durante el desarrollo), el LQR también empieza a fallar. La razón es que el
LQR de horizonte infinito **regula** hacia un punto de equilibrio minimizando
un costo acumulado, pero no sabe que "llegar con velocidad ≈0 justo cuando
la altitud toca 0" es una restricción dura — puede decidir que es "óptimo"
dejar caer el cohete un rato antes de frenar, y si no le da tiempo, se
estrella. Esto es exactamente por qué los sistemas de guiado reales (SpaceX
incluido) no usan LQR puro: usan una capa de **guiado** (trayectoria de
referencia) + LQR/MPC para *seguir* esa trayectoria. Extender este proyecto
a MPC (Model Predictive Control) con restricción terminal es el salto
natural una vez que entiendas por qué LQR solo no alcanza para el problema
completo.

## Limitaciones conocidas (para ser honesto en el README de GitHub)

- El modelo atmosférico es un exponencial simplificado, no un modelo
  estándar como el US Standard Atmosphere 1976.
- El tensor de inercia asume un cilindro uniforme; un cohete real tiene
  masa distribuida de forma no uniforme (tanques, motores).
- No hay modelo de sloshing de combustible líquido (efecto real en cohetes,
  ignorado aquí por simplicidad).
- El gimbal se modela como respuesta instantánea, sin dinámica de actuador.
