"""Entrenamiento de la recta de la red por descenso de gradiente.

A diferencia de ajustar la recta "de golpe" con scikit-learn, aquí la red se
ENTRENA visiblemente: cada época da un paso de gradiente sobre la pérdida
(cross-entropy + regularización L2). Esto hace que los hiperparámetros
(épocas, tasa de aprendizaje, C) sean significativos y observables.
"""

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def _clip_prob(p):
    return np.clip(p, 1e-9, 1 - 1e-9)


def cross_entropy(y, p):
    p = _clip_prob(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradient_descent_logistic(U, y, epochs=100, lr=0.5, C=1e-3, seed=0):
    """Entrena la recta w·u + b = 0 (logística con regularización L2).

    U: (N, 2) puntos transformados; y: (N,) etiquetas en {0, 1}.
    Devuelve (w, b, loss_history, acc):
        w = (w1, w2)  → recta w1·u + w2·v + b = 0
        acc          → accuracy de la recta sobre U
    Las entradas se escalan internamente para que la tasa de aprendizaje sea
    intuitiva; w se devuelve en la escala original de U.
    """
    n = len(y)
    y = np.asarray(y, dtype=float)
    s = float(np.std(U)) or 1.0
    Un = U / s

    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.1, size=2)
    b = 0.0
    losses = []

    for _ in range(int(epochs)):
        p = _sigmoid(Un @ w + b)
        err = p - y
        gw = (Un.T @ err) / n + 2.0 * C * w
        gb = err.mean()
        # clip de gradiente: evita divergencia con tasas altas
        gw = np.clip(gw, -10, 10)
        gb = np.clip(gb, -10, 10)
        w -= lr * gw
        b -= lr * gb

        loss = cross_entropy(y, p) + C * float(np.sum(w * w))
        losses.append(loss)
        if not np.isfinite(loss):
            break

    w_real = w / s
    acc = float(np.mean(((Un @ w + b) > 0).astype(int) == y))
    return w_real, b, np.array(losses), acc