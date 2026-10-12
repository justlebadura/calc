"""Funciones de activación de una variable y sus derivadas analíticas exactas.

Cada entrada del registro expone:
    f      : función R -> R (vectorizada con numpy)
    df     : derivada primera f'(x) analítica
    label  : etiqueta para mostrar en la UI
"""

import numpy as np


def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def _dsigmoide(x):
    s = _sigmoide(x)
    return s * (1.0 - s)


def _relu(x):
    return np.maximum(0.0, x)


def _drelu(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1.0, np.where(x < 0, 0.0, 0.5))


def _identidad(x):
    return np.asarray(x, dtype=float)


def _didentidad(x):
    return np.ones_like(np.asarray(x, dtype=float))


ACTIVATIONS = {
    "cuadratica": {
        "label": "x²",
        "f": lambda x: np.asarray(x, dtype=float) ** 2,
        "df": lambda x: 2.0 * np.asarray(x, dtype=float),
    },
    "cubica": {
        "label": "x³",
        "f": lambda x: np.asarray(x, dtype=float) ** 3,
        "df": lambda x: 3.0 * np.asarray(x, dtype=float) ** 2,
    },
    "sigmoide": {
        "label": "σ(x)",
        "f": _sigmoide,
        "df": _dsigmoide,
    },
    "tanh": {
        "label": "tanh(x)",
        "f": np.tanh,
        "df": lambda x: 1.0 - np.tanh(x) ** 2,
    },
    "relu": {
        "label": "ReLU(x)",
        "f": _relu,
        "df": _drelu,
    },
    "identidad": {
        "label": "x (identidad)",
        "f": _identidad,
        "df": _didentidad,
    },
}


def escalar(valor):
    """Convierte la salida de f/df evaluada en un punto a float de Python."""
    return float(np.asarray(valor, dtype=float))
