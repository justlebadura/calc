"""Métricas: separabilidad lineal del espacio transformado."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from deformation import transform_points

UMBRAL_SEPARABLE = 0.90


def linear_accuracy(X, y, t, f, g):
    """Accuracy de una recta (regresión logística) en el espacio transformado.

    Devuelve (accuracy, w, b) donde w·u + b = 0 es la recta separadora.
    """
    U = transform_points(X, t, f, g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = LogisticRegression(C=1000, max_iter=500)
        clf.fit(U, y)
    return float(clf.score(U, y)), clf.coef_[0], float(clf.intercept_[0])


def accuracy_curve(X, y, f, g, pasos=21):
    """Curva accuracy(t) para el sparkline del panel inferior."""
    ts = np.linspace(0, 1, pasos)
    accs = [linear_accuracy(X, y, t, f, g)[0] for t in ts]
    return ts, accs


def interpretacion(dfx0):
    """Texto y color del estado de la región según la magnitud de f'(x₀)."""
    mag = abs(dfx0)
    sufijo = " (invirtiendo orientación)" if dfx0 < 0 else ""
    if mag > 1.0:
        return f"Expandiendo datos ×{mag:.2f}{sufijo}", "#c0392b"
    if mag < 1.0:
        if mag == 0.0:
            return "Colapsando datos (f′ = 0)", "#7f8c8d"
        return f"Comprimiendo datos ×{mag:.2f}{sufijo}", "#1f6fb2"
    return "Sin deformación local (|f′| = 1)", "#555555"
