"""Generación de datasets sintéticos 2D con semilla fija (reproducible).

Clases: 0 = Clase A (azul), 1 = Clase B (roja).
"""

from functools import lru_cache

import numpy as np


def _circulos(n_inner=150, n_outer=250, r_inner=1.0, r_outer=2.0, ruido=0.08, seed=42):
    """Anillo rojo que rodea un núcleo azul: no separable linealmente."""
    rng = np.random.default_rng(seed)
    a1 = rng.uniform(0, 2 * np.pi, n_inner)
    a2 = rng.uniform(0, 2 * np.pi, n_outer)
    X1 = np.column_stack([r_inner * np.cos(a1), r_inner * np.sin(a1)])
    X2 = np.column_stack([r_outer * np.cos(a2), r_outer * np.sin(a2)])
    X = np.vstack([X1, X2]) + rng.normal(0, ruido, (n_inner + n_outer, 2))
    y = np.array([0] * n_inner + [1] * n_outer)
    return X, y


def _bandas(n=400, ancho=1.0, dominio=2.5, seed=42):
    """Banda vertical roja (|x| < ancho) entre puntos azules."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-dominio, dominio, (n, 2))
    y = (np.abs(X[:, 0]) < ancho).astype(int)
    return X, y


def _xor(n_por_grupo=100, centro=1.5, ruido=0.35, seed=42):
    """Cuatro grupos en disposición XOR: imposible para mapas componente a componente."""
    rng = np.random.default_rng(seed)
    centros = [(centro, centro), (-centro, -centro), (centro, -centro), (-centro, centro)]
    clases = [0, 0, 1, 1]
    Xs, ys = [], []
    for c, cl in zip(centros, clases):
        Xs.append(rng.normal(c, ruido, (n_por_grupo, 2)))
        ys.extend([cl] * n_por_grupo)
    return np.vstack(Xs), np.array(ys)


def _blobs(n=240, semilla=42):
    """Dos grupos gaussianos YA separables por una recta en el espacio original.

    Sirve como caso trivial: demuestra que hay problemas que la red resuelve
    SIN deformar los ejes (una recta basta desde t = 0).
    """
    rng = np.random.default_rng(semilla)
    c0 = [-1.4, -1.0]
    c1 = [1.4, 1.0]
    n0 = n // 2
    X0 = rng.normal(c0, 0.45, (n0, 2))
    X1 = rng.normal(c1, 0.45, (n - n0, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * (n - n0))
    return X, y


DATASETS = {
    "lineal": {"label": "Ya separables (una recta basta)", "gen": _blobs},
    "circulos": {"label": "Círculos concéntricos", "gen": _circulos},
    "bandas": {"label": "Bandas verticales", "gen": _bandas},
    "xor": {"label": "XOR (caso imposible)", "gen": _xor},
}


@lru_cache(maxsize=None)
def get_dataset(clave):
    """Devuelve (X, y) cacheado para no regenerar en cada callback."""
    return DATASETS[clave]["gen"]()
