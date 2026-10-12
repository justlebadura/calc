"""Deformación del plano: interpolación P(t) = (1-t)·X + t·F(X).

La transformación de la IA es componente a componente: T(x, y) = (f(x), g(y)).
El parámetro t ∈ [0, 1] anima la transición continua entre el espacio de
entrada y el espacio transformado.
"""

import numpy as np

RANGO = 2.5       # dominio de visualización [-RANGO, RANGO]
GRID_LINEAS = 9   # líneas verticales y horizontales de la cuadrícula
GRID_PUNTOS = 60  # muestras por línea de cuadrícula


def transform_points(X, t, f, g):
    """Aplica P(t) = (1-t)·[x, y] + t·[f(x), g(y)] a un array (N, 2)."""
    X = np.asarray(X, dtype=float)
    U = np.empty_like(X)
    U[:, 0] = f(X[:, 0])
    U[:, 1] = g(X[:, 1])
    return (1.0 - t) * X + t * U


def grid_segments(t, f, g, rango=RANGO, n_lineas=GRID_LINEAS, n_puntos=GRID_PUNTOS):
    """Cuadrícula cartesiana deformada: lista de segmentos (n_puntos, 2)."""
    coords = np.linspace(-rango, rango, n_lineas)
    s = np.linspace(-rango, rango, n_puntos)
    segmentos = []
    for c in coords:
        vert = np.column_stack([np.full(n_puntos, c), s])
        horiz = np.column_stack([s, np.full(n_puntos, c)])
        segmentos.append(transform_points(vert, t, f, g))
        segmentos.append(transform_points(horiz, t, f, g))
    return segmentos


def square_limits(segmentos, extra=None, margen=0.25):
    """Límites (limx, limy) con la MISMA escala en ambos ejes.

    Acomoda funciones expansivas (x³) expandiendo el eje de menor span para
    mantener la proporción 1:1 sin recortar la geometría.
    """
    nubes = [np.vstack(segmentos)]
    if extra is not None:
        nubes.append(np.asarray(extra, dtype=float))
    pts = np.vstack(nubes)
    lo = pts.min(axis=0) - margen
    hi = pts.max(axis=0) + margen
    centro = (lo + hi) / 2.0
    semi = float((hi - lo).max()) / 2.0
    return (centro[0] - semi, centro[0] + semi), (centro[1] - semi, centro[1] + semi)


def limites_deformacion(X, f, g):
    """Límites FIJOS para toda la animación: unión de los extremos t=0 y t=1.

    Cada punto viaja en línea recta entre P(0) y P(1), así que la caja que
    contiene ambos extremos contiene también todas las posiciones intermedias.
    Esto evita que los ejes "salten" y la animación sea inentendible.
    """
    segs = grid_segments(0.0, f, g) + grid_segments(1.0, f, g)
    extra = np.vstack([transform_points(X, 0.0, f, g),
                       transform_points(X, 1.0, f, g)])
    return square_limits(segs, extra=extra)
