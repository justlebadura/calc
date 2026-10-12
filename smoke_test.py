"""Verificación rápida sin navegador.

1. Construye las figuras para todos los datasets × funciones.
2. Comprueba las tesis pedagógicas y que el descenso de gradiente separa
   lo que debe separar.

Ejecutar:  .venv/bin/python smoke_test.py
"""

from activations import ACTIVATIONS
from datasets import DATASETS, get_dataset
from deformation import limites_deformacion, transform_points
from figures import (
    function_figure,
    loss_curve_figure,
    plane_figure,
    plane_figure_frames,
)
from metrics import accuracy_curve, linear_accuracy
from model import gradient_descent_logistic


def figuras_ok():
    for ds in DATASETS:
        X, y = get_dataset(ds)
        for fk in ACTIVATIONS:
            for gk in ACTIVATIONS:
                lims = limites_deformacion(
                    X, ACTIVATIONS[fk]["f"], ACTIVATIONS[gk]["f"])
                for t in (0.0, 0.5, 1.0):
                    acc, w, b = linear_accuracy(
                        X, y, t, ACTIVATIONS[fk]["f"], ACTIVATIONS[gk]["f"])
                    assert 0.0 <= acc <= 1.0
                    fig = plane_figure(X, y, t, fk, gk, sep=(acc, w, b),
                                       lims=lims)
                    assert len(fig.data) >= 4
                # figura animada por frames
                ts = [i / 20 for i in range(21)]
                res = []
                for t in ts:
                    U = transform_points(
                        X, t, ACTIVATIONS[fk]["f"], ACTIVATIONS[gk]["f"])
                    w, b, losses, acc = gradient_descent_logistic(U, y)
                    res.append((acc, w, b, float(losses[-1])))
                ffig = plane_figure_frames(X, y, fk, gk, ts, lims, res)
                assert len(ffig.frames) == len(ts) - 1
                assert ffig.layout.sliders and ffig.layout.updatemenus
        function_figure("cuadratica", "cubica", 0.8, -1.2)
        function_figure("sigmoide", "sigmoide", 0.5, 0.5)
        ts, accs = accuracy_curve(X, y, ACTIVATIONS["cuadratica"]["f"],
                                  ACTIVATIONS["cuadratica"]["f"], pasos=5)
        _, _, losses, _ = gradient_descent_logistic(
            X, y, epochs=30, lr=0.5, C=0.001)
        loss_curve_figure(losses, 0.5, 30)
    print("figuras OK")


def gd_entrena():
    # Círculos + x² separables con gradiente descendente en t=1
    X, y = get_dataset("circulos")
    U = transform_points(X, 1.0, ACTIVATIONS["cuadratica"]["f"],
                         ACTIVATIONS["cuadratica"]["f"])
    w, b, losses, acc = gradient_descent_logistic(U, y, epochs=150, lr=0.5)
    assert acc >= 0.95, f"GD círculos+x² t=1: {acc:.3f}"
    assert losses[-1] < losses[0], "la pérdida debería descender"

    # Círculos + x³ (impar) NO separables
    U3 = transform_points(X, 1.0, ACTIVATIONS["cubica"]["f"],
                          ACTIVATIONS["cubica"]["f"])
    _, _, _, acc3 = gradient_descent_logistic(U3, y, epochs=150, lr=0.5)
    assert acc3 < 0.9, f"GD círculos+x³ no debería separar: {acc3:.3f}"

    # Blobs ya separables en t=0
    Xl, yl = get_dataset("lineal")
    _, _, _, accl = gradient_descent_logistic(Xl, yl, epochs=100, lr=0.5)
    assert accl >= 0.9, f"GD blobs t=0: {accl:.3f}"
    print(f"GD entrena OK (círculos+x²: {acc:.1%}, x³: {acc3:.1%})")


if __name__ == "__main__":
    figuras_ok()
    gd_entrena()
    print("Smoke test superado ✓")