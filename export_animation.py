"""Exporta la deformación del plano a MP4/GIF con matplotlib.

Los ejes permanecen fijos durante toda la animación y la recta de la red
(regresión logística) se dibuja siempre, cambiando de color según su acierto
(gris → naranja → verde), para apreciar cómo intenta adaptarse al dataset.

Ejemplos:
    .venv/bin/python export_animation.py
    .venv/bin/python export_animation.py --dataset bandas --f cuadratica --g identidad --out bandas.mp4
    .venv/bin/python export_animation.py --frames 40 --out deformacion.gif
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from activations import ACTIVATIONS
from datasets import DATASETS
from deformation import (
    grid_segments,
    limites_deformacion,
    transform_points,
)
from metrics import UMBRAL_SEPARABLE, linear_accuracy

COLOR_A = "#1f6fb2"
COLOR_B = "#d62728"
COLOR_GRID = "#ccd5df"


def color_recta(acc):
    if acc >= UMBRAL_SEPARABLE:
        return "#2ca02c"
    if acc >= 0.75:
        return "#e67e22"
    return "#95a5a6"


def dibujar(ax, X, y, t, f, g, label_f, label_g, titulo_ds, lims, segs_fijos):
    ax.clear()
    limx, limy = lims
    segs = grid_segments(t, f, g)
    U = transform_points(X, t, f, g)

    for seg in segs_fijos:  # rejilla original: los ejes nunca cambian
        ax.plot(seg[:, 0], seg[:, 1], color="#e2e8ee", lw=0.7, ls=":", zorder=0)
    for seg in segs:
        ax.plot(seg[:, 0], seg[:, 1], color=COLOR_GRID, lw=0.8, zorder=1)
    ax.scatter(U[y == 0, 0], U[y == 0, 1], c=COLOR_A, s=18,
               label="Clase A", zorder=2, edgecolors="white", linewidths=0.3)
    ax.scatter(U[y == 1, 0], U[y == 1, 1], c=COLOR_B, s=18,
               label="Clase B", zorder=2, edgecolors="white", linewidths=0.3)

    acc, w, b = linear_accuracy(X, y, t, f, g)
    if abs(w[1]) > 1e-12:
        u = [limx[0], limx[1]]
        v = [-(w[0] * ui + b) / w[1] for ui in u]
        ax.plot(u, v, color=color_recta(acc), lw=2.5, ls="--",
                label=f"Recta de la red ({acc:.0%})", zorder=3)

    ax.set_xlim(limx)
    ax.set_ylim(limy)
    ax.set_aspect("equal")
    ax.axhline(0, color="#9fb0bf", lw=0.8)
    ax.axvline(0, color="#9fb0bf", lw=0.8)
    ax.set_title(f"{titulo_ds}  ·  f = {label_f}, g = {label_g}  ·  "
                 f"t = {t:.2f}  ·  separabilidad = {acc:.0%}", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="circulos", choices=list(DATASETS))
    parser.add_argument("--f", default="cuadratica", choices=list(ACTIVATIONS),
                        help="función del eje x")
    parser.add_argument("--g", default=None, choices=list(ACTIVATIONS),
                        help="función del eje y (por defecto, igual que f)")
    parser.add_argument("--out", default="deformacion.mp4",
                        help="archivo de salida (.mp4 o .gif)")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    g_key = args.g or args.f
    X, y = DATASETS[args.dataset]["gen"]()
    f, g = ACTIVATIONS[args.f]["f"], ACTIVATIONS[g_key]["f"]
    label_f, label_g = ACTIVATIONS[args.f]["label"], ACTIVATIONS[g_key]["label"]
    titulo_ds = DATASETS[args.dataset]["label"]
    lims = limites_deformacion(X, f, g)
    segs_fijos = grid_segments(0.0, f, g)  # estática: los ejes no cambian

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ts = [i / (args.frames - 1) for i in range(args.frames)]
    anim = FuncAnimation(
        fig,
        lambda t: dibujar(ax, X, y, t, f, g, label_f, label_g, titulo_ds,
                          lims, segs_fijos),
        frames=ts, interval=1000 / args.fps)

    if args.out.endswith(".gif"):
        anim.save(args.out, writer=PillowWriter(fps=args.fps))
    else:
        anim.save(args.out, writer=FFMpegWriter(fps=args.fps))
    print(f"Guardado en {args.out}")


if __name__ == "__main__":
    main()
