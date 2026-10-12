"""Constructores de figuras Plotly para los tres módulos visuales.

El plano usa ANIMACIÓN POR FRAMES de Plotly: el slider y play/pausa son
nativos del navegador, de modo que deformar el espacio no reconstruye ni
reenvía la figura al servidor en cada tick (sin update recurrente del DOM).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from activations import ACTIVATIONS, escalar
from deformation import RANGO, grid_segments, square_limits, transform_points
from metrics import UMBRAL_SEPARABLE

# Paleta clara y de alto contraste
COLOR_A = "#2563eb"      # azul
COLOR_B = "#e11d48"      # rojo/rosa
COLOR_GRID = "#c3ced9"   # rejilla deformada
COLOR_GRID_FIJA = "#e7ecf2"  # rejilla original (punteada)
COLOR_SEP = "#16a34a"    # verde: separa
COLOR_PARCIAL = "#f59e0b"  # naranja/ámbar: mejora
COLOR_FALLO = "#9ca3af"  # gris: no separa
COLOR_TAN = "#ea580c"    # tangente
COLOR_CURVA = "#1f2937"
TEMPLATE = "plotly_white"
GRID_EJE = "#eef2f6"
EJE_CERO = "#94a3b8"


def ecuacion_recta(w, b):
    """Ecuación de la recta de la red: w₁·u + w₂·v + b = 0."""
    return (f"{w[0]:.2f}·u {'+' if w[1] >= 0 else '−'} {abs(w[1]):.2f}·v "
            f"{'+' if b >= 0 else '−'} {abs(b):.2f} = 0")


def _segmentos_a_xy(segmentos):
    xs, ys = [], []
    for seg in segmentos:
        xs += seg[:, 0].tolist() + [None]
        ys += seg[:, 1].tolist() + [None]
    return xs, ys


def _recta_separadora(w, b, limx, limy):
    u = np.linspace(limx[0], limx[1], 200)
    if abs(w[1]) > 1e-12:
        v = -(w[0] * u + b) / w[1]
        m = (v >= limy[0]) & (v <= limy[1])
        if m.sum() >= 2:
            return [u[m][0], u[m][-1]], [v[m][0], v[m][-1]]
    if abs(w[0]) > 1e-12:
        v = np.linspace(limy[0], limy[1], 200)
        u2 = -(w[1] * v + b) / w[0]
        m = (u2 >= limx[0]) & (u2 <= limx[1])
        if m.sum() >= 2:
            return [u2[m][0], u2[m][-1]], [v[m][0], v[m][-1]]
    return [], []


def _color_recta(acc):
    if acc >= UMBRAL_SEPARABLE:
        return COLOR_SEP
    if acc >= 0.75:
        return COLOR_PARCIAL
    return COLOR_FALLO


def _status_annotation(t, acc, w, b, loss):
    """Texto de estado del instante t (ecuación de la recta + accuracy + pérdida)."""
    eq = ecuacion_recta(w, b)
    estado = "separadas" if acc >= UMBRAL_SEPARABLE else "buscando…"
    return go.layout.Annotation(
        x=0.02, y=1.0, xref="paper", yref="paper",
        text=(f"<b>t = {t:.2f}</b>  |  {eq}  |  accuracy {acc:.0%}  "
              f"|  pérdida {loss:.3f}  →  {estado}"),
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=13, color="#334155"))


def _traces_plano(X, y, t, f, g, lims, acc, w, b, showlegend):
    """Traces de un instante t (mismo orden y número en cada frame)."""
    limx, limy = lims
    segs = grid_segments(t, f, g)
    U = transform_points(X, t, f, g)

    fjx, fjy = _segmentos_a_xy(grid_segments(0.0, f, g))
    gx, gy = _segmentos_a_xy(segs)
    traces = [
        go.Scatter(x=fjx, y=fjy, mode="lines", name="Rejilla original (fija)",
                   line=dict(color=COLOR_GRID_FIJA, width=1, dash="dot"),
                   hoverinfo="skip", showlegend=showlegend),
        go.Scatter(x=gx, y=gy, mode="lines",
                   line=dict(color=COLOR_GRID, width=1),
                   hoverinfo="skip", showlegend=False),
    ]
    for clase, nombre, color in ((0, "Clase A", COLOR_A), (1, "Clase B", COLOR_B)):
        m = y == clase
        traces.append(go.Scatter(
            x=U[m, 0], y=U[m, 1], mode="markers", name=nombre,
            marker=dict(size=9, color=color, opacity=0.92,
                        line=dict(width=1.2, color="white")),
            customdata=np.column_stack([X[m, 0], X[m, 1]]),
            hovertemplate=(f"{nombre}<br>x₀=%{{customdata[0]:.3f}}, "
                           f"y₀=%{{customdata[1]:.3f}}<br>u=%{{x:.3f}}, "
                           f"v=%{{y:.3f}}<extra></extra>"),
            showlegend=showlegend))
    sx, sy = _recta_separadora(w, b, limx, limy)
    separado = acc >= UMBRAL_SEPARABLE
    traces.append(go.Scatter(
        x=sx, y=sy, mode="lines", name=f"Recta de la red ({acc:.0%})",
        line=dict(color=_color_recta(acc), width=3.5 if separado else 2.5,
                  dash="solid" if separado else "dash"),
        hoverinfo="skip", showlegend=showlegend))
    return traces


def plane_figure(X, y, t, f_key, g_key, sep=None, lims=None):
    """Plano estático para un instante t (usado por los tests)."""
    f = ACTIVATIONS[f_key]["f"]
    g = ACTIVATIONS[g_key]["f"]
    limx, limy = lims if lims is not None else square_limits(
        grid_segments(t, f, g), extra=transform_points(X, t, f, g))
    acc, w, b = (sep if sep is not None else (0.0, (0.0, 0.0), 0.0))
    fig = go.Figure(data=_traces_plano(X, y, t, f, g, (limx, limy),
                                       acc, w, b, True))
    fig.update_layout(template=TEMPLATE, height=580,
                      xaxis=dict(range=limx), yaxis=dict(range=limy))
    return fig


def plane_figure_frames(X, y, f_key, g_key, t_values, lims, resultados):
    """Plano ANIMADO por frames de Plotly (cliente).

    resultados: lista de tuplas (acc, w, b, loss) para cada t en t_values.
    El slider y play/pausa corren en el navegador; el estado por frame (recta,
    accuracy y pérdida) viaja como anotación de layout.
    """
    f = ACTIVATIONS[f_key]["f"]
    g = ACTIVATIONS[g_key]["f"]
    n = len(t_values)

    frames = []
    data0 = _traces_plano(X, y, t_values[0], f, g, lims, *resultados[0][:3],
                          showlegend=True)
    for i in range(1, n):
        acc, w, b, loss = resultados[i]
        frames.append(go.Frame(
            data=_traces_plano(X, y, t_values[i], f, g, lims, acc, w, b,
                               showlegend=False),
            layout=dict(annotations=[
                _status_annotation(t_values[i], acc, w, b, loss)]),
            name=f"f{i}"))

    fig = go.Figure(data=data0)
    fig.frames = frames
    fig.update_layout(annotations=[
        _status_annotation(t_values[0], *resultados[0])])

    slider_steps = [
        dict(method="animate",
             args=[[f"f{i}"], dict(mode="immediate",
                                   frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))],
             label=f"{t_values[i]:.2f}")
        for i in range(n)
    ]

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text="<b>Plano de datos</b> — deformación del espacio",
                   font=dict(size=15)),
        xaxis=dict(range=lims[0], zeroline=True, zerolinecolor=EJE_CERO,
                   zerolinewidth=1.4, showgrid=False, title="u"),
        yaxis=dict(range=lims[1], zeroline=True, zerolinecolor=EJE_CERO,
                   zerolinewidth=1.4, showgrid=False, title="v",
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0.5,
                    xanchor="center", font=dict(size=12)),
        sliders=[dict(
            active=0, pad=dict(b=6, t=34), steps=slider_steps,
            currentvalue=dict(prefix="t = ", visible=True, xanchor="right"),
            font=dict(size=11))],
        updatemenus=[dict(
            type="buttons", showactive=False, direction="left",
            x=0.72, y=1.12, xanchor="left", yanchor="top", pad=dict(r=10, t=4),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=150, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pausa", method="animate",
                     args=[[None], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=False))]),
            ])],
        margin=dict(l=55, r=25, t=70, b=90),
        height=600,
    )
    return fig


def _agregar_funcion(fig, f_key, x0, tag, row=None):
    """Curva y = f(x), punto (x₀, f(x₀)) y su recta tangente, con guías."""
    f = ACTIVATIONS[f_key]["f"]
    df = ACTIVATIONS[f_key]["df"]
    label = ACTIVATIONS[f_key]["label"]
    xs = np.linspace(-RANGO, RANGO, 400)
    ys = f(xs)
    kw = {} if row is None else {"row": row, "col": 1}
    ann = {} if row is None else {"row": row, "col": 1}

    ylo, yhi = float(np.min(ys)), float(np.max(ys))
    margen = 0.12 * max(yhi - ylo, 1e-9)
    rango_y = [ylo - margen, yhi + margen]
    if row is None:
        fig.update_xaxes(range=[-RANGO - 0.1, RANGO + 0.1])
        fig.update_yaxes(range=rango_y)
    else:
        fig.update_xaxes(range=[-RANGO - 0.1, RANGO + 0.1], row=row, col=1)
        fig.update_yaxes(range=rango_y, row=row, col=1)

    fx0 = escalar(f(x0))
    m = escalar(df(x0))

    fig.add_trace(go.Scatter(x=[x0, x0], y=[0, fx0], mode="lines",
                             line=dict(color="#94a3b8", width=1, dash="dot"),
                             hoverinfo="skip", showlegend=False), **kw)
    fig.add_trace(go.Scatter(x=[0, x0], y=[fx0, fx0], mode="lines",
                             line=dict(color="#94a3b8", width=1, dash="dot"),
                             hoverinfo="skip", showlegend=False), **kw)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name=f"f(x) = {label}",
        line=dict(color=COLOR_CURVA, width=2.5),
        customdata=[[tag]] * len(xs),
        hovertemplate="x=%{x:.3f}<br>f(x)=%{y:.3f}<extra></extra>",
    ), **kw)

    xt = np.linspace(x0 - 0.9, x0 + 0.9, 10)
    fig.add_trace(go.Scatter(
        x=xt, y=fx0 + m * (xt - x0), mode="lines", name=f"tangente (m = {m:.3f})",
        line=dict(color=COLOR_TAN, width=3), hoverinfo="skip",
    ), **kw)
    fig.add_trace(go.Scatter(
        x=[x0], y=[fx0], mode="markers", name=f"x₀ = {x0:.3f}",
        marker=dict(size=12, color=COLOR_TAN,
                    line=dict(width=1.5, color="white")),
        hoverinfo="skip",
    ), **kw)
    fig.add_annotation(
        x=x0, y=fx0, text=f"f′(x₀) = {m:.3f}",
        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.2,
        ax=34, ay=-46, font=dict(size=12, color=COLOR_TAN), **ann)
    return fx0, m


def function_figure(f_key, g_key, x0, y0):
    """Gráfico de Cálculo 1: y = f(x) con la recta tangente en x₀."""
    misma = f_key == g_key
    if misma:
        fig = go.Figure()
        fx0, m = _agregar_funcion(fig, f_key, x0, "f")
        signo = "+" if m >= 0 else "−"
        fig.update_layout(title=dict(
            text=(f"<b>f(x) = {ACTIVATIONS[f_key]['label']}</b>  ·  "
                  f"tangente en x₀ = {x0:.3f}<br>"
                  f"<span style='font-size:12px;color:#94a3b8'>"
                  f"y = {fx0:.3f} {signo} {abs(m):.3f}·(x − {x0:.3f})</span>"),
            font=dict(size=15)))
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=-0.2, showarrow=False,
            xanchor="center",
            text=("La pendiente f′(x₀) es el factor de escala local: |f′|>1 "
                  "expande, |f′|<1 comprime."),
            font=dict(size=12, color="#94a3b8"),
        )
        alto = 580
    else:
        fig = make_subplots(
            rows=2, cols=1, vertical_spacing=0.16,
            subplot_titles=(f"<b>f (eje x) = {ACTIVATIONS[f_key]['label']}</b>  ·  "
                            f"tangente en x₀ = {x0:.3f}",
                            f"<b>g (eje y) = {ACTIVATIONS[g_key]['label']}</b>  ·  "
                            f"tangente en y₀ = {y0:.3f}"))
        _agregar_funcion(fig, f_key, x0, "f", row=1)
        _agregar_funcion(fig, g_key, y0, "g", row=2)
        alto = 660

    fig.update_layout(
        template=TEMPLATE, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5,
                    xanchor="center", font=dict(size=12)),
        margin=dict(l=55, r=25, t=70, b=78), height=alto,
    )
    fig.update_xaxes(title="x", showgrid=True, gridcolor=GRID_EJE,
                     zeroline=True, zerolinecolor=EJE_CERO)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_EJE,
                     zeroline=True, zerolinecolor=EJE_CERO)
    return fig


def loss_curve_figure(losses, lr, epochs):
    """Pérdida (cross-entropy) por época en la transformación final (t = 1).

    Muestra cómo el descenso de gradiente reduce la pérdida y revela el efecto
    de los hiperparámetros (épocas y tasa de aprendizaje) sobre la convergencia.
    """
    xs = np.arange(1, len(losses) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=losses, mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor="rgba(31,41,55,0.07)",
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=losses, mode="lines", line=dict(color=COLOR_CURVA, width=2.2),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[losses[-1]], mode="markers",
        marker=dict(size=10, color=COLOR_TAN,
                    line=dict(width=1.5, color="white")),
        hoverinfo="skip", showlegend=False))
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text="<b>Pérdida de la recta por época</b>  ·  t = 1",
                   font=dict(size=13)),
        xaxis=dict(title="época", showgrid=False),
        yaxis=dict(title="pérdida", showgrid=True, gridcolor=GRID_EJE),
        margin=dict(l=55, r=15, t=42, b=42), height=210,
    )
    return fig