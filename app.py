"""Visualizador geométrico de funciones de activación y deformación espacial.

Panel lateral: datos (dataset, f, g) y HIPERPARÁMETROS del modelo (épocas,
tasa de aprendizaje, regularización C) — la recta se entrena por descenso de
gradiente, de modo que esos valores cambian de verdad la convergencia.

Módulos principales:
  1. Plano 2D animado por FRAMES de Plotly (sin update recurrente del DOM).
  2. Gráfica y = f(x) con la recta tangente en el punto seleccionado.
  3. Curva de pérdida por época + chips de métricas (información concisa).

Ejecutar:  .venv/bin/python app.py   →  http://127.0.0.1:8050
"""

import numpy as np
from dash import Dash, Input, Output, State, ctx, dcc, html

from activations import ACTIVATIONS
from datasets import DATASETS, get_dataset
from deformation import limites_deformacion, transform_points
from figures import (
    function_figure,
    loss_curve_figure,
    plane_figure_frames,
)
from metrics import UMBRAL_SEPARABLE, interpretacion
from model import gradient_descent_logistic

N_FRAMES = 21   # pasos de t para la animación

# Valores por defecto de los hiperparámetros
DEF_EPOCHS = 100
DEF_LR = 0.5
DEF_C_LOG = -3   # C = 10^DEF_C_LOG = 0.001


# ---------------------------------------------------------------- cachés

_RES_CACHE = {}
_LIM_CACHE = {}


def _resultados(ds, fk, gk, epochs, lr, C):
    """(t_values, resultados) con (accuracy, w, b, loss) por frame, cacheado.

    También devuelve el historial de pérdida de la transformación final (t=1).
    """
    clave = (ds, fk, gk, epochs, lr, C)
    if clave not in _RES_CACHE:
        X, y = get_dataset(ds)
        f, g = ACTIVATIONS[fk]["f"], ACTIVATIONS[gk]["f"]
        ts = np.linspace(0, 1, N_FRAMES)
        res = []
        loss_final = None
        for t in ts:
            U = transform_points(X, t, f, g)
            w, b, losses, acc = gradient_descent_logistic(U, y, epochs, lr, C)
            res.append((acc, w, b, float(losses[-1])))
            if t >= 1.0:
                loss_final = losses
        _RES_CACHE[clave] = (ts, res, loss_final)
    return _RES_CACHE[clave]


def limites_fijos(ds, fk, gk):
    clave = (ds, fk, gk)
    if clave not in _LIM_CACHE:
        X, _ = get_dataset(ds)
        _LIM_CACHE[clave] = limites_deformacion(
            X, ACTIVATIONS[fk]["f"], ACTIVATIONS[gk]["f"])
    return _LIM_CACHE[clave]


# ---------------------------------------------------------------- layout

app = Dash(__name__, title="Deformación espacial en IA")


def _opciones(registro):
    return [{"label": v["label"], "value": k} for k, v in registro.items()]


def _selector(etiqueta, id_, opciones, valor):
    return html.Div(className="campo", children=[
        html.Label(etiqueta),
        dcc.Dropdown(id=id_, options=opciones, value=valor,
                     clearable=False, searchable=False),
    ])


def _slider_campo(etiqueta, id_, min_, max_, step, valor, fmt=None):
    fmt = fmt or (lambda v: f"{v:.2f}")
    return html.Div(className="campo", children=[
        html.Div(className="campo-titulo", children=[
            html.Label(etiqueta),
            html.Span(fmt(valor), id=f"{id_}-val", className="valor"),
        ]),
        dcc.Slider(id=id_, min=min_, max=max_, step=step, value=valor,
                   marks=None, tooltip={"placement": "bottom",
                                        "always_visible": True}),
    ])


def _chip(titulo, valor, color="#1f2937"):
    return html.Div(className="tarjeta", children=[
        html.Div(titulo, className="tarjeta-titulo"),
        html.Div(valor, className="tarjeta-valor", style={"color": color}),
    ])


sidebar = html.Div(className="panel-lateral", children=[
    html.Div(className="bloque", children=[
        html.H3("Datos"),
        _selector("Dataset", "dd-dataset", _opciones(DATASETS), "circulos"),
        _selector("f (eje x)", "dd-f", _opciones(ACTIVATIONS), "cuadratica"),
        _selector("g (eje y)", "dd-g", _opciones(ACTIVATIONS), "cuadratica"),
    ]),
    html.Div(className="bloque", children=[
        html.H3("Modelo · descenso de gradiente"),
        _slider_campo("Épocas", "dd-epochs", 10, 300, 10, DEF_EPOCHS,
                      fmt=lambda v: f"{v:d}"),
        _slider_campo("Tasa de aprendizaje", "dd-lr", 0.05, 1.0, 0.05, DEF_LR),
        _slider_campo("Regularización C (L2)", "dd-C", -4, -1, 0.1, DEF_C_LOG,
                      fmt=lambda v: f"10^{v:d}"),
    ]),
    html.Div(className="bloque resumen", children=[
        html.H3("Resultado (t = 1)"),
        html.Div(id="resumen-modelo", className="resumen-chips"),
    ]),
])


app.layout = html.Div(className="contenedor", children=[
    html.Header(children=[
        html.H1("Deformación espacial en IA: el papel de f(x) y f′(x)"),
        html.P("La IA deforma el plano con funciones de una variable mientras "
               "una recta (la red) se entrena por descenso de gradiente para "
               "separar las clases. Ajusta los hiperparámetros en el panel "
               "lateral y observa la convergencia.",
               className="subtitulo"),
    ]),

    html.Div(className="app-shell", children=[
        sidebar,

        html.Div(className="main", children=[
            html.Div(className="leyenda", children=[
                html.Span(className="grupo", children=[
                    html.Span(className="chip chip-a"), "Clase A",
                ]),
                html.Span(className="grupo", children=[
                    html.Span(className="chip chip-b"), "Clase B",
                ]),
                html.Span("|", className="sep"),
                html.Span("Recta de la red:"),
                html.Span(className="grupo", children=[
                    html.Span(className="chip chip-fallo"), "no separa",
                ]),
                html.Span(className="grupo", children=[
                    html.Span(className="chip chip-parcial"), "mejora",
                ]),
                html.Span(className="grupo", children=[
                    html.Span(className="chip chip-sep"), "separa",
                ]),
                html.Span("|", className="sep"),
                html.Span(className="grupo", children=[
                    html.Span(className="chip chip-linea"),
                    "rejilla original (ejes fijos)",
                ]),
            ]),

            html.Div(className="fila", children=[
                dcc.Graph(id="plano", className="grafico",
                          config={"displayModeBar": False}),
                dcc.Graph(id="graf-funcion", className="grafico",
                          config={"displayModeBar": False}),
            ]),

            html.Div(className="fila panel-inferior", children=[
                html.Div(className="col-metricas", children=[
                    html.Div(className="tarjetas", id="panel-metricas"),
                ]),
                dcc.Graph(id="perdida", className="sparkline",
                          config={"displayModeBar": False}),
            ]),
        ]),
    ]),

    html.Footer(
        "f′(x) estira (|f′|>1) o comprime (|f′|<1) cada región: así una "
        "función de una variable vuelve separable un problema no lineal.",
        className="footer",
    ),

    dcc.Store(id="seleccion", data={"x0": 1.0, "y0": 1.5}),
])


# ---------------------------------------------------------------- callbacks

@app.callback(
    Output("plano", "figure"),
    Output("perdida", "figure"),
    Output("resumen-modelo", "children"),
    Input("dd-dataset", "value"),
    Input("dd-f", "value"),
    Input("dd-g", "value"),
    Input("dd-epochs", "value"),
    Input("dd-lr", "value"),
    Input("dd-C", "value"),
)
def actualizar_visuales(ds, fk, gk, epochs, lr, c_log):
    """Reconstruye las figuras solo al cambiar configuración o hiperparámetros.

    La deformación (slider/play) es nativa del navegador (frames de Plotly):
    sin updates recurrentes del DOM durante la animación.
    """
    C = 10.0 ** c_log
    X, y = get_dataset(ds)
    ts, resultados, loss_final = _resultados(ds, fk, gk, epochs, lr, C)
    lims = limites_fijos(ds, fk, gk)

    fig_plano = plane_figure_frames(X, y, fk, gk, ts, lims, resultados)
    fig_perdida = loss_curve_figure(loss_final, lr, epochs)

    acc1, w1, b1, loss1 = resultados[-1]
    color = "#16a34a" if acc1 >= UMBRAL_SEPARABLE else "#9ca3af"
    estado = "separadas" if acc1 >= UMBRAL_SEPARABLE else "no separadas"
    resumen = [
        _chip("Accuracy", f"{acc1:.0%}", color),
        _chip("Pérdida final", f"{loss1:.3f}"),
        _chip("Estado", estado, color),
    ]
    return fig_plano, fig_perdida, resumen


@app.callback(
    Output("graf-funcion", "figure"),
    Output("panel-metricas", "children"),
    Input("seleccion", "data"),
    Input("dd-f", "value"),
    Input("dd-g", "value"),
)
def actualizar_tangente(sel, fk, gk):
    sel = sel or {"x0": 1.0, "y0": 1.5}
    x0, y0 = float(sel["x0"]), float(sel["y0"])
    fig = function_figure(fk, gk, x0, y0)

    f = ACTIVATIONS[fk]["f"]
    df = ACTIVATIONS[fk]["df"]
    fx0 = float(f(x0))
    dfx0 = float(df(x0))
    estado, color_estado = interpretacion(dfx0)

    tarjetas = [
        _chip("x₀", f"{x0:.3f}"),
        _chip(f"f(x₀) — {ACTIVATIONS[fk]['label']}", f"{fx0:.4f}"),
        _chip("f′(x₀)", f"{dfx0:.4f}"),
        _chip("Región", estado, color_estado),
    ]
    if fk != gk:
        g = ACTIVATIONS[gk]["f"]
        dg = ACTIVATIONS[gk]["df"]
        gy0 = float(g(y0))
        dgy0 = float(dg(y0))
        estado_g, color_g = interpretacion(dgy0)
        tarjetas += [
            _chip(f"g(y₀) — {ACTIVATIONS[gk]['label']}", f"{gy0:.4f}"),
            _chip("g′(y₀)", f"{dgy0:.4f}"),
            _chip("Región (eje y)", estado_g, color_g),
        ]
    return fig, tarjetas


@app.callback(
    Output("seleccion", "data"),
    Input("plano", "hoverData"),
    Input("plano", "clickData"),
    Input("graf-funcion", "hoverData"),
    State("seleccion", "data"),
)
def actualizar_seleccion(hov_plano, click_plano, hov_funcion, sel):
    sel = dict(sel or {"x0": 1.0, "y0": 1.5})
    disparador = ctx.triggered_id
    if disparador == "graf-funcion" and hov_funcion:
        pt = hov_funcion["points"][0]
        tag = pt.get("customdata")
        eje = tag[0] if isinstance(tag, (list, tuple)) else tag
        if eje == "g":
            sel["y0"] = pt["x"]
        else:
            sel["x0"] = pt["x"]
    elif disparador == "plano":
        datos = click_plano or hov_plano
        if datos:
            cd = datos["points"][0].get("customdata")
            if cd:
                sel["x0"], sel["y0"] = float(cd[0]), float(cd[1])
    return sel


@app.callback(
    Output("dd-epochs-val", "children"),
    Input("dd-epochs", "value"),
)
def _f_epochs(v):
    return f"{v:d}"


@app.callback(
    Output("dd-lr-val", "children"),
    Input("dd-lr", "value"),
)
def _f_lr(v):
    return f"{v:.2f}"


@app.callback(
    Output("dd-C-val", "children"),
    Input("dd-C", "value"),
)
def _f_c(v):
    return f"10^{v:d}"


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)