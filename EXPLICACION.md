# Explicabilidad y funcionamiento del proyecto

> Documento técnico-didáctico que explica **qué** demuestra la herramienta y
> **cómo** lo hace internamente. Para instalación y uso rápido, ver
> [`README.md`](./README.md).

---

## 1. ¿Qué es y qué problema demuestra?

La aplicación es un **visualizador interactivo** de un concepto central del
*deep learning*: las redes neuronales no "ven" datos no lineales, pero pueden
**deformar el espacio de entrada** con funciones de una variable hasta que un
problema complicado se vuelve separable por una **recta**.

Formalmente, la red aplica una transformación componente a componente

$$T(x, y) = \begin{bmatrix} f(x) \\ g(y) \end{bmatrix}$$

donde $f$ y $g$ son funciones de activación del Cálculo de una variable.
El parámetro $t \in [0,1]$ anima la transición:

$$P(t) = (1-t)\begin{bmatrix}x\\y\end{bmatrix} + t\begin{bmatrix}f(x)\\g(y)\end{bmatrix}$$

El proyecto conecta **dos ramas** del conocimiento:

| Área | Qué aporta |
|---|---|
| **Cálculo Diferencial ($\mathbb{R}\to\mathbb{R}$)** | La función $f$ y su derivada $f'$ controlan la deformación |
| **Geometría vectorial 2D** | La recta $w_1u + w_2v + b = 0$ como el modelo que separa las clases |

---

## 2. El papel de la derivada (explicabilidad matemática)

Para dos puntos cercanos $x_1, x_2$ en el eje $x$, con $\Delta x = x_2 - x_1$:

$$\Delta u = f(x_2) - f(x_1) \approx f'(x)\cdot\Delta x$$

La derivada $f'(x)$ — la **pendiente de la recta tangente** — es el **factor de
escala local** de la deformación:

- **$|f'| > 1$** → la región se **expande** (las distancias crecen).
- **$|f'| < 1$** → la región se **comprime**.
- **$f' = 0$** → los puntos colapsan en un mismo valor.

La herramienta hace visible esta idea de tres maneras:
1. La **gráfica y = f(x)** dibuja la tangente en el punto seleccionado y muestra
   su pendiente $f'(x_0)$, con guías de proyección.
2. Los **chips de métricas** traducen la derivada a lenguaje: *"Expandiendo
   datos ×2.34"* o *"Comprimiendo datos ×0.31"*.
3. Una **nota breve** en la gráfica aclara el propósito: la pendiente $f'(x_0)$
   es el factor de escala local (|f′|>1 expande, |f′|<1 comprime).

### División de roles (clave para la explicabilidad)

| Pieza | ¿Se aprende? | Qué hace |
|---|---|---|
| Deformación $(f, g)$ | ❌ **Fija** (la elige el usuario) | Estira/comprime regiones del espacio |
| Recta $w_1u + w_2v + b = 0$ | ✅ **La red la entrena** por descenso de gradiente | Separa las clases en el espacio ya deformado |

La app enfatiza que **los ejes no rotan**: como la transformación es
componente a componente ($u$ depende solo de $x$, $v$ solo de $y$), las líneas
de la cuadrícula permanecen verticales/horizontales — solo cambia su
**espaciado**. Una rejilla original punteada y fija queda como referencia.

---

## 3. Explicabilidad visual: panel lateral y módulos

### Panel lateral (configuración)

- **Datos:** dataset y funciones $f$ (eje x) y $g$ (eje y).
- **Modelo · descenso de gradiente:** hiperparámetros **épocas**, **tasa de
  aprendizaje** y **regularización $C$** (L2), cada uno con slider.
- **Resultado (t = 1):** chips con accuracy, pérdida final y estado.

### 3.1 Plano de datos (espacio de entrada → transformado)

- Cuadrícula que se deforma según $f$ y $g$, y los puntos de las dos clases
  (con halo blanco para legibilidad).
- La **recta de la red** cambia de color según su acierto: **gris** (no separa)
  → **naranja** (mejora) → **verde** (separa). Además, es **sólida cuando
  separa** y **punteada cuando aún busca**.
- Un texto de estado muestra en cada instante la ecuación de la recta, la
  accuracy y la pérdida: *"t = 0.45 | 2.90u + 2.65v − … = 0 | accuracy 76% |
  pérdida 0.35 → buscando…"*.

### 3.2 Gráfica de la función y la recta tangente

- Curva $y = f(x)$, punto $(x_0, f(x_0))$ y su **recta tangente** con la
  ecuación $y = f(x_0) + f'(x_0)(x - x_0)$.
- Hover sobre un punto del plano o sobre la curva desplaza $x_0$.

### 3.3 Curva de pérdida y chips de métricas

- **Curva de pérdida por época** (en $t=1$): muestra la convergencia del
  descenso de gradiente y revela el efecto de los hiperparámetros (épocas
  insuficientes = subajuste; tasa alta = divergencia; $C$ = regularización).
- **Chips de métricas:** $x_0$, $f(x_0)$, $f'(x_0)$ y el estado de la región,
  con información concisa (sin saturar la pantalla).

---

## 4. Funcionamiento técnico

### 4.1 Stack

- **Interfaz web:** Dash 4 + Plotly 6 (100 % Python, sin JavaScript).
- **Entrenamiento del modelo:** NumPy (descenso de gradiente propio en
  `model.py`); scikit-learn se conserva solo para la exportación y los tests.
- **Exportación:** Matplotlib + FFmpeg (MP4) / Pillow (GIF).
- **Entorno:** `.venv` con `--system-site-packages`.

### 4.2 Optimización clave: animación por frames (sin update recurrente del DOM)

En la primera versión, cada movimiento del deslizador disparaba un callback de
Dash que **reconstruía y reenviaba la figura completa** al navegador (re-render
recurrente del DOM, lento). La versión actual lo resuelve con **frames de
Plotly**:

1. Al cambiar dataset, funciones o **hiperparámetros**, el callback
   `actualizar_visuales` entrena la recta por descenso de gradiente en cada uno
   de los 21 instantes $t$ y empaqueta la figura con **20 frames**:
   `go.Figure(data=...); fig.frames = [go.Frame(...)]`.
2. El **slider y el botón play/pausa son nativos de Plotly** (propiedades
   `sliders` y `updatemenus`), de modo que la animación corre **en el
   navegador** sin volver a tocar el servidor.
3. El texto de estado dependiente de $t$ (ecuación de la recta + accuracy +
   pérdida) se empaqueta como **anotación de layout por frame**.
4. Los límites de los ejes son **fijos** (calculados una vez con
   `limites_deformacion`) para que la geometría no "salte".

Con esto, **el servidor solo recalcula cuando cambia la configuración**, no por
cada tick de la animación.

### 4.3 Entrenamiento del modelo (`model.py`)

La recta se entrena **visiblemente** con descenso de gradiente:

$$L(w,b) = \text{cross-entropy}\big(y, \sigma(w\cdot u + b)\big) + C\,\|w\|^2$$

- Cada **época** da un paso $w \leftarrow w - \eta\,\nabla_w L$, $b \leftarrow b - \eta\,\partial_b L$.
- Hiperparámetros expuestos: **épocas** (número de pasos), **tasa de
  aprendizaje** $\eta$ y **regularización** $C$ (L2).
- Las entradas se **escalan** internamente para que $\eta$ sea intuitiva; $w$
  se devuelve en la escala original.
- Se aplica *gradient clipping* y se corta si la pérdida diverge (tasas altas).

Esto hace que los hiperparámetros sean **significativos y observables** a
través de la curva de pérdida, a diferencia de ajustar la recta "de golpe".

### 4.4 Flujo de datos

```
[Panel lateral: dataset, f, g, épocas, lr, C]
        │  (callback actualizar_visuales)
        ▼
get_dataset(ds) → X, y                       (cacheado)
_resultados(ds,f,g,épocas,lr,C) → [(acc,w,b,loss) por t]   (cacheado)
limites_fijos(ds,f,g) → lims                 (cacheado)
        ▼
plane_figure_frames(...)  →  20 frames + slider + play (cliente)
loss_curve_figure(...)    →  pérdida por época en t=1
resumen-modelo            →  chips accuracy / pérdida / estado
        ▼
[Plano]  ← hover/clic →  actualizar_seleccion →  Store "seleccion"
[Graf. tangente + chips] ← actualizar_tangente (on demand, no recurrente)
```

### 4.5 Cachés

- `get_dataset` (`functools.lru_cache`): los datasets son deterministas (semilla
  fija), se generan una sola vez.
- `_resultados` y `limites_fijos` (dicts): evitan re-entrenar y recalcular los
  límites para cada combinación de (dataset, f, g, épocas, lr, C).

---

## 5. Estructura del proyecto

```
calc/
├── app.py              # App Dash: panel lateral + 3 módulos + callbacks + cachés
├── activations.py      # Registro de funciones f → (f, f′, etiqueta)
├── datasets.py         # Generadores de datasets (semilla fija)
├── deformation.py      # P(t), deformación de la cuadrícula, límites fijos
├── metrics.py          # Interpretación de f′ y umbral de separabilidad
├── model.py            # Recta por descenso de gradiente (épocas, lr, C)
├── figures.py          # Figuras Plotly: plano (frames), tangente, pérdida
├── export_animation.py # CLI matplotlib → MP4/GIF
├── smoke_test.py       # Verificación automática de las tesis pedagógicas
├── requirements.txt
└── assets/style.css    # Tema visual (panel lateral, tarjetas, leyenda)
```

---

## 6. Verificación de las tesis

`smoke_test.py` construye todas las figuras y comprueba numéricamente que:

| Afirmación | Resultado |
|---|---|
| Círculos + $x^2$ separables en $t=1$ (descenso de gradiente) | 100 % (círculo → recta $u+v=r^2$) |
| Círculos + $x^3$ (impar) no separables con GD | 62.5 % |
| Blobs ya separables en $t=0$ con GD | ≥ 90 % |
| La pérdida desciende durante el entrenamiento | sí |
| Figuras de frames: 20 frames + slider + play | OK |

---

## 7. Ideas pedagógicas clave

1. **La IA no "hace magia"**: deforma el espacio con funciones de cálculo hasta
   volver lineal lo que no lo era.
2. **La pendiente $f'$ es la palanca** que estira (|f′|>1) o comprime (|f′|<1)
   regiones; la recta tangente es su manifestación geométrica.
3. **Los ejes son inmutables**: la transformación es componente a componente.
4. **La red solo aprende 3 números** ($w_1, w_2, b$); la deformación $(f, g)$
   es fija y se elige para aislar el concepto de Cálculo 1.
5. El caso "Ya separables" muestra que no siempre hace falta deformar; el
   dataset XOR muestra que hay problemas que **ningún** mapa componente a
   componente monótono puede resolver (motiva redes con más de una capa o
   mezcla de coordenadas).