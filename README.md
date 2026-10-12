# Visualizador Geométrico de Funciones de Activación y Deformación Espacial en IA

Aplicación web 100 % Python (Dash + Plotly) que demuestra de forma interactiva cómo el **Cálculo Diferencial de una variable** ($f$ y $f'$) y la **geometría vectorial 2D** permiten a un algoritmo de IA separar clases no lineales.

> 📖 Para el **porqué didáctico y el funcionamiento interno** (incluida la
> explicabilidad y la animación por *frames* sin updates del DOM), ver
> [`EXPLICACION.md`](./EXPLICACION.md).

La idea central: la IA deforma el plano punto a punto con funciones de activación $T(x,y) = (f(x), g(y))$ hasta que una **simple recta** pueda separar los datos. La **derivada** $f'(x)$ — la pendiente de la recta tangente — es la métrica que mide cuánto estira o comprime cada región del espacio.

## Base matemática

### Deformación del espacio

El parámetro $t \in [0, 1]$ anima la transición continua entre el espacio original y el transformado:

$$P(t) = (1-t) \begin{bmatrix} x \\ y \end{bmatrix} + t \begin{bmatrix} f(x) \\ g(y) \end{bmatrix}$$

Como la transformación es **componente a componente** ($u$ depende solo de $x$, $v$ solo de $y$), los ejes **no rotan**: las líneas de la cuadrícula siguen verticales/horizontales; lo único que cambia es su espaciado.

### El papel de la derivada $f'(x)$

Para dos puntos cercanos $x_1, x_2$, con $\Delta x = x_2 - x_1$:

$$\Delta u = f(x_2) - f(x_1) \approx f'(x) \cdot \Delta x$$

- **$|f'| > 1$:** la región se **expande** (la pendiente estira las distancias).
- **$|f'| < 1$:** la región se **comprime**.
- **$f' = 0$:** los puntos colapsan en un mismo valor.

La pendiente $f'(x)$ es, por tanto, el **factor de escala local** de la deformación.

### Qué aprende la red

La deformación $(f, g)$ es **fija** (la elige el usuario). La red aprende la recta

$$w_1 u + w_2 v + b = 0$$

entrenándola con **descenso de gradiente** sobre la pérdida *cross-entropy*
(+ regularización L2). Sus **hiperparámetros** — épocas, tasa de aprendizaje y
regularización $C$ — se configuran desde el **panel lateral** y afectan
visiblemente la convergencia (se observa en la curva de pérdida).

## Requisitos previos

- Python 3.10 o superior.
- `ffmpeg` (solo necesario para exportar animaciones **MP4**; las GIF funcionan sin él).
- Acceso a internet para instalar `dash` y `plotly`.

## Instalación y ejecución

```bash
# 1. Crear el entorno (reutiliza numpy/scikit-learn/matplotlib del sistema)
python3 -m venv --system-site-packages .venv

# 2. Instalar las dependencias de la web
.venv/bin/pip install -r requirements.txt   # o: pip install dash plotly

# 3. Arrancar la app
.venv/bin/python app.py

# 4. Abrir en el navegador
#    http://127.0.0.1:8050
```

## Cómo usar la app

La interfaz tiene un **panel lateral de configuración** y tres módulos visuales:

**Panel lateral**
- **Datos:** dataset y las funciones $f$ (eje x) y $g$ (eje y).
- **Modelo (descenso de gradiente):** épocas, tasa de aprendizaje y regularización $C$ (L2). Cambiarlos altera la convergencia de la recta.
- **Resultado (t = 1):** chips con accuracy, pérdida final y estado.

**Módulos**
1. **Plano de datos 2D** — cuadrícula deformada + puntos de las dos clases + la recta de la red. Usa el **deslizador t** y el botón **▶ Play** **integrados en la gráfica**: la animación corre en el navegador vía *frames* de Plotly, sin recargar el servidor en cada paso.
2. **Gráfica y = f(x)** — curva con su **recta tangente** en el punto seleccionado y su ecuación. Pasa el cursor sobre un punto del plano o sobre la curva para cambiar $x_0$.
3. **Curva de pérdida + métricas** — la pérdida por época muestra la convergencia en $t=1$; los chips de métricas resumen $x_0$, $f(x_0)$, $f'(x_0)$ y el estado de la región.

### Leyenda de la recta de la red

El color de la recta indica qué tan bien logra separar las clases:

- **Gris** — no separa (accuracy baja).
- **Naranja** — va mejorando.
- **Verde** — clases separadas (accuracy ≥ 90 %).

La **rejilla original punteada** que queda fija debajo de la deformada sirve como referencia visual de que los ejes no cambian.

## Funciones y datasets

### Funciones de activación ($f$ y $g$, seleccionables por eje)

| Función | $f(x)$ | $f'(x)$ | Efecto |
|---|---|---|---|
| Cuadrática | $x^2$ | $2x$ | Expande extremos, comprime el centro; **separa círculos** |
| Cúbica | $x^3$ | $3x^2$ | Impar: no separa círculos |
| Sigmoide | $\sigma(x)$ | $\sigma(1-\sigma)$ | Comprime todo a $(0,1)$ |
| Tanh | $\tanh(x)$ | $1-\tanh^2(x)$ | Comprime todo a $(-1,1)$ |
| ReLU | $\max(0,x)$ | escalón | Colapsa el semiplano negativo |
| Identidad | $x$ | $1$ | Sin deformación en ese eje |

### Datasets

| Dataset | Separabilidad | Propósito |
|---|---|---|
| **Ya separables (blobs)** | Sí, desde t=0 | Caso trivial: la red resuelve **sin deformar** los ejes |
| **Círculos concéntricos** | Con $x^2$ (cada círculo → recta $u+v=r^2$) | Caso no lineal canónico |
| **Bandas verticales** | Con $x^2$ en x + identidad en y | Deformar solo un eje |
| **XOR** | Imposible con mapas componente a componente | Contraejemplo didáctico |

> **Dato clave:** las funciones **impares o monótonas** (x³, σ, tanh) **no** separan los círculos concéntricos; solo $x^2$ (par, pliega el plano) lo consigue. Esto se observa en vivo en la app.

## Exportar una animación (MP4/GIF)

```bash
# MP4 con configuración por defecto (círculos + x²)
.venv/bin/python export_animation.py

# GIF
.venv/bin/python export_animation.py --out deformacion.gif

# Bandas: deformar solo el eje x
.venv/bin/python export_animation.py --dataset bandas --f cuadratica --g identidad --out bandas.mp4

# Parámetros
#   --dataset lineal|circulos|bandas|xor
#   --f / --g   cuadratica|cubica|sigmoide|tanh|relu|identidad
#   --frames N  --fps N   --out ARCHIVO.mp4|.gif
```

## Estructura del proyecto

```
calc/
├── app.py              # App Dash: panel lateral + 3 módulos + callbacks + cachés
├── activations.py      # Registro de funciones f → (f, f′, etiqueta)
├── datasets.py         # Generadores de datasets (semilla fija, reproducibles)
├── deformation.py      # P(t) y deformación de la cuadrícula; límites de ejes fijos
├── metrics.py          # Interpretación de f′ y umbral de separabilidad
├── model.py            # Recta por descenso de gradiente (épocas, lr, C)
├── figures.py          # Figuras Plotly: plano (frames), tangente, curva de pérdida
├── export_animation.py # CLI matplotlib → MP4/GIF
├── smoke_test.py       # Verificación automática de las tesis pedagógicas
├── requirements.txt    # dash, plotly, numpy, scikit-learn, matplotlib
└── assets/style.css    # Tema visual (panel lateral, tarjetas, leyenda)
```

## Verificación

```bash
.venv/bin/python smoke_test.py
```

El smoke test construye todas las figuras y comprueba las tesis centrales
(con la recta entrenada por **descenso de gradiente**):

- Círculos + $x^2$ alcanzan **100 %** de separabilidad en $t=1$ (cada círculo colapsa a la recta $u+v=r^2$).
- Círculos + $x^3$ (impar) **no** separan (≈ 62 %): solo $x^2$ pliega el plano.
- Blobs ya separables en $t=0$ (≥ 90 %).
- La pérdida desciende durante el entrenamiento.
- Las figuras animadas contienen los 20 frames + slider + play.

## Notas pedagógicas clave

- **Los ejes no rotan:** la transformación es componente a componente, así que solo cambia el espaciado de la cuadrícula.
- **La red aprende solo 3 números** ($w_1, w_2, b$); la deformación $(f, g)$ es fija.
- **La derivada $f'$ es la "palanca"** que estira o comprime regiones para volver separables problemas no lineales.