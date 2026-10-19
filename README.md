# Optimización Matemática

Implementaciones en **Python** de los métodos vistos en el curso de **Optimización Matemática**: métodos de Newton, gradiente, gradiente conjugado y cuasi-Newton (BFGS) para optimización sin restricciones, y multiplicadores de Lagrange, SLSQP, programación cuadrática secuencial (SQP) y método simplex para optimización con restricciones.

La mayoría de los métodos están programados desde cero con NumPy. Varios scripts los comparan con `scipy.optimize`.

## Contenido

### 1. Método de Newton en una variable

| Archivo | Contenido |
|---|---|
| `metodonewton.py` | Newton para hallar raíces con derivada numérica y dos criterios de parada: \|f(x)\| y \|xₖ₊₁ − xₖ\| |
| `metodonewton_modificado.py` | Newton con estimación del orden de convergencia *p* a partir de los errores sucesivos |
| `grdiente_prueba.py` | Newton 1D con tabla de iteraciones, error eₖ y orden de convergencia |

### 2. Método del gradiente (descenso más pronunciado)

| Archivo | Contenido |
|---|---|
| `METODO_GRADIENTE.PY` | Descenso del gradiente con búsqueda lineal de Armijo (*backtracking*) e historial de iteraciones |
| `gradienteexactaQ.py` | Gradiente con búsqueda lineal exacta (`minimize_scalar`) sobre una función no cuadrática |
| `metodogradiente_funcion_NOQ.py` | Variante con f(x) = ln(1 + x₁² + x₂²) y búsqueda lineal exacta |

### 3. Gradiente conjugado no lineal (Fletcher-Reeves)

| Archivo | Contenido |
|---|---|
| `METODO_GRADIENTE_CONJUGADO.PY` | Gradiente conjugado de Fletcher-Reeves con Armijo y reinicio cuando la dirección deja de ser de descenso |
| `armijoquadratica.py` | Gradiente conjugado sobre la cuadrática f(x) = ½ xᵀQx + bᵀx |
| `busquedalinealarmijo.py`, `final.py` | Gradiente conjugado en ℝ³ con la distancia ‖xₖ − x*‖ en cada iteración |
| `busquedalinealarmijogradiente.py`, `casoespecialGRAD.py` | Gradiente conjugado sobre una función cúbica, comparando con el punto x* = (13.5, 5) |

### 4. Método de Newton en varias variables

| Archivo | Contenido |
|---|---|
| `newtonquadratica.py` | Newton sobre una cuadrática (Hessiano constante): converge en una iteración |
| `testrosenbrock.py` | Gradiente, Hessiana y método de Newton para la función de Rosenbrock |

### 5. Métodos cuasi-Newton (BFGS)

| Archivo | Contenido |
|---|---|
| `CuasiNewton_BFGS.PY` | BFGS completo con *backtracking* de Armijo, salvaguarda de curvatura y reinicio de la aproximación del Hessiano inverso |
| `cuasinewtonELEGIR.py` | BFGS con paso unitario; permite elegir una función cuadrática o no cuadrática |
| `metodocuasi-newton.py` | BFGS modificado que usa la Hessiana como matriz inicial en lugar de la identidad (Rosenbrock) |

### 6. Optimización con SciPy

| Archivo | Contenido |
|---|---|
| `MINIMIZEGENERAL.py` | Plantilla general con `scipy.optimize.minimize` (BFGS) y una referencia de funciones de NumPy para escribir la función objetivo |
| `minimice.py` | `minimize` con gradiente analítico sobre una cuadrática |

### 7. Optimización con restricciones

| Archivo | Contenido |
|---|---|
| `Coubb_Douglas.py` | Maximización de una función de producción Cobb-Douglas con restricción presupuestaria (SLSQP). Óptimo: unas 256 unidades de trabajo y 81 de capital, con producción máxima de 15 360 |
| `Langrange.py` | El mismo problema con multiplicadores de Lagrange, sin gradientes analíticos |
| `codigo1404.py` | Restricción de igualdad sobre una esfera con SLSQP |
| `Codigo_Mixto.py` | Problema con restricciones de igualdad y de desigualdad a la vez |
| `METODO_QP.py` | **Programación cuadrática secuencial (SQP) programada desde cero** para el problema de Betts (5 variables, 2 restricciones no lineales): subproblema QP resuelto con el sistema KKT, función de mérito L1 y actualización BFGS de la Hessiana de la Lagrangiana |

### 8. Programación lineal

| Archivo | Contenido |
|---|---|
| `SIMPLEX.py` | Maximización lineal con dos restricciones de desigualdad usando `linprog` (método HiGHS) |

## Cómo ejecutarlo

Requisitos: **Python 3** con NumPy, SciPy y SymPy.

```bash
pip install numpy scipy sympy
```

Cada archivo es un script independiente:

```bash
python METODO_QP.py
```

Para probar otra función, cambia `f`, su gradiente y el punto inicial `x0` al inicio o al final de cada script.

## Autor

- **Jacobo Londoño**, [LinkedIn](https://www.linkedin.com/in/jacobo-londo%C3%B1o-baquero-a2b86733a/)
