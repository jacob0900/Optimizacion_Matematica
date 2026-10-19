import numpy as np
from scipy.optimize import minimize

# ==================================================
# REFERENCIA DE CONSTANTES Y FUNCIONES ESPECIALES
# ==================================================
# CONSTANTES:
#   np.pi          → π ≈ 3.14159...
#   np.e           → e (Euler) ≈ 2.71828...
#   np.inf         → ∞ (infinito)
#
# POTENCIAS Y RAÍCES:
#   x**2           → x cuadrado
#   x**n           → x a la n
#   np.sqrt(x)     → raíz cuadrada √x
#   x**(1/3)       → raíz cúbica ∛x
#   x**(1/n)       → raíz n-ésima
#
# EXPONENCIAL Y LOGARITMOS:
#   np.exp(x)      → e^x
#   np.log(x)      → logaritmo natural ln(x)
#   np.log2(x)     → logaritmo base 2
#   np.log10(x)    → logaritmo base 10
#   np.log(x)/np.log(b) → logaritmo base b
#
# TRIGONOMÉTRICAS (ángulos en radianes):
#   np.sin(x)      → seno
#   np.cos(x)      → coseno
#   np.tan(x)      → tangente
#   np.arcsin(x)   → arcoseno
#   np.arccos(x)   → arcocoseno
#   np.arctan(x)   → arcotangente
#
# HIPERBÓLICAS:
#   np.sinh(x)     → seno hiperbólico
#   np.cosh(x)     → coseno hiperbólico
#   np.tanh(x)     → tangente hiperbólica
#
# OTRAS ÚTILES:
#   np.abs(x)      → valor absoluto |x|
#   np.sign(x)     → signo de x (-1, 0 o 1)
#   np.floor(x)    → redondeo hacia abajo
#   np.ceil(x)     → redondeo hacia arriba
#   np.round(x, n) → redondear a n decimales
# ==================================================


def f(x):
    """
    Función objetivo a minimizar.

    Parámetros:
    -----------
    x : array-like
        Vector de variables de decisión [x1, x2, x3, x4].

    Retorna:
    --------
    float
        Valor escalar de la función evaluada en x.

    Función:
        f(x1, x2, x3, x4) = x1² + 2·x2² + 3·x3² + x1·x2 - x3 + 3·x4

    Nota:
    -----
    Para cambiar el número de variables, modifica el desempaquetado
    de 'x' y la expresión matemática de retorno. El tamaño de x0
    debe coincidir con el número de variables definidas aquí.
    """
    # Desempaquetar el vector en variables individuales
    x1, x2, x3 = x

    # Evaluar y retornar la función objetivo
    return np.exp(x1) * np.sin(x2) + np.log(x3**2)


# ==================================================
# PUNTO INICIAL
# ==================================================
# Vector de inicio para el algoritmo de optimización.
# - Su tamaño define la dimensión del problema (aquí: 4 variables).
# - El algoritmo comenzará a buscar el mínimo desde este punto.
# - Cambiar estos valores puede afectar la velocidad de convergencia.
x0 = np.array([1, 1, 1])   # Punto inicial: x1=1, x2=1, x3=1, x4=1


# ==================================================
# PROCESO DE MINIMIZACIÓN
# ==================================================
# Se utiliza el método BFGS (Broyden–Fletcher–Goldfarb–Shanno),
# un algoritmo iterativo de optimización basado en gradientes que:
#   1. Estima el gradiente (dirección de mayor cambio) de la función.
#   2. Aproxima la matriz Hessiana (curvatura) para dar pasos eficientes.
#   3. Actualiza la posición iterativamente hasta encontrar un mínimo local.
#
# Parámetros de minimize():
#   f      → función objetivo a minimizar
#   x0     → punto de partida
#   method → algoritmo de optimización a usar
resultado = minimize(
    f,
    x0,
    method="BFGS"
)


# ==================================================
# RESULTADOS
# ==================================================
print("\n========== RESULTADOS ==========")

# Indica si el algoritmo convergió exitosamente a un mínimo
print("¿Convergió?:", resultado.success)

# Mensaje descriptivo del estado final del algoritmo
print("Mensaje:", resultado.message)

# Vector solución: valores de [x1, x2, x3, x4] que minimizan f
print("\nMinimizador encontrado:")
print(resultado.x)

# Valor mínimo de la función: f(x1*, x2*, x3*, x4*)
print("\nValor mínimo de f(x):")
print(resultado.fun)

# Número de iteraciones que necesitó el algoritmo para converger
print("\nIteraciones:", resultado.nit)