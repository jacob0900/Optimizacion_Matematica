import numpy as np
from scipy.optimize import minimize
import sympy as sp

#---------------------------------------------------------
# Ejemplo optimización con restricciones de desigualdad
#---------------------------------------------------------

# Función objetivo
def f_objetivo(x):
    return (-x[0]-x[1]-x[2])

# Gradiente de la función objetivo
def grad_f(x):
    return np.array([-1, -1, -1])

# Restricción (scipy usa c(x) >= 0)
# Original: 36 - x1^2 - x2^2 <= 0
# Transformada: x1^2 + x2^2 - 36 >= 0
def restriccion(x):
    return np.array([x[0]**2 + x[1]**2 + x[2]**2 - 300])

# Jacobiano (gradiente de la restricción)


# Las restricciones se escriben como una n-tupla
# Cada restricción se define con un diccionario
# Si quiero usar con restriccion de igualdad uso eq en lugar de ineq
restricciones = ({'type': 'eq','fun': restriccion})

# Cotas para las variables (debe tener 3 elementos para x0)
cotas_var = [(None, None), (None, None), (None, None)]

# Punto inicial
x0 = np.array([0.0, 0.0, 0.0])

# Optimización (usa KKT internamente)
result = minimize(f_objetivo, x0, method='SLSQP',  constraints=restricciones, bounds=cotas_var)

# Resultados
print("Solución:", result.x)
print("Valor mínimo:", result.fun)
print("Valor de la restricción:", restriccion(result.x))
print(result)







# Definimos las variables
#x1, x2 = sp.symbols('x1 x2')

# Definimos las restricciones (ejemplo: h1 y h2)
#h1 = x1**2 + x2**2 - 1   # círculo unidad
#h2 = x1 + x2 - 0.5       # recta

# Vector de restricciones
#h = sp.Matrix([h1, h2])

# Vector de variables
#x = sp.Matrix([x1, x2])

# Jacobiano de las restricciones
#J_h = h.jacobian(x)