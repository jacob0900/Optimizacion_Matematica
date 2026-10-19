import numpy as np
from scipy.optimize import minimize

# Ejemplo de optimización con restricciones de igualdad
def f_objetivo(x):
    f_objetivo = - ( 80*(x[0]**0.75)* (x[1]**0.25))
    return f_objetivo

# Restricciones de igualdad
cons = (
    {'type': 'eq', 'fun': lambda x: 243*x[0] + 256*x[1] - 82944},
    
)

# Cotas (sin límites)
cotas_var = [[0, None], [0, None]]

# Punto inicial
x0 = np.array([1, 1])

# Optimización usando SLSQP
resultado = minimize(f_objetivo, x0, constraints=cons, bounds=cotas_var, method='SLSQP')

# Mostrar resultados
print(resultado)
## DEBO AGRGAR LOS GRADIENTES DE LA FUNCION OBJETIVO
#  Y DE LAS RESTRICCIONES PARA MEJORAR LA OPTIMIZACION.   