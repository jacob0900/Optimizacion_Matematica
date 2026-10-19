import numpy as np
from scipy.optimize import minimize

#---------------------------------------------------------
# Ejemplo optimización Cobb-Douglas con restricción de igualdad
#---------------------------------------------------------

# Función objetivo (negativa porque minimize busca mínimo)
def f_objetivo(vars):
    x, y = vars
    return -80 * (x**(3/4)) * (y**(1/4))  # maximizar => minimizar el negativo

# Gradiente de la función objetivo
def grad_f(vars):
    x, y = vars
    df_dx = -80 * (3/4) * (x**(-1/4)) * (y**(1/4))
    df_dy = -80 * (1/4) * (x**(3/4)) * (y**(-3/4))
    return np.array([df_dx, df_dy])

# Restricción de igualdad: 243x + 256y - 82944 = 0
def restriccion_eq(vars):
    x, y = vars
    return 243*x + 256*y - 82944

# Jacobiano de la restricción
def jac_restriccion_eq(vars):
    return np.array([243.0, 256.0])

# Definición de la restricción como igualdad
restricciones = ({'type': 'eq','fun': restriccion_eq,'jac': jac_restriccion_eq})

# Cotas para las variables (no negativas)
cotas_var = [(0, None), (0, None)]

# Punto inicial (una suposición razonable)
x0 = np.array([100.0, 100.0])

# Optimización
result = minimize(f_objetivo, x0, method='SLSQP', jac=grad_f,
                  constraints=restricciones, bounds=cotas_var)

# Resultados
x_opt, y_opt = result.x
produccion_max = -result.fun  # recordar que usamos negativo

print("Unidades óptimas de trabajo (x):", x_opt)
print("Unidades óptimas de capital (y):", y_opt)
print("Producción máxima:", produccion_max)
print("Valor de la restricción:", restriccion_eq(result.x))
print(result)


# Conclusión:
# La combinación óptima de recursos es invertir aproximadamente 256 unidades de trabajo y 81 unidades de capital,
# lo que permite alcanzar una producción máxima cercana a 15,360 unidades. Esta solución cumple exactamente
# con la restricción presupuestaria de $82,944 y refleja que la mayor parte del gasto debe destinarse a mano
# de obra, dado su menor costo y mayor contribución marginal en la función de producción.


