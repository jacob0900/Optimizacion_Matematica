#Multiplicadores de Lagrange - OM

import numpy as np
from scipy.optimize import minimize

def f(x):
    return np.array([x[0]*2 + x[1]*2])

def gradf(x):
    return np.array([2*x[0], 
                      2*x[1]])


# Python usa >= 0
def restricciones_eq(x):
    return np.array([x[0] + x[1] - 18,
                     x[1]**(1/2) - x[0] - 4])                                          # Python usa >= 0

def restricciones_ineq(x):
    return np.array([x[0],
                     (x[1] - 17),
                     -(x[0] - 12)])

cons = [{'type': 'ineq','fun': restricciones_ineq},
        {'type': 'eq','fun': restricciones_eq}]

n = 2
cotas_var = [[None, None] for i in range(n)]
x0 = np.array([0 for i in range(n)], dtype=float)
#resultado = minimize(f, x0, constraints = cons, bounds = cotas_var, method = 'SLSQP', options = {'ftol': 1e-6, 'eps': 1e-8, 'maxiter': 1000})
resultado = minimize(f, x0, jac = gradf, constraints = cons, bounds = cotas_var, method = 'SLSQP', options = {'ftol': 1e-6, 'eps': 1e-8, 'maxiter': 1000}) #Usando gradientes
print(resultado)