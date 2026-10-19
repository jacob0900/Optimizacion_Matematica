import numpy as np
from scipy.optimize import minimize

# funcion
def f(x):
    x1, x2 = x
    return 0.5*(4*x1**2 + 2*x1*x2 + 3*x2**2) - x1 - 2*x2

# gradiente (para ayudar al optimizador)
def grad_f(x):
    x1, x2 = x
    g1 = 4*x1 + x2 - 1
    g2 = x1 + 3*x2 - 2
    return np.array([g1, g2])

# punto inicial
x0 = np.array([2.0, 2.0])

res = minimize(f, x0, jac=grad_f, method='BFGS')

print("Resultado de minimize:")
print("x* =", res.x)
print("f(x*) =", res.fun)
print("Iteraciones =", res.nit)