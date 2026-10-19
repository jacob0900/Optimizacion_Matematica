import numpy as np
from scipy.optimize import minimize_scalar

# Función objetivo (no cuadrática)
def f(x):
    return np.log(1 + x[0]**2 + x[1]**2)

# Gradiente
def grad_f(x):
    denom = 1 + x[0]**2 + x[1]**2
    return np.array([2*x[0]/denom, 2*x[1]/denom])

# Parámetros
tol = 1e-8
nmax = 100

# Punto inicial
x = np.array([2.0, 2.0])

k = 0
grad = grad_f(x)

print("x inicial:", x)

while np.linalg.norm(grad) > tol and k < nmax:
    
    d = -grad  # dirección de descenso
    
    # 🔥 Búsqueda lineal exacta
    phi = lambda alpha: f(x + alpha * d)
    
    res = minimize_scalar(phi)  # minimiza phi(alpha)
    alpha = res.x
    
    # Actualización
    x = x + alpha * d
    
    grad = grad_f(x)
    k += 1

print("\nSolución aproximada:")
print(x)
print("Número de iteraciones:", k)
print("Norma del gradiente:", np.linalg.norm(grad))