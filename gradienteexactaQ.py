import numpy as np
from scipy.optimize import minimize_scalar

# Función objetivo (NO cuadrática)
def f(x):
    return np.log(1 + x[0]**2 + x[1]**2) + x[0]**2

# Gradiente de la función
def grad_f(x):
    denom = 1 + x[0]**2 + x[1]**2
    df_dx = (2*x[0])/denom + 2*x[0]
    df_dy = (2*x[1])/denom
    return np.array([df_dx, df_dy])

# Método del gradiente con búsqueda lineal exacta
def gradiente_exacto(f, grad_f, x0, tol=1e-6, max_iter=100):
    xk = x0.copy()
    
    for k in range(max_iter):
        grad = grad_f(xk)
        
        # Criterio de parada
        if np.linalg.norm(grad) < tol:
            print(f"Convergió en {k} iteraciones")
            break
        
        dk = -grad  # dirección de descenso
        
        # Función de una variable: phi(alpha)
        phi = lambda alpha: f(xk + alpha * dk)
        
        # Búsqueda lineal exacta (numérica)
        res = minimize_scalar(phi)
        alpha_k = res.x
        
        # Actualización
        xk = xk + alpha_k * dk
        
        print(f"Iter {k}: x = {xk}, f(x) = {f(xk)}, ||grad|| = {np.linalg.norm(grad)}")
    
    return xk

# Punto inicial
x0 = np.array([2.0, 2.0])

# Ejecutar
sol = gradiente_exacto(f, grad_f, x0)

print("\nSolución aproximada:", sol)
print("Valor de la función:", f(sol))