import numpy as np

# Definimos Q y b
Q = np.array([[4, 1],
              [1, 3]])

b = np.array([-1, -2])

# Función cuadrática
def f(x):
    return 0.5 * x.T @ Q @ x + b.T @ x

# Gradiente
def grad_f(x):
    return Q @ x + b

# Búsqueda lineal de Armijo
def armijo(f, grad_f, xk, dk, alpha0=1, rho=0.5, c=1e-4):
    alpha = alpha0
    fk = f(xk)
    grad_fk = grad_f(xk)
    
    while f(xk + alpha * dk) > fk + c * alpha * np.dot(grad_fk, dk):
        alpha *= rho
    
    return alpha

# Gradiente Conjugado con Armijo
def gradiente_conjugado_armijo(f, grad_f, x0, tol=1e-6, max_iter=100):
    xk = x0.copy()
    gk = grad_f(xk)
    dk = -gk
    
    for k in range(max_iter):
        if np.linalg.norm(gk) < tol:
            print(f"Convergió en {k} iteraciones")
            break
        
        # Paso con Armijo
        alpha_k = armijo(f, grad_f, xk, dk)
        
        # Actualización
        xk_new = xk + alpha_k * dk
        gk_new = grad_f(xk_new)
        
        # Beta (Fletcher-Reeves)
        beta_k = np.dot(gk_new, gk_new) / np.dot(gk, gk)
        
        # Nueva dirección conjugada
        dk = -gk_new + beta_k * dk
        
        # Actualizar variables
        xk = xk_new
        gk = gk_new
        
        print(f"Iter {k}: x = {xk}, f(x) = {f(xk)}, ||grad|| = {np.linalg.norm(gk)}")
    
    return xk

# Punto inicial
x0 = np.array([2.0, 1.0])

# Ejecutar
sol = gradiente_conjugado_armijo(f, grad_f, x0)

print("\nSolución aproximada:", sol)
print("Valor de la función:", f(sol))