import numpy as np

# Función objetivo
def f(x):
    x1, x2 = x
    return x1**2 + x2**3 - 6*x1*x2 + 3*x1 + 6*x2

# Gradiente
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2*x1 - 6*x2 + 3
    df_dx2 = 3*x2**2 - 6*x1 + 6
    return np.array([df_dx1, df_dx2])

# Armijo
def armijo(f, grad_f, xk, dk, alpha=1.0, rho=0.5, c=1e-4):
    while f(xk + alpha * dk) > f(xk) + c * alpha * np.dot(grad_f(xk), dk):
        alpha *= rho
    return alpha

# Gradiente Conjugado (Fletcher-Reeves)
def gradiente_conjugado_armijo(f, grad_f, x0, x_star, tol=1e-6, max_iter=50):
    xk = x0
    gk = grad_f(xk)
    dk = -gk

    print(f"{'Iter':<5} {'x':<12} {'y':<12} {'||xk - x*||':<15}")
    print("-"*45)

    for k in range(max_iter):
        # Norma del error
        error = np.linalg.norm(xk - x_star)

        print(f"{k:<5} {xk[0]:<12.6f} {xk[1]:<12.6f} {error:<15.6f}")

        if np.linalg.norm(gk) < tol:
            break

        # Paso Armijo
        alpha = armijo(f, grad_f, xk, dk)

        # Actualización
        xk_new = xk + alpha * dk
        gk_new = grad_f(xk_new)

        # Beta (Fletcher-Reeves)
        beta = np.dot(gk_new, gk_new) / np.dot(gk, gk)

        # Nueva dirección
        dk = -gk_new + beta * dk

        # Actualizar variables
        xk = xk_new
        gk = gk_new

# Parámetros
x0 = np.array([-3.0, 2.0])
x_star = np.array([13.5, 5.0])

# Ejecutar
gradiente_conjugado_armijo(f, grad_f, x0, x_star)