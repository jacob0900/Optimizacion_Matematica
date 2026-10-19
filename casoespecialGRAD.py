import numpy as np

# Función
def f(x):
    x1, x2 = x
    return x1**2 + x2**3 - 6*x1*x2 + 3*x1 + 6*x2

# Gradiente
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2*x1 - 6*x2 + 3
    df_dx2 = 3*x2**2 - 6*x1 + 6
    return np.array([df_dx1, df_dx2], dtype=float)

# Búsqueda lineal simple (backtracking)
def line_search(f, x, d, grad, alpha=1.0, rho=0.5, c=1e-4):
    while f(x + alpha*d) > f(x) + c*alpha*np.dot(grad, d):
        alpha *= rho
    return alpha

# Método del gradiente conjugado (no lineal - Fletcher-Reeves)
def gradiente_conjugado(x0, x_star, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    g = grad_f(x)
    d = -g
    results = []

    for k in range(max_iter):
        # Norma pedida
        norma = np.linalg.norm(x - x_star)
        results.append(f"{k:<5} {x[0]:<12.6f} {x[1]:<12.6f} {norma:<15.6f}")

        # Criterio de parada
        if np.linalg.norm(g) < tol:
            break

        # Búsqueda lineal
        alpha = line_search(f, x, d, g)

        # Actualización
        x_new = x + alpha * d
        g_new = grad_f(x_new)

        # Beta (Fletcher-Reeves)
        beta = np.dot(g_new, g_new) / np.dot(g, g) if np.dot(g, g) > 0 else 0

        # Nueva dirección
        d = -g_new + beta * d

        # Preparar siguiente iteración
        x = x_new
        g = g_new

    # Imprimir tabla con las últimas 4 iteraciones
    print("Iter  x1          x2          ||xk - x*||")
    print("-" * 45)
    for line in results[-4:]:
        print(line)

    return x

# Datos del problema
x0 = [-3, 2]
x_star = np.array([13.5, 5])

# Ejecutar
sol = gradiente_conjugado(x0, x_star)

print("\nSolución aproximada:", sol)
