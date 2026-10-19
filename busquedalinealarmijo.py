import numpy as np

# ==========================================================
# FUNCIÓN OBJETIVO GENERALIZADA
# ==========================================================

def f(x):
    return (x[0]**2) + (x[0]*x[2]) + x[1]**2 + 2*x[2]**2 + x[0]*x[1] + 5*x[0]


# ==========================================================
# GRADIENTE GENERALIZADO
# ==========================================================

def grad(x):
    g1 = 2*x[0] + x[2] + x[1] + 5
    g2 = 2*x[1] + x[0]
    g3 = x[0] + 4*x[2]
    return np.array([g1, g2, g3])


# ==========================================================
# BÚSQUEDA LINEAL DE ARMIJO
# ==========================================================

def armijo(xk, dk):
    alpha = 1
    tau = 0.5
    c1 = 1e-4
    gk = grad(xk)

    while f(xk + alpha*dk) > f(xk) + c1*alpha*np.dot(gk, dk):
        alpha = tau * alpha
    return alpha




def conjugate_gradient(x0, tol=1e-8, n_iter=1000):
    xk = x0.copy()
    gk = grad(xk)
    dk = -gk

    trayectoria = [xk]  # guardamos todos los iterados

    for k in range(n_iter):
        if np.linalg.norm(gk) < tol:
            print("Convergió en", k, "iteraciones")
            break

        alpha = armijo(xk, dk)
        x_next = xk + alpha*dk
        g_next = grad(x_next)

        beta = np.dot(g_next, g_next) / np.dot(gk, gk)
        dk = -g_next + beta*dk

        xk = x_next
        gk = g_next
        trayectoria.append(xk)

    return xk, trayectoria




x0 = np.array([-4.0, 2.0, 2.0])  # vector inicial en R^3
x_star, trayectoria = conjugate_gradient(x0)

print("=================================")
print("x* =", x_star)
print("f(x*) =", f(x_star))


print("=================================")
print("Norma ||x_k - x*|| en cada iteración:")
for k, xk in enumerate(trayectoria):
    norma = np.linalg.norm(xk - x_star)
    print(f"Iter {k}: ||x_k - x*|| = {norma}")
