import numpy as np

def f(x):
    x1, x2 = x
    return 100*(x2 - x1**2)**2 + (1 - x1)**2


def grad_f(x):
    x1, x2 = x
    return np.array([
        -400*x1*(x2 - x1**2) + 2*(x1 - 1),
        200*(x2 - x1**2)
    ])

def hess_f(x):
    x1, x2 = x
    return np.array([
        [1200*x1**2 - 400*x2 + 2, -400*x1],
        [-400*x1, 200]
    ])


def newton_con_hessiana(x0, tol=1e-6, max_iter=50):
    xk = np.array(x0, dtype=float)

    for k in range(max_iter):
        grad = grad_f(xk)

        # criterio de parada
        if np.linalg.norm(grad) < tol:
            print(f"Convergió en iteración {k}")
            return xk

        Hk = hess_f(xk)  

        # Resolver Hk * p = grad
        p = np.linalg.solve(Hk, grad)

        # actualización
        xk = xk - p

        print(f"Iter {k}: x = {xk}, ||grad|| = {np.linalg.norm(grad)}")

    print("Máximo de iteraciones alcanzado")
    return xk