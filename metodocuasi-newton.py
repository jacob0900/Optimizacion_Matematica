import numpy as np

# =========================
# FUNCIÓN DE ROSENBROCK
# =========================
def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2


# =========================
# GRADIENTE
# =========================
def grad_f(x):
    return np.array([
        -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
        200*(x[1] - x[0]**2)
    ])


# =========================
# HESSIANA
# =========================
def hess_f(x):
    x1, x2 = x
    return np.array([
        [1200*x1**2 - 400*x2 + 2, -400*x1],
        [-400*x1, 200]
    ])


# =========================
# BFGS MODIFICADO (I → HESSIANA)
# =========================
def bfgs_modificado(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):

    xk = np.array(x0, dtype=float)
    n = len(xk)

    
    Hk = hess_f(xk) 

    for k in range(max_iter):

        grad_k = grad_f(xk)

        # criterio de parada
        if np.linalg.norm(grad_k) < tol:
            print(f"Convergió en iteración {k}")
            return xk

        # paso principal
        x_next = xk - Hk @ grad_k

        # diferencias
        d_k = x_next - xk
        grad_next = grad_f(x_next)
        c_k = grad_next - grad_k

        denom = c_k.T @ d_k
        if denom <= 0:
            print("Problema: c_k^T d_k <= 0")
            return xk

        rho_k = 1.0 / denom

       
        I = np.eye(n)  

        term1 = I - rho_k * np.outer(d_k, c_k)
        term2 = I - rho_k * np.outer(c_k, d_k)

        # actualización tipo BFGS (pero modificada)
        Hk = term1 @ Hk @ term2 + rho_k * np.outer(d_k, d_k)

        # actualizar punto
        xk = x_next

        print(f"Iter {k}: x = {xk}, ||grad|| = {np.linalg.norm(grad_k)}")

    print("Máximo de iteraciones alcanzado")
    return xk



x0 = np.array([-1.2, 1.0])

sol = bfgs_modificado(f, grad_f, hess_f, x0)

print("\nSolución aproximada:", sol)
print("Valor de f:", f(sol))