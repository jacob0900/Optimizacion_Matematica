import numpy as np

def armijo_line_search(f, grad_f, x_k, d_k, alpha=1.0, tau=0.5, c1=1e-4):
    """
    Búsqueda lineal de Armijo (Backtracking Line Search)
    """
    f_xk = f(x_k)
    grad_fxk = grad_f(x_k)
    
    while f(x_k + alpha * d_k) > f_xk + c1 * alpha * np.dot(grad_fxk, d_k):
        alpha = tau * alpha
    
    return alpha


def gradient_descent_armijo(f, grad_f, x0, tol=1e-6, max_iter=10000,
                             alpha0=1.0, tau=0.5, c1=1e-4, verbose=True):
    """
    Método del Gradiente con Búsqueda Lineal de Armijo

    Parámetros:
    -----------
    f        : función objetivo f(x)
    grad_f   : gradiente de f, grad_f(x)
    x0       : punto inicial (numpy array)
    tol      : tolerancia para criterio de parada ||∇f(x_k)|| < tol
    max_iter : número máximo de iteraciones
    alpha0   : paso inicial para Armijo (default: 1.0)
    tau      : factor de reducción τ ∈ (0,1) (default: 0.5)
    c1       : constante de Armijo c1 ∈ (0,1) (default: 1e-4)
    verbose  : imprimir progreso (default: True)

    Retorna:
    --------
    x_k      : aproximación al mínimo
    historia  : dict con historial de iteraciones
    """

    x_k = x0.copy().astype(float)
    historia = {"x": [x_k.copy()], "f": [f(x_k)], "grad_norm": [], "alpha": []}

    if verbose:
        print(f"{'Iter':>5} | {'f(x_k)':>15} | {'||∇f(x_k)||':>14} | {'α_k':>10}")
        print("-" * 55)

    for k in range(max_iter):
        grad_k = grad_f(x_k)
        grad_norm = np.linalg.norm(grad_k)

        historia["grad_norm"].append(grad_norm)

        if verbose and (k % max(1, max_iter // 20) == 0 or grad_norm < tol):
            print(f"{k:>5} | {f(x_k):>15.8f} | {grad_norm:>14.8f} | {historia['alpha'][-1] if historia['alpha'] else alpha0:>10.6f}")

        # Criterio de parada
        if grad_norm < tol:
            if verbose:
                print(f"\n✓ Convergencia alcanzada en iteración {k}")
            break

        # Dirección de descenso: d_k = -∇f(x_k)
        d_k = -grad_k

        # Paso con Armijo
        alpha_k = armijo_line_search(f, grad_f, x_k, d_k, alpha=alpha0, tau=tau, c1=c1)

        # Actualización: x_{k+1} = x_k + α_k * d_k
        x_k = x_k + alpha_k * d_k

        historia["x"].append(x_k.copy())
        historia["f"].append(f(x_k))
        historia["alpha"].append(alpha_k)

    else:
        if verbose:
            print(f"\n⚠ Se alcanzó el máximo de iteraciones ({max_iter})")

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Resultado final:")
        print(f"  x*     = {x_k}")
        print(f"  f(x*)  = {f(x_k):.10f}")
        print(f"  ||∇f|| = {np.linalg.norm(grad_f(x_k)):.2e}")
        print(f"{'='*55}")

    return x_k, historia


# ── Ejemplos de uso ─────────────────────────────────────────────────────────────
if __name__ == "__main__":



    def f1(v):
        x,y=v
        return 6*(y-x**2)**2 + (1-x)**2

    def grad_f1(v):
        x,y=v
        return np.array(-24*(y-x**2)*x - 2*(1-x), 12+(y-x**2))

    x0 = np.array([0.0, 0.0])
    x_opt, hist = gradient_descent_armijo(f1, grad_f1, x0)

        
        