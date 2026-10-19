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


def conjugate_gradient_armijo(f, grad_f, x0, tol=1e-6, max_iter=10000,
                               alpha0=1.0, tau=0.5, c1=1e-4, verbose=True):
    """
    Método del Gradiente Conjugado (Fletcher-Reeves) con Búsqueda Lineal de Armijo

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
    grad_k = grad_f(x_k)
    d_k = -grad_k  # Dirección inicial: descenso más pronunciado

    historia = {"x": [x_k.copy()], "f": [f(x_k)], "grad_norm": [], "alpha": [], "beta": []}

    if verbose:
        print(f"{'Iter':>5} | {'f(x_k)':>15} | {'||∇f(x_k)||':>14} | {'α_k':>10} | {'β_k':>10}")
        print("-" * 65)

    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad_k)
        historia["grad_norm"].append(grad_norm)

        if verbose and (k % max(1, max_iter // 20) == 0 or grad_norm < tol):
            alpha_prev = historia["alpha"][-1] if historia["alpha"] else alpha0
            beta_prev  = historia["beta"][-1]  if historia["beta"]  else 0.0
            print(f"{k:>5} | {f(x_k):>15.8f} | {grad_norm:>14.8f} | {alpha_prev:>10.6f} | {beta_prev:>10.6f}")

        # Criterio de parada
        if grad_norm < tol:
            if verbose:
                print(f"\n✓ Convergencia alcanzada en iteración {k}")
            break

        # Búsqueda lineal de Armijo sobre la dirección conjugada d_k
        alpha_k = armijo_line_search(f, grad_f, x_k, d_k, alpha=alpha0, tau=tau, c1=c1)

        # Actualización del punto
        x_k = x_k + alpha_k * d_k

        # Nuevo gradiente
        grad_k_new = grad_f(x_k)

        # ── Coeficiente de Fletcher-Reeves ──────────────────────────────────
        # β_k = ||∇f(x_{k+1})||² / ||∇f(x_k)||²
        beta_k = np.dot(grad_k_new, grad_k_new) / np.dot(grad_k, grad_k)

        # Reinicio si β_k < 0 o la dirección pierde descenso (Powell restart)
        if np.dot(-grad_k_new, d_k) >= 0:
            beta_k = 0.0  # Reinicia a gradiente puro

        # Nueva dirección conjugada: d_{k+1} = -∇f(x_{k+1}) + β_k * d_k
        d_k = -grad_k_new + beta_k * d_k
        grad_k = grad_k_new

        historia["x"].append(x_k.copy())
        historia["f"].append(f(x_k))
        historia["alpha"].append(alpha_k)
        historia["beta"].append(beta_k)

    else:
        if verbose:
            print(f"\n⚠ Se alcanzó el máximo de iteraciones ({max_iter})")

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Resultado final:")
        print(f"  x*     = {x_k}")
        print(f"  f(x*)  = {f(x_k):.10f}")
        print(f"  ||∇f|| = {np.linalg.norm(grad_f(x_k)):.2e}")
        print(f"{'='*65}")

    return x_k, historia


# ── Ejemplo de uso ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    def f1(v):
        x, y = v
        return np.exp(x)*(x**2-y**2+1)

    def grad_f1(v):
        x, y = v
        return np.array([np.exp(x)*(x**2-y**2+1),
                           np.exp(x)*(-2*y)])

    x0 = np.array([0.1, 0.0])
    x_opt, hist = conjugate_gradient_armijo(f1, grad_f1, x0)