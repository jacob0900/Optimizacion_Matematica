"""
Algoritmo BFGS (Broyden-Fletcher-Goldfarb-Shanno)
Búsqueda lineal: backtracking simple con condición de Armijo.
Sin condición de curvatura (Wolfe).
"""

import numpy as np


# ─────────────────────────────────────────────────────────
# Búsqueda lineal — backtracking (solo Armijo)
# ─────────────────────────────────────────────────────────

def backtracking(f, grad_f, xk, pk, alpha=1.0, rho=0.5, c=1e-4):
    """
    Reduce alpha hasta cumplir la condición de Armijo:

        f(xk + alpha·pk) ≤ f(xk) + c·alpha·(∇f(xk)ᵀ pk)

    Parámetros
    ----------
    f      : función objetivo
    grad_f : gradiente
    xk     : punto actual
    pk     : dirección de descenso
    alpha  : paso inicial       (default 1.0)
    rho    : factor reducción   (default 0.5)
    c      : constante Armijo   (default 1e-4)

    Retorna
    -------
    alpha  : tamaño de paso aceptado
    """
    fk    = f(xk)
    slope = grad_f(xk) @ pk      # derivada direccional < 0

    for _ in range(100):
        if f(xk + alpha * pk) <= fk + c * alpha * slope:
            return alpha
        alpha *= rho

    return alpha


# ─────────────────────────────────────────────────────────
# Algoritmo BFGS
# ─────────────────────────────────────────────────────────

def bfgs(f, grad_f, x0, tol=1e-6, max_iter=500, verbose=True):
    """
    Minimiza f(x) usando BFGS con backtracking (Armijo).

    Parámetros
    ----------
    f        : función objetivo  callable(x) -> float
    grad_f   : gradiente         callable(x) -> np.ndarray shape (n,)
    x0       : punto inicial     np.ndarray  shape (n,)
    tol      : tolerancia ‖∇f‖ ≤ tol
    max_iter : número máximo de iteraciones
    verbose  : muestra tabla de progreso

    Retorna
    -------
    dict:
      'x'        : solución encontrada
      'f'        : f(x*)
      'grad'     : ∇f(x*)
      'iters'    : iteraciones realizadas
      'historial': lista de (k, f_k, ‖∇f_k‖)
    """
    n  = len(x0)
    xk = np.array(x0, dtype=float)
    Hk = np.eye(n)          # H₀ = I (aproximación inicial del Hessiano inverso)

    historial = []

    if verbose:
        print(f"{'Iter':>5}  {'f(x)':>14}  {'‖∇f‖':>12}")
        print("─" * 38)

    for k in range(max_iter):
        gk     = grad_f(xk)
        norm_g = np.linalg.norm(gk)
        fk     = f(xk)

        historial.append((k, float(fk), float(norm_g)))

        if verbose:
            print(f"{k:>5}  {fk:>14.6f}  {norm_g:>12.2e}")

        # ── 1. Criterio de parada ───────────────────────────
        if norm_g <= tol:
            if verbose:
                print(f"\n✓ Convergencia en iteración {k}.")
            break

        # ── 2. Dirección de búsqueda  pk = -Hk · ∇f ────────
        pk = -Hk @ gk

        # Salvaguarda: si pk no baja, reiniciar H = I
        if pk @ gk >= 0:
            Hk = np.eye(n)
            pk = -gk

        # ── 3. Backtracking (Armijo) ─────────────────────────
        alpha = backtracking(f, grad_f, xk, pk)

        # ── 4. Actualizar posición ───────────────────────────
        xk1 = xk + alpha * pk

        # ── 5. Vectores de actualización ────────────────────
        sk = xk1 - xk               # sk = xk₊₁ − xk
        yk = grad_f(xk1) - gk       # yk = ∇f(xk₊₁) − ∇f(xk)

        # ── 6. Verificar curvatura mínima ────────────────────
        curvatura = float(yk @ sk)
        if curvatura <= 1e-15:
            xk = xk1
            continue                # saltar actualización; H no cambia

        # ── 7. Fórmula de actualización BFGS ────────────────
        #
        #   ρk    = 1 / (ykᵀ sk)
        #   Hk₊₁ = (I − ρk sk ykᵀ) Hk (I − ρk yk skᵀ) + ρk sk skᵀ
        #
        rho_k = 1.0 / curvatura
        I     = np.eye(n)
        A     = I - rho_k * np.outer(sk, yk)
        B     = I - rho_k * np.outer(yk, sk)
        Hk    = A @ Hk @ B + rho_k * np.outer(sk, sk)

        xk = xk1

    else:
        if verbose:
            print(f"\n⚠ Máximo de iteraciones ({max_iter}) alcanzado.")

    return {
        'x'        : xk,
        'f'        : float(f(xk)),
        'grad'     : grad_f(xk),
        'iters'    : k + 1,
        'historial': historial,
    }


# ─────────────────────────────────────────────────────────
# Ejemplos de uso
# ─────────────────────────────────────────────────────────

def rosenbrock(x):
    """(1−x)² + 100(y−x²)²   →  mínimo en (1, 1)"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
         200*(x[1] - x[0]**2)
    ])


def cuadratica(x):
    """0.5 xᵀ diag(1,2,3) x   →  mínimo en (0, 0, 0)"""
    A = np.diag([1.0, 2.0, 3.0])
    return 0.5 * x @ A @ x

def grad_cuadratica(x):
    return np.diag([1.0, 2.0, 3.0]) @ x


if __name__ == "__main__":

    sep = "=" * 50

  

    # ── Ejemplo 2: cuadrática 3D ───────────────────────────
    print(f"\n{sep}")
    print("  BFGS + Backtracking — Cuadratica 3D")
    print(sep)
    x0 = np.array([3.0, -2.0, 5.0])
    print(f"  x0 = {x0}   |   minimo esperado: (0, 0, 0)\n")

    r2 = bfgs(cuadratica, grad_cuadratica, x0, tol=1e-10)

    print(f"\n{'─'*50}")
    print(f"  x*       = {np.round(r2['x'], 10)}")
    print(f"  f(x*)    = {r2['f']:.2e}")
    print(f"  ||∇f||   = {np.linalg.norm(r2['grad']):.2e}")
    print(f"  iters    = {r2['iters']}")