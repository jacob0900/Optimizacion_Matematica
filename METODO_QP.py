"""
================================================================================
Programación Cuadrática Secuencial (SQP) - Problema de Betts
================================================================================
 
    min  f(x) = (x1-1)^2 + (x1-x2)^2 + (x3-1)^2 + (x4-1)^4 + (x5-1)^6
 
    s.a. h1(x) = x4*x1^2 + sin(x4 - x5) - 2*sqrt(2) = 0
         h2(x) = x2 + x3^4*x4^2 - 8 - sqrt(2)        = 0
 
         -10 <= xi <= 10,  i = 1,...,5
 
El método aplica Newton a las condiciones KKT, resolviendo en cada iteración
un subproblema de programación cuadrática (QP):
 
    min  (1/2) d^T B_k d + ∇f(x_k)^T d
    s.a. Dh(x_k) d + h(x_k) = 0
 
La búsqueda lineal se realiza sobre la función de mérito:
    φ(x) = f(x) + ρ * Σ|hj(x)|
 
La Hessiana de la Lagrangiana se aproxima con actualizaciones BFGS.
 
Referencia: Diapositivas "Programación Secuencial Cuadrática (SQP)",
            Curso de Optimización Matemática.
================================================================================
"""
 
import numpy as np
 
 
# ==============================================================================
# DEFINICIÓN DEL PROBLEMA DE BETTS
# ==============================================================================
 
def f(x):
    """
    Función objetivo del Problema de Betts:
        f(x) = (x1-1)^2 + (x1-x2)^2 + (x3-1)^2 + (x4-1)^4 + (x5-1)^6
    """
    return ((x[0] - 1)**2
            + (x[0] - x[1])**2
            + (x[2] - 1)**2
            + (x[3] - 1)**4
            + (x[4] - 1)**6)
 
 
def grad_f(x):
    """
    Gradiente de f:
        ∂f/∂x1 =  2(x1 - 1) + 2(x1 - x2)
        ∂f/∂x2 = -2(x1 - x2)
        ∂f/∂x3 =  2(x3 - 1)
        ∂f/∂x4 =  4(x4 - 1)^3
        ∂f/∂x5 =  6(x5 - 1)^5
    """
    return np.array([
         2*(x[0] - 1) + 2*(x[0] - x[1]),   # ∂f/∂x1
        -2*(x[0] - x[1]),                    # ∂f/∂x2
         2*(x[2] - 1),                       # ∂f/∂x3
         4*(x[3] - 1)**3,                    # ∂f/∂x4
         6*(x[4] - 1)**5                     # ∂f/∂x5
    ])
 
 
def h(x):
    """
    Restricciones de igualdad del Problema de Betts:
        h1(x) = x4*x1^2 + sin(x4 - x5) - 2*sqrt(2)
        h2(x) = x2 + x3^4*x4^2 - 8 - sqrt(2)
    """
    return np.array([
        x[3]*x[0]**2 + np.sin(x[3] - x[4]) - 2*np.sqrt(2),
        x[1] + x[2]**4 * x[3]**2 - 8 - np.sqrt(2)
    ])
 
 
def grad_h(x):
    """
    Jacobiana de h (matriz 2x5):
 
        Dh(x) = [ ∂h1/∂x1  ∂h1/∂x2  ∂h1/∂x3  ∂h1/∂x4        ∂h1/∂x5      ]
                [ ∂h2/∂x1  ∂h2/∂x2  ∂h2/∂x3  ∂h2/∂x4        ∂h2/∂x5      ]
 
              = [ 2*x4*x1   0        0         x1^2+cos(x4-x5) -cos(x4-x5) ]
                [ 0         1        4*x3^3*x4^2  2*x3^4*x4     0           ]
    """
    c = np.cos(x[3] - x[4])
    return np.array([
        [2*x[3]*x[0],  0,  0,
         x[0]**2 + c,  -c],
        [0,  1,  4*x[2]**3 * x[3]**2,
         2*x[2]**4 * x[3],  0]
    ])
 
 
# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
 
def lagrangiano_gradiente(x, lam):
    """
    Gradiente de la Lagrangiana respecto a x:
        ∇_x L(x, λ) = ∇f(x) + Dh(x)^T λ
    """
    return grad_f(x) + grad_h(x).T @ lam
 
 
def funcion_merito(x, rho):
    """
    Función de mérito (penalización L1 exacta):
        φ(x) = f(x) + ρ * Σ|hj(x)|
 
    Balancea minimizar f y satisfacer las restricciones.
    Se elige ρ > ||λ̂||∞ para garantizar descenso.
    """
    return f(x) + rho * np.sum(np.abs(h(x)))
 
 
def resolver_subproblema_qp(x_k, B_k):
    """
    Resuelve el subproblema QP en cada iteración:
 
        min_d  (1/2) d^T B_k d + ∇f(x_k)^T d
        s.a.   Dh(x_k) d + h(x_k) = 0
 
    Mediante el sistema KKT de silla de montar (n+r) x (n+r):
 
        | B_k      Dh(x_k)^T | | d  |   | -∇f(x_k) |
        | Dh(x_k)  0         | | ν  | = | -h(x_k)  |
 
    Retorna
    -------
    d  : array (5,) - dirección de búsqueda
    nu : array (2,) - multiplicadores del subproblema QP (nueva estimación λ)
    """
    n     = len(x_k)          # n = 5
    Dh_k  = grad_h(x_k)       # (2, 5)
    r     = Dh_k.shape[0]     # r = 2
 
    # Sistema silla de montar (7x7 para este problema)
    KKT = np.block([
        [B_k,                   Dh_k.T            ],   # (5, 7)
        [Dh_k,  np.zeros((r, r))                  ]    # (2, 7)
    ])
 
    rhs = np.concatenate([-grad_f(x_k), -h(x_k)])      # (7,)
 
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
 
    return sol[:n], sol[n:]    # d (5,), nu (2,)
 
 
def busqueda_lineal_armijo(x_k, d_k, rho, lam_k,
                            alpha0=1.0, beta=0.5, sigma=1e-4, max_iter=50):
    """
    Búsqueda lineal con condición de Armijo sobre φ(x).
 
    Retroceso: α ← β * α hasta que φ(x_k + α*d_k) < φ(x_k) + σ*α*Dφ
    """
    alpha  = alpha0
    phi_0  = funcion_merito(x_k, rho)
    Dphi_0 = grad_f(x_k) @ d_k - rho * np.sum(np.abs(h(x_k)))
 
    for _ in range(max_iter):
        if funcion_merito(x_k + alpha * d_k, rho) <= phi_0 + sigma * alpha * Dphi_0:
            return alpha
        alpha *= beta
 
    return alpha
 
 
def actualizar_bfgs(B_k, delta_x, delta_g):
    """
    Actualización BFGS del Hessiano de la Lagrangiana:
 
        B_{k+1} = B_k + (c_k c_k^T)/(c_k^T d_k)
                       - (B_k d_k)(B_k d_k)^T / (d_k^T B_k d_k)
 
    donde d_k = x_{k+1} - x_k,  c_k = ∇_x L(x_{k+1},λ_{k+1}) - ∇_x L(x_k,λ_k)
    Mantiene B_k simétrica y definida positiva sin derivadas de segundo orden.
    """
    cd  = delta_g @ delta_x          # c_k^T d_k
    if cd <= 1e-10:
        return B_k                   # No actualizar si falla curvatura positiva
    Bd  = B_k @ delta_x
    return (B_k
            + np.outer(delta_g, delta_g) / cd
            - np.outer(Bd, Bd) / (delta_x @ Bd))
 
 
# ==============================================================================
# ALGORITMO SQP PRINCIPAL
# ==============================================================================
 
def sqp_betts(x0, lam0=None, tol=1e-6, max_iter=200, rho_factor=1.5,
              verbose=True):
    """
    Algoritmo SQP aplicado al Problema de Betts.
 
    ALGORITMO (según diapositivas del curso):
    ─────────────────────────────────────────
    0. Inicializar: x0, λ0, B0 (s.d.p.), ε, kmax
    1. Convergencia: ||∇_x L|| < ε  y  ||h|| < ε  →  PARAR
    2. Construir QP: evaluar ∇f(x_k), Dh(x_k), B_k
    3. Resolver QP:  obtener d_k y multiplicadores ν_k
    4. Búsqueda lineal: αk sobre φ(x) (Armijo)
    5. Actualizar: x_{k+1}, λ_{k+1}, B_{k+1} (BFGS)
    6. k ← k+1; ir al Paso 1
 
    Parámetros
    ----------
    x0         : array (5,)  - punto inicial
    lam0       : array (2,)  - multiplicadores iniciales (default: ceros)
    tol        : float       - tolerancia de convergencia
    max_iter   : int         - máximo de iteraciones
    rho_factor : float       - ρ = rho_factor * max|ν_k|
    verbose    : bool        - imprimir tabla de progreso
 
    Retorna
    -------
    dict con x*, λ*, f(x*), h(x*), iteraciones, convergencia, historial
    """
 
    # ── Paso 0: Inicialización ──────────────────────────────────────────────
    x_k   = np.array(x0, dtype=float)
    n     = len(x_k)                                   # n = 5
    lam_k = np.zeros(2) if lam0 is None else np.array(lam0, dtype=float)
    B_k   = np.eye(n)                                  # B0 = I (s.d.p.)
    history_gradL = []
    history_h     = []
    history_f     = []
 
    if verbose:
        print("=" * 72)
        print("  SQP — Problema de Betts")
        print("  min (x1-1)^2+(x1-x2)^2+(x3-1)^2+(x4-1)^4+(x5-1)^6")
        print("  s.a. h1(x)=0, h2(x)=0")
        print("=" * 72)
        print(f"{'Iter':>5}  {'||∇L||':>12}  {'||h||':>12}  {'f(x)':>14}  {'α':>8}")
        print("-" * 72)
 
    converged = False
 
    for k in range(max_iter):
 
        # ── Paso 1: Verificar convergencia ─────────────────────────────────
        gL      = lagrangiano_gradiente(x_k, lam_k)
        norm_gL = np.linalg.norm(gL)
        norm_h  = np.linalg.norm(h(x_k))
        fval    = f(x_k)
 
        history_gradL.append(norm_gL)
        history_h.append(norm_h)
        history_f.append(fval)
 
        if (norm_gL < tol and norm_h < tol) or (norm_h < 1e-12 and norm_gL < 1e-4):
            if verbose:
                print(f"{k:>5}  {norm_gL:>12.2e}  {norm_h:>12.2e}  {fval:>14.8f}  {'—':>8}")
            converged = True
            break
 
        # ── Pasos 2 & 3: Construir y resolver el subproblema QP ────────────
        d_k, nu_k = resolver_subproblema_qp(x_k, B_k)
 
        # ── Paso 4: Búsqueda lineal sobre φ(x) ─────────────────────────────
        rho     = rho_factor * max(np.max(np.abs(nu_k)), 1.0)
        alpha_k = busqueda_lineal_armijo(x_k, d_k, rho, lam_k)
 
        if verbose:
            print(f"{k:>5}  {norm_gL:>12.2e}  {norm_h:>12.2e}  {fval:>14.8f}  {alpha_k:>8.4f}")
 
        # Si el paso es prácticamente cero, no hay más progreso
        if alpha_k < 1e-14:
            if verbose:
                print(f"       [Paso mínimo alcanzado en iter {k}. Deteniendo.]")
            break
 
        # ── Paso 5: Actualizar x, λ y B (BFGS) ────────────────────────────
        x_new   = x_k   + alpha_k * d_k
        lam_new = lam_k + alpha_k * (nu_k - lam_k)   # λ_{k+1} = λ_k + α(ν-λ)
 
        delta_x = x_new - x_k
        delta_g = (lagrangiano_gradiente(x_new, lam_new)
                   - lagrangiano_gradiente(x_k,  lam_k))
 
        B_k   = actualizar_bfgs(B_k, delta_x, delta_g)
        x_k   = x_new
        lam_k = lam_new
 
    # ── Resultado final ─────────────────────────────────────────────────────
    if verbose:
        print("-" * 72)
        estado = "✓ CONVERGIÓ" if converged else "✗ MAX ITERACIONES alcanzadas"
        print(f"\n  Estado    : {estado}")
        print(f"  Iteraciones: {k+1}")
        print(f"\n  x*  = {x_k}")
        print(f"  λ*  = {lam_k}")
        print(f"\n  f(x*)     = {f(x_k):.10f}")
        print(f"  ||h(x*)||  = {np.linalg.norm(h(x_k)):.2e}   "
              f"(h1={h(x_k)[0]:.2e}, h2={h(x_k)[1]:.2e})")
        print("=" * 72)
 
    return {
        'x'          : x_k,
        'lam'        : lam_k,
        'f_val'      : f(x_k),
        'h_val'      : h(x_k),
        'iters'      : k + 1,
        'converged'  : converged,
        'history_gradL': history_gradL,
        'history_h'    : history_h,
        'history_f'    : history_f
    }
 
 
# ==============================================================================
# EJECUCIÓN
# ==============================================================================
 
if __name__ == "__main__":
 
    # ── Verificación: subproblema QP en x0 = (0,0,0,0,0) ──────────────────
    print("\n" + "=" * 72)
    print("  VERIFICACIÓN: Subproblema QP inicial en x0 = (0,0,0,0,0)")
    print("=" * 72)
 
    x0_vec = np.zeros(5)
    print(f"\n  ∇f(x0)  = {grad_f(x0_vec)}")
    print(f"  h(x0)   = {h(x0_vec)}")
    print(f"\n  H0 = B0 (Hessiana de f en x0, con λ0=0):")
    # Para lambda0=0, el hessiano del lagrangiano coincide con el de f
    # (calculado como diferencias finitas para verificar)
    eps = 1e-5
    H0  = np.zeros((5, 5))
    gf0 = grad_f(x0_vec)
    for i in range(5):
        ei        = np.zeros(5); ei[i] = eps
        H0[:, i]  = (grad_f(x0_vec + ei) - gf0) / eps
    H0 = 0.5 * (H0 + H0.T)   # simetrizar
    print(np.array2string(H0, precision=4, suppress_small=True,
                          prefix="  "))
    print(f"\n  Dh(x0)  =\n{grad_h(x0_vec)}")
 
    d0, nu0 = resolver_subproblema_qp(x0_vec, H0)
    print(f"\n  Solución del QP inicial:")
    print(f"    d0  = {d0}")
    print(f"    ν0  = {nu0}  (primera estimación de λ)")
 
    # ── Ejecución del SQP desde x0 = (0,0,0,0,0) ──────────────────────────
    print("\n\n")
    x0   = np.zeros(5)
    lam0 = np.zeros(2)
 
    resultado = sqp_betts(x0, lam0, tol=1e-8, max_iter=200, verbose=True)