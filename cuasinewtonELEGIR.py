import numpy as np

# =====================================
# SELECCIÓN DE FUNCIÓN
# =====================================
def seleccionar_funcion(tipo="cuadratica"):
    
    if tipo == "cuadratica":
        print("Usando función CUADRÁTICA")
        
        Q = np.array([[4, 1],
                      [1, 3]])
        b = np.array([-1, -2])
        
        def f(x):
            return 0.5 * x.T @ Q @ x + b.T @ x
        
        def grad_f(x):
            return Q @ x + b
        
    elif tipo == "no_cuadratica":
        print("Usando función NO CUADRÁTICA")
        
        def f(x):
            return np.log(1 + x[0]**2 + x[1]**2) + x[0]**2
        
        def grad_f(x):
            denom = 1 + x[0]**2 + x[1]**2
            df_dx = (2*x[0])/denom + 2*x[0]
            df_dy = (2*x[1])/denom
            return np.array([df_dx, df_dy])
    
    else:
        raise ValueError("Usa 'cuadratica' o 'no_cuadratica'")
    
    return f, grad_f


# =====================================
# MÉTODO BFGS (SIN BÚSQUEDA LINEAL)
# =====================================
def bfgs(f, grad_f, x0, tol=1e-6, max_iter=100):
    n = len(x0)
    xk = x0.copy()
    Hk = np.eye(n)  # H0 = I
    
    for k in range(max_iter):
        gk = grad_f(xk)
        
        # Criterio de parada
        if np.linalg.norm(gk) < tol:
            print(f"Convergió en {k} iteraciones")
            break
        
        # Paso directo (alpha = 1)
        xk_new = xk - Hk @ gk
        
        # Definiciones del algoritmo
        sk = xk_new - xk              # d_k
        yk = grad_f(xk_new) - gk      # c_k
        
        # Evitar problemas numéricos
        if np.dot(yk, sk) <= 1e-12:
            print("Condición yk^T sk no positiva, se detiene")
            break
        
        rho_k = 1.0 / np.dot(yk, sk)
        
        I = np.eye(n)
        
        # Actualización BFGS
        Hk = (I - rho_k * np.outer(sk, yk)) @ Hk @ (I - rho_k * np.outer(yk, sk)) \
             + rho_k * np.outer(sk, sk)
        
        # Actualizar x
        xk = xk_new
        
        print(f"Iter {k}: x = {xk}, f(x) = {f(xk)}, ||grad|| = {np.linalg.norm(gk)}")
    
    return xk


# =====================================
# EJECUCIÓN
# =====================================
tipo_funcion = "cuadratica"  # cambia a "no_cuadratica"

f, grad_f = seleccionar_funcion(tipo_funcion)

x0 = np.array([2.0, 2.0])

sol = bfgs(f, grad_f, x0)

print("\nSolución aproximada:", sol)
print("Valor de la función:", f(sol))