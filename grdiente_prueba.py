import math

def newton_1d(f, df, x0, max_iter=100, tol=0):
    x      = float(x0)
    hist_x = [x]

    print(f"{'Iter':>5} | {'x':>20} | {'f(x)':>15} | {'|f(x)|':>15} | {'e_k':>15} | {'p':>10}")
    print("-" * 97)

    for k in range(max_iter):
        fx     = f(x)
        norm_f = abs(fx)

        if k >= 1:
            ek     = abs(hist_x[k] - hist_x[k-1])
            ek_str = f"{ek:.6e}"
        else:
            ek     = None
            ek_str = "        ---"

        if k >= 3:
            ek0 = abs(hist_x[k]   - hist_x[k-1])
            ek1 = abs(hist_x[k-1] - hist_x[k-2])
            ek2 = abs(hist_x[k-2] - hist_x[k-3])
            if ek0 > 0 and ek1 > 0 and ek2 > 0 and ek1 != ek2:
                p     = math.log(ek0 / ek1) / math.log(ek1 / ek2)
                p_str = f"{p:.4f}"
            else:
                p_str = "       ---"
        else:
            p_str = "       ---"

        print(f"{k:>5} | {x:>20.10f} | {fx:>15.6e} | {norm_f:>15.6e} | {ek_str:>15} | {p_str:>10}")

        if norm_f < tol:
            print(f"\nConvergio en {k} iteraciones  |f(x)| = {norm_f:.2e} < {tol}")
            return x, {"converged": True, "iterations": k, "norm_f": norm_f, "reason": "norm_f"}

        dfx = df(x)
        if dfx == 0:
            print("\nDerivada nula, el metodo falla.")
            return x, {"converged": False, "iterations": k, "norm_f": norm_f, "reason": "zero_derivative"}

        x = x - fx / dfx
        hist_x.append(x)

    norm_f = abs(f(x))
    print(f"\nMaximo de iteraciones ({max_iter}) alcanzado.  |f(x)| = {norm_f:.2e}")
    return x, {"converged": False, "iterations": max_iter, "norm_f": norm_f, "reason": "max_iter"}



print("=" * 97)
print("Ejemplo 1:  f(x) = x**2 - 2")
print("=" * 97)
sol, info = newton_1d(
    f  = lambda x: x**2 - 2,
    df = lambda x: 2 * x,
    x0 = 1.5
)
print(f"\nNumero de iteraciones para converger: {info['iterations']}")
print(f"Solucion: x* = {sol:.12f}")