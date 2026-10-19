import math

# derivada numérica
def derivada(f, x, h=1e-6):
    return (f(x+h) - f(x-h))/(2*h)


def newton(f, x0, tol1, tol2, nmax):

    x_prev = x0
    errores = []

    print("k\t x_k\t\t e_k=|x_k-x_{k-1}|\t p")

    for k in range(nmax):

        fx = f(x_prev)
        dfx = derivada(f, x_prev)

        if dfx == 0:
            print("Derivada cero. El método falla.")
            return None

        x_new = x_prev - fx/dfx

        # error aproximado empleando los x_k
        e_k = abs(x_new - x_prev)
        errores.append(e_k)

        # cálculo del orden p
        p = None
        if len(errores) >= 3:
            e_k = errores[-1]
            e_km1 = errores[-2]
            e_km2 = errores[-3]

            if e_km1 != 0 and e_km2 != 0:
                p = math.log(e_k/e_km1) / math.log(e_km1/e_km2)

        print(k, "\t", x_new, "\t", e_k, "\t", p)

        # criterios de parada
        if abs(f(x_new)) < tol1 or e_k < tol2:
            print("Convergencia alcanzada")
            return x_new

        x_prev = x_new

    print("Se alcanzó nmax")
    return x_prev


# función del ejercicio
def f(x):
    return x**2 - 2


# parámetros
x0 = -2
tol1 = 1e-6
tol2 = 1e-6
nmax = 50


raiz = newton(f, x0, tol1, tol2, nmax)

print("\nAproximación de la raíz:", raiz)