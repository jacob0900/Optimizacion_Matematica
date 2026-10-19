import numpy as np

def gradiente(x1, x2):
    g1 = 4*x1 + x2 - 1
    g2 = x1 + 3*x2 - 2
    return np.array([g1, g2])

def hessiano():
    return np.array([[4,1],
                     [1,3]])

def newton(x1, x2, tol=1e-6, max_iter=10):

    H = hessiano()

    print("k | x1 | x2 | ||grad||")
    print("-"*40)

    for k in range(max_iter):

        grad = gradiente(x1, x2)
        grad_norm = np.linalg.norm(grad)

        print(k, "|", x1, "|", x2, "|", grad_norm)

        if grad_norm < tol:
            print("\nGradiente menor que la tolerancia")
            break

        # direccion de Newton
        d = -np.linalg.solve(H, grad)

        # actualizar
        x1 = x1 + d[0]
        x2 = x2 + d[1]

    return x1, x2


# punto inicial
x1_0 = 0.00000001
x2_0 = 0.00000001

sol = newton(x1_0, x2_0)

print("\nMinimo encontrado:", sol)