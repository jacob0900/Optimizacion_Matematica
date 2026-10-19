import math


def derivada(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2*h)


# Método de Newton
def newton(f, x0, tol1, tol2, nmax):
    
    x = x0
    
    print("k\t x_k\t\t f(x_k)\t\t error")
    
    for k in range(nmax):
        
        fx = f(x)
        
        # criterio 1
        if abs(fx) < tol1:
            print("Convergencia alcanzada por |f(x)|")
            return x
        
        dfx = derivada(f, x)
        
        if dfx == 0:
            print("La derivada es cero. El método falla.")
            return None
        
        x_new = x - fx/dfx
        
        error = abs(x_new - x)
        
        print(k, "\t", x, "\t", fx, "\t", error)
        
        # criterio 2
        if error < tol2:
            print("Convergencia alcanzada por |x_{k+1}-x_k|")
            return x_new
        
        x = x_new
    
    print("Se alcanzó el número máximo de iteraciones", k)
    return x


# Definimos la función
def f(x):
    return x**2-2


# Parámetros
x0 = -2
tol1 = 1e-8
tol2 = 1e-8
nmax = 50


# Ejecutar método
raiz = newton(f, x0, tol1, tol2, nmax)

print("\nAproximación de la raíz:", raiz)