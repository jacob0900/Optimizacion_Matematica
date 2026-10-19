from scipy.optimize import linprog

# Coeficientes de la función objetivo (negativos porque linprog minimiza)
c = [-7, -10]

# Restricciones de desigualdad
A_ub = [
    [4, 5],   # 4x + 5y <= 200
    [6, 3]    # 6x + 3y <= 420
]
b_ub = [200,240]

# Límites de las variables (no negativas)
x_bounds = (0, None)
y_bounds = (0, None)

# Resolver el problema
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[x_bounds, y_bounds], method="highs")

# Resultados
print("x óptimo:", res.x[0])
print("y óptimo:", res.x[1])
print("Valor máximo de f(x,y):", -res.fun)
