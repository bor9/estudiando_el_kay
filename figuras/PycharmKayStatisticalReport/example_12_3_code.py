import numpy as np

## x es el vector de datos de largo N (x.shape=(N,))
# número de parámetros
p = 2
# condiciones iniciales del LMMSE secuencial
theta_i = np.zeros((p, ))
M_i = 1e2 * np.eye(p)
# matriz de MSE mínimo
M = M_i
# estimador
hat_theta = theta_i
for n in np.arange(N):
    # fila n-ésima de la matriz de observación
    h = np.array([np.cos(2 * np.pi * f0 * n), np.sin(2 * np.pi * f0 * n)])
    # cálculo del vector ganancia
    K = (M @ h) / (var + h @ M @ h)
    # cálculo de la matriz de MSE mínimo
    M = (np.eye(p) - np.outer(K, h)) @ M
    # cálculo del estimador
    hat_theta = hat_theta + K * (x[n] - h @ hat_theta)
