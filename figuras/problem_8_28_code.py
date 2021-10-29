import numpy as np
from scipy import signal, linalg, optimize

# hd es la señal deseada dada, con hd.shape = (N, )
# el filtro a diseñar tiene función de transferencia con q coeficientes
# en el numerador y p coeficientes en el denominador, con p = q + 1.

# función no lineal en a a optimizar: hd^T @ A @ (A^T @ A)^{-1} @ A^T @ hd
def ja(a):
    c = np.concatenate(([a[0]], np.zeros((N - (p + 1), ))))
    r = np.concatenate((a, [1], np.zeros((N - (p + 1), ))))
    AT = linalg.toeplitz(c, r)
    return hd @ AT.T @ linalg.inv(AT @ AT.T) @ AT @ hd

## cálculo de a empleando scipy.optimize.minimize
# a0: valor inicial para el algoritmo de optimización
a0 = np.zeros((p, ))
a0[0] = 0.2
res = optimize.minimize(ja, a0)
a = res.x
a = np.concatenate(([1], a[::-1]))

## cálculo de b
# cálculo de la matriz G empleando el valor de a obtenido en la optimización
delta = np.zeros((N, ))
delta[0] = 1
g = signal.lfilter([1], a, delta)
G = linalg.toeplitz(g, np.concatenate(([g[0]], np.zeros((p - 1, )))))
b = linalg.inv(G.T @ G) @ G.T @ hd
