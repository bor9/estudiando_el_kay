import numpy as np
from numpy.linalg import inv


def fun_H(s):
    """
    Implementación del cálculo de la matriz H[n]
    :param s: vector de estados
    """
    rx = s[0]
    ry = s[1]
    R2 = rx ** 2 + ry ** 2
    return np.array([[rx / np.sqrt(R2), ry / np.sqrt(R2), 0, 0], [-ry / R2, rx / R2, 0, 0]])


def fun_h(s):
    """
    Implementación del cálculo de h(s[n])
    :param s: vector de estados
    """
    rx = s[0]
    ry = s[1]
    return [np.sqrt(rx ** 2 + ry ** 2), np.arctan2(ry, rx)]

# Implementación del filtro de Kalman extendido.
# p es el número de elementos de la señal, N es el número de muestras,
# A es la matriz (p, p) de transición de estados, Q es la matriz (p, p) de ruido de
# medición, C es la matriz (2, 2) de ruido de observación,
# s_est_i es el vector (p, ) de estado inicial hat{s}[-1|-1] y C_s_i es la matriz (p, p)
# de MSE mínimo inicial M[-1|-1], x una matriz (2, N) con los vectores de observación
# en las columnas.
s_est = s_est_i
M_est = C_s_i
for n in np.arange(N):
    # predicción del estimador
    s_pred = A @ s_est
    # predicción de la matriz de MSE mínimo
    M_pred = A @ M_est @ A.T + Q
    # cálculo de H
    H = fun_H(s_pred)
    # cálculo de la ganancia de Kalman
    K = M_pred @ H.T @ inv(C + H @ M_pred @ H.T)
    # corrección del estimador
    s_est = s_pred + K @ (x[:, n] - fun_h(s_pred))
    # corrección de la matriz de MSE mínimo
    M_est = (np.eye(p) - K @ H) @ M_pred