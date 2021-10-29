import numpy as np

# Filtro de kalman para estimación de los p=2 coeficientes del canal.
# h_est es hat{h}[n|n], la estimación de los coeficientes en el paso n
# y M_est es M[n|n], el MSE mínimo en el paso n.
# v[n] es la entrada al canal. Tiene largo N+1 y el primer elemento es v[-1]=0.
# A y Q son la matrices correspondientes al modelo de estado.
# var_w es la varianza del ruido de observación.
# x[n] es el vector de observación, de largo N.

# condiciones iniciales del filtro de Kalman
# h[-1|-1]
mu_h = [[0], [0]]
# M[-1|-1]
C_h = 100 * np.eye(p)
# inicialización de la recursión
h_est = mu_h
M_est = C_h
for n in np.arange(N):
    # cálculo de la predicción \hat{h}[n|n-1]
    h_pred = A @ h_est
    # cálculo de la predicción del MSE mínimo M[n|n-1]
    M_pred = A @ M_est @ A.T + Q
    # construcción del vector v[n] de dimensión (1, 2)
    vn = np.array([[v[n+1]], [v[n]]])
    # cálculo de la ganancia de Kalman
    K = (M_pred @ vn) / (var_w + vn.T @ M_pred @ vn)
    # corrección \hat{h}[n|n]
    h_est = h_pred + K * (x[n] - vn.T @ h_pred)
    # cálculo del MSE mínimo M[n|n]
    M_est = (np.eye(p) - K @ vn.T) @ M_pred