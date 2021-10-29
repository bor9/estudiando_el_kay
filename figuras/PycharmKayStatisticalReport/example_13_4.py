import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from numpy.linalg import inv
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib import rc

from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col11 = scalarMap.to_rgba(0)
col12 = scalarMap.to_rgba(0.15)
col21 = scalarMap.to_rgba(1)
col22 = scalarMap.to_rgba(0.9)


def fun_H(s):
    rx = s[0]
    ry = s[1]
    R2 = rx ** 2 + ry ** 2
    return np.array([[rx / np.sqrt(R2), ry / np.sqrt(R2), 0, 0], [-ry / R2, rx / R2, 0, 0]])


def fun_h(s):
    rx = s[0]
    ry = s[1]
    return [np.sqrt(rx ** 2 + ry ** 2), np.arctan2(ry, rx)]


####### Parámetros #######

# número de muestras
N = 100
# delta - intevalo de tiempo entre muestras
D = 1
# velocidades
vx = -0.2
vy = 0.2
# condiciones iniciales del estado, s[-1]
rx_i = 10
ry_i = -5
s_i = [rx_i, ry_i, vx, vy]
# número de parámetros
p = 4
# matriz de transcición de estados
A = np.array([[1, 0, D, 0], [0, 1, 0, D], [0, 0, 1, 0], [0, 0, 0, 1]])
# media de u
mu_u = [0, 0, 0, 0]
# covarianza de u
var_u = 0.0001
Q = np.zeros((p, p))
Q[2, 2] = var_u
Q[3, 3] = var_u
# varianza de R[n] y beta[n]
var_R = 0.1
var_B = 0.01
C = [[var_R, 0], [0, var_B]]
# condiciones iniciales del filtro de Kalman
# s[-1|-1]
s_est_i = [5, 5, 0, 0]
# M[-1|-1]
C_s_i = 100 * np.eye(p)

### Fin de parámetros ###


# vector de muestras
n = np.arange(N)
# trayectoria ideal
rx_ideal =rx_i + vx * n
ry_ideal = ry_i + vy * n
# trayectoria verdadera - proceso de Gauss-Markov s[n] = As[n-1] + u[n]
s = np.zeros((p, N))  # para almacenar los valores
s_prev = s_i
for ni in n:
    u = np.random.multivariate_normal(mu_u, Q, 1)[0]
    s_prev = A @ s_prev + u
    s[:, ni] = s_prev

rx_true = s[0, :]
ry_true = s[1, :]

# construcción de la distancia y e ángulo a partir de rx[n] y ry[n]
R = np.sqrt(np.square(rx_true) + np.square(ry_true))
B = np.arctan2(ry_true, rx_true)

# observaciones ruidosas
hat_R = R + np.sqrt(var_R) * np.random.randn(N)
hat_B = B + np.sqrt(var_B) * np.random.randn(N)

# construcción del vector x de observacion
x = np.array([hat_R, hat_B])
#x = [hat_R, hat_B]

# filtro de kalman

# variables para guardar los resultados
s_ests = np.zeros((p, N))
Ms = np.zeros((p, N))
s_est = s_est_i
M_est = C_s_i
for ni in n:
    print(ni)
    s_pred = A @ s_est
    M_pred = A @ M_est @ A.T + Q
    H = fun_H(s_pred)
    K = M_pred @ H.T @ inv(C + H @ M_pred @ H.T)
    s_est = s_pred + K @ (x[:, ni] - fun_h(s_pred))
    M_est = (np.eye(p) - K @ H) @ M_pred
    # se salvan los resultados
    s_ests[:, ni] = s_est.ravel()
    Ms[:, ni] = np.diag(M_est)


xmin = 0
xmax = N-1
fs = 12

# observaciones ruidosas en coordenadas cartesianas - solo para gráfica
hat_rx = hat_R * np.cos(hat_B)
hat_ry = hat_R * np.sin(hat_B)
# estimaciones con el valor inicial
s_ests = np.insert(s_ests, 0, s_est_i, axis=1)


ax = plt.figure(0, figsize=(9, 5), frameon=False)
plt.plot(rx_ideal, ry_ideal, color=col22, label='$h_n[0]$')
plt.plot(rx_true, ry_true, color=col21, label='$h_n[1]$')
plt.plot(hat_rx, hat_ry, color=col12, label='$h_n[1]$', zorder=0)
plt.plot(s_ests[0, :], s_ests[1, :], color='k', label='$h_n[1]$')
plt.show()


