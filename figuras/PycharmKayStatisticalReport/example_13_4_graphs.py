import matplotlib.pyplot as plt
import numpy as np
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

# seed para graficas
np.random.seed(29)


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

# construcción de la distancia y el ángulo a partir de rx[n] y ry[n]
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
    s_pred = A @ s_est
    M_pred = A @ M_est @ A.T + Q
    H = fun_H(s_pred)
    K = M_pred @ H.T @ inv(C + H @ M_pred @ H.T)
    s_est = s_pred + K @ (x[:, ni] - fun_h(s_pred))
    M_est = (np.eye(p) - K @ H) @ M_pred
    # se salvan los resultados
    s_ests[:, ni] = s_est.ravel()
    Ms[:, ni] = np.diag(M_est)


fs = 12

# observaciones ruidosas en coordenadas cartesianas - solo para gráfica
hat_rx = hat_R * np.cos(hat_B)
hat_ry = hat_R * np.sin(hat_B)
# estimaciones con el valor inicial
s_ests = np.insert(s_ests, 0, s_est_i, axis=1)

fig = plt.figure(0, figsize=(9, 3), frameon=False)
ax = plt.subplot2grid((1, 8), (0, 0), rowspan=1, colspan=4)
plt.plot(rx_ideal, ry_ideal, color=col21, label='$\mathrm{Posici\\acute{o}n\;ideal}$')
plt.plot(rx_true, ry_true, color='r', label='$\mathrm{Posici\\acute{o}n\;verdadera}$')
plt.text(rx_true[0]+1, ry_true[0]-1.5, '$\mathrm{Inicio}$', fontsize=fs, ha='center', va='top')
plt.text(rx_true[-1]-1, ry_true[-1]+0.8, '$\mathrm{Fin}$', fontsize=fs, ha='center', va='bottom')
leg = plt.legend(loc=3, frameon=False, fontsize=fs)
plt.axis([-22, 16, -10, 21])
plt.xlabel('$r_x[n]$', fontsize=fs)
plt.ylabel('$r_y[n]$', fontsize=fs)
ax = plt.subplot2grid((1, 8), (0, 4), rowspan=1, colspan=4)
plt.plot(rx_true, ry_true, color='r', label='$\mathrm{Posici\\acute{o}n\;verdadera}$')
plt.plot(hat_rx, hat_ry, color=col12, label='$\mathrm{Posici\\acute{o}n\;medida\;}(\mathbf{x}[n])$', zorder=0)
leg = plt.legend(loc=3, frameon=False, fontsize=fs)
plt.axis([-22, 16, -10, 21])
ax.set_yticklabels([])
plt.xlabel('$r_x[n]$', fontsize=fs)
plt.savefig('example_13_4_tracks.pdf', bbox_inches='tight')


fig = plt.figure(1, figsize=(9, 3), frameon=False)
ax = plt.subplot(121)
plt.plot(n, R, color='k', label='$R[n]$')
plt.plot(n, hat_R, color='r', label='$\hat{R}[n]$')
leg = plt.legend(loc=2, frameon=False, fontsize=fs)
plt.axis([n[0], n[-1], 0, 28])
plt.xlabel('$n$', fontsize=fs)
plt.ylabel('$\mathrm{Distancia\;(metros)}$', fontsize=fs)
ax = plt.subplot(122)
plt.plot(n, np.rad2deg(B), color='k', label='$\\beta[n]$')
plt.plot(n, np.rad2deg(hat_B), color='r', label='$\hat{\\beta}[n]$')
leg = plt.legend(loc=2, frameon=False, fontsize=fs)
plt.axis([n[0], n[-1], -40, 160])
plt.xlabel('$n$', fontsize=fs)
plt.ylabel('$\mathrm{\\acute{A}ngulo\;(grados)}$', fontsize=fs)
plt.savefig('example_13_4_observations.pdf', bbox_inches='tight')


ax = plt.figure(2, figsize=(9, 5), frameon=False)
plt.plot(rx_true, ry_true, color='r', label='$\mathrm{Posici\\acute{o}n\;verdadera}$')
plt.plot(hat_rx, hat_ry, color=col12, label='$\mathrm{Posici\\acute{o}n\;medida\;}(\mathbf{x}[n])$', zorder=0)
plt.plot(s_ests[0, :], s_ests[1, :], color='k', label='$\mathrm{Estimador\;del\;filtro\;de\;Kalman\;extendido}$')
plt.axis([-22, 16, -10, 21])
plt.xlabel('$r_x[n]$', fontsize=fs)
plt.ylabel('$r_y[n]$', fontsize=fs)
leg = plt.legend(loc=1, frameon=False, fontsize=fs)
plt.savefig('example_13_4_kalman_estimator.pdf', bbox_inches='tight')


fig = plt.figure(3, figsize=(9, 3), frameon=False)
ax = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
plt.plot(n, Ms[0, :], color='k')
plt.axis([n[0], n[-1], 0, 0.8])
plt.xlabel('$n$', fontsize=fs)
plt.title('$\mathrm{MSE\;m\\acute{\i}nimo\;para\;}r_x[n]$', fontsize=fs)
ax = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
plt.plot(n, Ms[1, :], color='k', label='$\\beta[n]$')
plt.axis([n[0], n[-1], 0, 3.5])
plt.xlabel('$n$', fontsize=fs)
plt.title('$\mathrm{MSE\;m\\acute{\i}nimo\;para\;}r_y[n]$', fontsize=fs)
#ax.yaxis.tick_right()
#ax.yaxis.set_label_position("right")
plt.savefig('example_13_4_minimum_mse.pdf', bbox_inches='tight')
plt.show()

