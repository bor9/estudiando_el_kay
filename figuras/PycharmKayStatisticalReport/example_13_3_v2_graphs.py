import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
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

####### Parámetros #######

# número de muestras
N = 100
# periodo de la señal cuadrada
T = 10
# muestras
n = np.arange(N)
# número de parámetros
p = 2
# media de h[-1]
mu_hi = [1, 1]
# covarianza de h
C_hi = [[0.1, 0], [0, 0.1]]
# matriz de transcición de estados
A = np.array([[0.99, 0], [0, 0.999]])
# media de u
mu_u = [0, 0]
# covarianza de u
Q = [[0.0001, 0], [0, 0.0001]]
# varianza de w[n]
var_w = 0.1
# condiciones iniciales del filtro de Kalman
# h[-1|-1]
mu_h = np.array([[0], [0]])
mu_h = [[0], [0]]
# M[-1|-1]
C_h = 100 * np.eye(p)

### Fin de parámetros ###

# seed para graficas
#np.random.seed(3)
#np.random.seed(4)
#np.random.seed(8)
#np.random.seed(10)
#np.random.seed(13)
#np.random.seed(21)
np.random.seed(32)


# señal cuadrada
v = (1 - signal.square(2 * np.pi / T * n + 1e-14, duty=0.5)) / 2

# generación del proceso de Gauss-Markov h[n] = Ah[n-1] + u[n]
h = np.zeros((2, N))  # para almacenar los valores
# valor inicial h[-1]: variable aleatoria de distribución normal multivariada con
# media mu_h y matriz de covarianza C_h
h_i = np.random.multivariate_normal(mu_hi, C_hi, 1)[0]
h_prev = h_i
for ni in n:
    u = np.random.multivariate_normal(mu_u, Q, 1)[0]
    h_prev = A @ h_prev + u
    h[:, ni] = h_prev

v_ext = np.insert(v, 0, 0)
# filtrado de v[n] con filtro variable con coeficientes h
y = np.zeros((N, ))
for ni in n:
    y[ni] = [v_ext[ni+1], v_ext[ni]] @ h[:, ni]

# construcción de x[n]
x = y + np.sqrt(var_w) * np.random.randn(N)

# filtro de kalman

# variables para guardar los resultados
Ks = np.zeros((2, N))
hs = np.zeros((2, N))
Ms = np.zeros((2, N))

# h_ii es h[-1|-1] = [0 0]^T
h_est = mu_h
M_est = C_h
for ni in n:
    h_pred = A @ h_est
    M_pred = A @ M_est @ A.T + Q
    vn = np.array([[v_ext[ni+1]], [v_ext[ni]]])
    K = (M_pred @ vn) / (var_w + vn.T @ M_pred @ vn)
    h_est = h_pred + K * (x[ni] - vn.T @ h_pred)
    M_est = (np.eye(p) - K @ vn.T) @ M_pred
    # se salvan los resultados
    Ks[:, ni] = K.ravel()
    hs[:, ni] = h_est.ravel()
    Ms[:, ni] = np.diag(M_est)


xmin = 0
xmax = N-1
fs = 12

ax = plt.figure(0, figsize=(5, 3), frameon=False)
plt.plot(n, h[0, :], color=col11, label='$h_n[0]$')
plt.plot(n, h[1, :], color=col21, label='$h_n[1]$')
plt.ylim(0, 1.5)
plt.xlim(xmin, xmax)
plt.xlabel('$n$', fontsize=fs)
leg = plt.legend(loc=0, frameon=False, fontsize=fs)
plt.savefig('example_13_3_coefficients.pdf', bbox_inches='tight')


fig = plt.figure(1, figsize=(9, 5), frameon=False)
ax = plt.subplot2grid((8, 4), (0, 0), rowspan=4, colspan=4)
plt.plot(n, v, 's-', color=col11, markersize=3, label='$v[n]$')
plt.plot(n, y, 's-', color=col21, markersize=3, label='$y[n]$')
plt.ylim(-0.8, 2.8)
plt.xlim(xmin, xmax)
ax.set_xticklabels([])
leg = plt.legend(loc=1, frameon=False, fontsize=fs, ncol=2)
ax = plt.subplot2grid((8, 4), (4, 0), rowspan=4, colspan=4)
plt.plot(n, x, 's-', color=col11, markersize=3, label='$x[n]$')
plt.ylim(-0.8, 2.8)
plt.xlim(xmin, xmax)
leg = plt.legend(bbox_to_anchor=(0.8, 0.78), loc="lower left", frameon=False, fontsize=fs)
plt.xlabel('$n$', fontsize=fs)
plt.savefig('example_13_3_channel_input_output.pdf', bbox_inches='tight')


fig = plt.figure(2, figsize=(9, 4), frameon=False)
plt.plot(n, hs[0, :], color=col11, label='$\hat{h}_n[0]$', zorder=2)
plt.plot(n, hs[1, :], color=col21, label='$\hat{h}_n[1]$', zorder=2)
plt.plot(n, h[0, :], '--', color=col12, label='$h_n[0]$', dashes=(3, 1), zorder=1)
plt.plot(n, h[1, :], '--', color=col22, label='$h_n[1]$', dashes=(3, 1), zorder=1)
plt.ylim(0, 1.5)
plt.xlim(xmin, xmax)
plt.xlabel('$n$', fontsize=fs)
leg = plt.legend(bbox_to_anchor=(0.1, 0.05), loc="lower left", frameon=False, fontsize=fs, ncol=2)
plt.savefig('example_13_3_channel_estimators.pdf', bbox_inches='tight')


fig = plt.figure(3, figsize=(9, 4), frameon=False)
plt.plot(n, Ks[0, :], 's-', color=col11, label='$K_1[n]$', markersize=3)
plt.plot(n, Ks[1, :], 's-', color=col21, label='$K_2[n]$', markersize=3)
plt.xlim(xmin, xmax)
plt.xlabel('$n$', fontsize=fs)
leg = plt.legend(loc=1, frameon=False, fontsize=fs)

plt.savefig('example_13_3_kalman_gain.pdf', bbox_inches='tight')

fig = plt.figure(4, figsize=(9, 4), frameon=False)
plt.plot(n, Ms[0, :], 's-', color=col11, label='$\mathbf{M}_{11}[n|n]$', markersize=3)
plt.plot(n, Ms[1, :], 's-', color=col21, label='$\mathbf{M}_{22}[n|n]$', markersize=3)
plt.xlim(xmin, xmax)
plt.ylim(0, 0.2)
plt.xlabel('$n$', fontsize=fs)
leg = plt.legend(loc=1, frameon=False, fontsize=fs)

plt.savefig('example_13_3_minimum_mse.pdf', bbox_inches='tight')


plt.show()
