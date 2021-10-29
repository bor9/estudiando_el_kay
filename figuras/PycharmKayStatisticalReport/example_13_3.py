import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# colors
lgray = "#dddddd"  # ligth gray

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
mu_h = [1, 1]
# covarianza de h
C_h = [[0.1, 0], [0, 0.1]]
# matriz de transcición de estados
A = np.array([[0.99, 0], [0, 0.999]])
# media de u
mu_u = [0, 0]
# covarianza de u
Q = np.array([[0.0001, 0], [0, 0.0001]])
# varianza de w[n]
var_w = 0.1
# condiciones iniciales del filtro de Kalman
# h[-1|-1]
h_ii = np.array([[0], [0]])
print(h_ii.shape)
# M[-1|-1]
M_ii = 100 * np.eye(p)

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

# generación de h[n]
# valor inicial h[-1]
h_i = np.random.multivariate_normal(mu_h, C_h, 1)[0]
h = np.zeros((N, 2))
h_prev = h_i
for ni in n:
    u = np.random.multivariate_normal(mu_u, Q, 1)[0]
    h_prev = A @ h_prev + u
    h[ni, :] = h_prev

v_ext = np.insert(v, 0, 0)
# filtrado de v[n] con filtro variable con coeficientes h
y = np.zeros((N, ))
for ni in n:
    y[ni] = [v_ext[ni+1], v_ext[ni]] @ h[ni, :]

# construcción de x[n]
x = y + np.sqrt(var_w) * np.random.randn(N)

# filtro de kalman

# variables para guardar los resultados
Ks = np.zeros((2, N))
hs = np.zeros((2, N))
Ms = np.zeros((2, N))

# h_ii es h[-1|-1] = [0 0]^T
h_est = h_ii
M_est = M_ii
for ni in n:
    # cálculo de h[n|n-1]
    h_pred = A @ h_est
    #
    M_pred = A @ M_est @ A.T + Q
    vn = np.array([[v_ext[ni+1]], [v_ext[ni]]])
    K = (M_pred @ vn) / (var_w + vn.T @ M_pred @ vn)
    h_est = h_pred + K * (x[ni] - vn.T @ h_pred)
    M_est = (np.eye(p) - K @ vn.T) @ M_pred
    # se salvan los resultados
    Ks[:, ni] = K.ravel()
    hs[:, ni] = h_est.ravel()
    Ms[:, ni] = np.diag(M_est)


fig = plt.figure(0, figsize=(9, 4), frameon=False)
plt.plot(n, v, 'ks-', markersize=3)
plt.plot(n, y, 'rs-', markersize=3)
plt.plot(n, x, 'bs-', markersize=3)
plt.ylim(-0.5, 3)


fig = plt.figure(1, figsize=(9, 4), frameon=False)
plt.plot(n, v, 'ks-', markersize=3)
plt.plot(n, y, 'rs-', markersize=3)
plt.plot(n, x, 'bs-', markersize=3)
plt.ylim(-0.5, 3)

fig = plt.figure(2, figsize=(9, 4), frameon=False)
plt.plot(n, h)
plt.plot(n, hs.T, '--')

fig = plt.figure(3, figsize=(9, 4), frameon=False)
plt.plot(n, Ks.T)

plt.show()
