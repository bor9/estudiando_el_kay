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


####### Parámetros #######

# número de muestras
N = 400
# parámetro a de la ecuación de estado
a = 1
# varianza del ruido de excitación
var_u = 0.0000005
# varianza del ruido de observación
var_w = 0.1
# media y varianza de f0[-1]
mu_f0_i = 0.2
var_f0_i = 0.05

# parámetros del filtro de Kalman
# número de parámetros
p = 2
# matriz de transcición de estados
A = np.array([[a, 0], [2 * np.pi * a, 1]])
B = np.array([1, 2 * np.pi])
# condiciones iniciales del filtro de Kalman
# s[-1|-1]
s_est_i = np.array([[mu_f0_i], [0]])
# M[-1|-1]
C_s_i = 1 * np.eye(p)


### Fin de parámetros ###

ns = np.arange(N)

# generación de la frecuencia instantanea
f0d_1 = np.zeros((N,))
N1 = 100
N2 = 300
f01 = 0.1
f02 = 0.3
f0d_1[:N1] = f01
f0d_1[N1:N2] = (f02 - f01) / (N2 - N1) * np.arange(N2 - N1) + f01
f0d_1[N2:] = f02

# f01 = 0.1
# f02 = 0.3
# N1 = 200
# f0d_1[:N1] = f01
# f0d_1[N1:] = f02
# var_u = 0.000001


# generación de las observaciones
phi = 2 * np.pi * np.cumsum(f0d_1)
y = np.cos(phi)
x = y + np.random.normal(0, np.sqrt(var_w), N)


# variables para guardar los resultados
s_ests = np.zeros((p, N))
Ms = np.zeros((p, N))
s_est = s_est_i
M_est = C_s_i
for n in ns:
    s_pred = A @ s_est
    M_pred = A @ M_est @ A.T + var_u * B @ B
    H = np.array([[0, -np.sin(s_pred[1])]])
    K = M_pred @ H.T / (var_w + H @ M_pred @ H.T)
    s_est = s_pred + K * (x[n] - np.cos(s_pred[1]))
    M_est = (np.eye(p) - K @ H) @ M_pred
    s_ests[:, n] = s_est.ravel()
    Ms[:, n] = np.diag(M_est)




plt.figure(0)
plt.subplot(311)
plt.plot(ns, f0d_1, 'k')
plt.plot(ns, s_ests[0, :], 'r')
#plt.plot(ns[:-1], (s_ests[1, 1:]-s_ests[1, :-1])/(2 * np.pi), 'b')
plt.subplot(312)
plt.plot(ns, phi, 'k')
plt.plot(ns, s_ests[1, :], 'r')
plt.subplot(313)
plt.plot(ns, y, 'k', zorder=2)
plt.plot(ns, x, 'r', zorder=1)

plt.figure(1)
plt.plot(ns, f0d_1, 'k')
plt.plot(ns, s_ests[0, :], 'r')



plt.show()


