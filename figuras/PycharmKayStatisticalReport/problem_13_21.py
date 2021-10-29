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
N = 200
# parámetro a de la ecuación de estado
a = 0.995
# varianza del ruido de excitación
var_u = 0.000001
# varianza del ruido de observación
var_w = 0.5
# media y varianza de f0[-1]
mu_f0_i = 0.25
var_f0_i = 0.01

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
C_s_i = 100 * np.eye(p)


### Fin de parámetros ###

ns = np.arange(N)

# generación de f0[n], que es un proceso de Gauss-Markov de primer orden
# se sortea f0[-1]
f0_i = np.random.normal(mu_f0_i, np.sqrt(var_f0_i), 1)
# generación del proceso
# ruido de excitación
u = np.random.normal(0, np.sqrt(var_u), N)
# filtrado del ruido de excitación con condiciones iniciales
f0, z_f = signal.lfilter([1], [1, -a], u, zi=a * f0_i)


# generación de las observaciones
phi = 2 * np.pi * np.cumsum(f0)
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


f0s = s_ests[0, :]
int_part  = np.floor(f0s/0.5)
f0s_unw = f0s
for n in ns:
    if int_part[n] != 0:
        if int_part[n] % 2 == 0:
            f0s_unw[n] = f0s[n] - int_part[n] * 0.5
        else:
            f0s_unw[n] = 0.5 - (f0s[n] - int_part[n] * 0.5)


plt.figure(0)
plt.subplot(311)
plt.plot(ns, f0, 'k')
plt.plot(ns, s_ests[0, :], 'r')
#plt.plot(ns[:-1], (s_ests[1, 1:]-s_ests[1, :-1])/(2 * np.pi), 'b')
plt.subplot(312)
plt.plot(ns, phi, 'k')
plt.plot(ns, s_ests[1, :], 'r')
plt.subplot(313)
plt.plot(ns, y, 'k', zorder=2)
plt.plot(ns, x, 'r', zorder=1)

plt.figure(1)
plt.subplot(111)
plt.plot(ns, f0, 'k')
plt.plot(ns, f0s_unw, 'r')

plt.show()


