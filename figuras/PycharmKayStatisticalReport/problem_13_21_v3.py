import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from numpy.linalg import inv
import matplotlib.colors as colors
import math

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
var_u = 0.005
# varianza del ruido de observación
var_w = 0.05
# media y varianza de f0[-1]
mu_f0_i = 0.2
var_f0_i = 0.1

# parámetros del filtro de Kalman
# número de parámetros
p = 3
# matriz de transcición de estados
B = np.array([[0], [0], [1]])
H = np.array([[1, 0, 0]])
# condiciones iniciales del filtro de Kalman
# s[-1|-1]
s_est_i = np.array([[0.5], [0.5], [mu_f0_i]])
# M[-1|-1]
C_s_i = 100 * np.eye(p)
q = 0.001

def fun_a(s):
    a_1 = s[0] * math.cos(s[2]) - s[1] * math.sin(s[2])
    a_2 = s[0] * math.sin(s[2]) + s[1] * math.cos(s[2])
    a_3 = a * s[2]
    return np.array([a_1, a_2, a_3])


def fun_A(s):
    A_1 = [math.cos(s[2]), -math.sin(s[2]), -s[0] * math.sin(s[2]) - s[1] * math.cos(s[2])]
    A_2 = [math.sin(s[2]),  math.cos(s[2]),  s[0] * math.cos(s[2]) - s[1] * math.sin(s[2])]
    A_3 = [0, 0, a]
    return np.array([A_1, A_2, A_3])



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

f01 = 0.1
f02 = 0.3
N1 = 200
f0d_1[:N1] = f01
f0d_1[N1:] = f02
var_u = 0.01
q = 0.005


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
    s_pred = fun_a(s_est)
    A = fun_A(s_est)
    M_pred = A @ M_est @ A.T + q * np.eye(p)  #var_u * B @ B.T
    K = M_pred @ H.T / (var_w + H @ M_pred @ H.T)
    s_est = s_pred + K * (x[n] - H @ s_pred)
    M_est = (np.eye(p) - K @ H) @ M_pred
    s_ests[:, n] = s_est.ravel()
    Ms[:, n] = np.diag(M_est)



plt.figure(0)
plt.subplot(211)
plt.plot(ns, 2 * np.pi * f0d_1, 'k')
plt.plot(ns, s_ests[2, :], 'r')
#plt.plot(ns[:-1], (s_ests[1, 1:]-s_ests[1, :-1])/(2 * np.pi), 'b')
plt.subplot(212)
plt.plot(ns, y, 'k', zorder=2)
plt.plot(ns, x, 'r', zorder=1)

plt.figure(1)
plt.subplot(111)
plt.plot(ns, 2 * np.pi * f0d_1, 'k')
plt.plot(ns, s_ests[2, :], 'r')

plt.figure(2)
plt.subplot(111)
plt.plot(ns, y, 'k')
plt.plot(ns, x, 'b')
plt.plot(ns, s_ests[0, :], 'r')


plt.show()


