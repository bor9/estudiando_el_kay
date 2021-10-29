import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# Parámetros

# número de muestras
N = 100
# número de parámetros
p = 2
# condiciones iniciales
theta_i = np.zeros((p, ))
M_i = 1e2 * np.eye(p)
#M_i = np.eye(p)
# parámetros verdaderos de la sinusoide
a = 2
b = -1
f0 = 1 / 20  # periodo N = 10
# varianza del ruido
var = 2

# Fin de parámetros

# muestras
n = np.arange(N)
# señal
np.random.seed(4)
#np.random.seed(46)
x = a * np.cos(2 * math.pi * f0 * n) + b * np.sin(2 * math.pi * f0 * n) + math.sqrt(var) * np.random.randn(N)
# almacenamiento de variables
Ks = np.zeros((N, p))
Ms = np.zeros((N, p, p))
hat_ts = np.zeros((N+1, p))
hat_ts[0, :] = theta_i

# procesamiento
M = M_i
hat_theta = theta_i
for i in n:
    h = np.array([np.cos(2 * np.pi * f0 * i), np.sin(2 * np.pi * f0 * i)])
    K = (M @ h) / (var + h @ M @ h)
    M = (np.eye(p) - np.outer(K, h)) @ M
    hat_theta = hat_theta + K * (x[i] - h @ hat_theta)
    # save variables
    Ks[i, :] = K
    Ms[i, :, :] = M
    hat_ts[i+1, :] = hat_theta

print(hat_ts)

n2 = np.arange(-1, N)
nmin = -2
nmax = N-1
xleg = 0.93
ms = 3
fs = 12

fig = plt.figure(1, figsize=(9, 6), frameon=False)

ax = plt.subplot2grid((9, 1), (0, 0), rowspan=3, colspan=1)
plt.xlim(nmin, nmax)
plt.ylim(-6.5, 6.5)
plt.plot(n, a * np.cos(2 * math.pi * f0 * n) + b * np.sin(2 * math.pi * f0 * n), linestyle='-',
         color='r', marker='s', markersize=ms)
plt.plot(n, x, linestyle='-', color='k', marker='s', markersize=ms, label='$x[n]$')
ax.set_xticklabels([])
leg = plt.legend(loc='center', bbox_to_anchor=(xleg, 0.88), frameon=False, fontsize=fs)
ax.set_ylabel('${\\rm Se\\tilde{n}al\;limpia\;y}'+'\n'+'${\\rm se\\tilde{n}al\;ruidosa}$', fontsize=fs)

ax = plt.subplot2grid((9, 1), (3, 0), rowspan=3, colspan=1)
plt.xlim(nmin, nmax)
plt.gca().set_prop_cycle(plt.cycler('color', ['k', 'r']))
plt.plot(n2, hat_ts, linestyle='-', marker='s', markersize=ms)
plt.plot([nmin, nmax], [a, a], linestyle='--', lw=1, color='grey')
plt.plot([nmin, nmax], [b, b], linestyle='--', lw=1, color='grey')
ax.set_xticklabels([])
leg = plt.legend(['$\hat{a}[n]$', '$\hat{b}[n]$'], loc='center', bbox_to_anchor=(xleg, 0.6),
                 frameon=False, fontsize=fs)
ax.set_ylabel('${\\rm Estimadores}$', fontsize=fs)

ax = plt.subplot2grid((9, 1), (6, 0), rowspan=3, colspan=1)
plt.gca().set_prop_cycle(plt.cycler('color', ['k', 'r']))
plt.xlim(nmin, nmax)
plt.plot(n, Ks, linestyle='-', marker='s', markersize=ms)
leg = plt.legend(['$K_1[n]$', '$K_2[n]$'], loc='center', bbox_to_anchor=(xleg, 0.78),
                 frameon=False, fontsize=fs)
ax.set_ylabel('${\\rm Ganancia}$', fontsize=fs)
ax.set_xlabel('$n$', fontsize=fs)

plt.savefig('example_12_3.pdf', bbox_inches='tight')

plt.show()


