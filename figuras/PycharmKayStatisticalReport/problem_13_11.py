import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from matplotlib import rc

from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

col1 = 'b'
col2 = 'k'
col3 = 'r'

####### Parámetros #######

# número de muestras para la ganancia y el MSE mínimo
N1 = 101
# número de muestras para el filtrado
N2 = 51
# a del modelo de estados
a = 0.9
# varianza del ruiso de excitación
var_u = 1
# media y varianza de s[-1]
mu_s = 0
var_s = 1
# los tres casos de la varianza del ruido de observación para n=1
var_w = [0.9, 1, 1.1]

### Fin de parámetros ###

# seed para graficas
np.random.seed(4)

ns1 = np.arange(N1)
n_exp = len(var_w)

## Cálculo de la ganancia de Kalman y el MSE mínimo
# inicialización de variables para guardar los resultados
Ks = np.zeros((N1, n_exp))
Ms = np.zeros((N1, n_exp))
# M[-1|-1] = var_s
M = var_s
for k in np.arange(n_exp):
    for n in ns1:
        M_pred = (a ** 2) * M + var_u
        K = M_pred / (var_w[k] ** n + M_pred)
        M = (1 - K) * M_pred
        # se guardan los resultados
        Ks[n, k] = K
        Ms[n, k] = M

## Filtrado
ns2 = np.arange(N2)
# generación de los procesos de Markov
# ruido de excitación - se emplea el mismo en todos los casos
u = np.sqrt(var_u) * np.random.randn(N2)
# s[-1]
s_i = np.random.normal(mu_s, np.sqrt(var_s), 1)
# generación del proceso
s, z_f = signal.lfilter([1], [1, -a], u, zi=a * s_i)
# ruido de observación
w = np.random.randn(N2)
# generación de las observaciones
xs = np.zeros((N2, n_exp))
for k in np.arange(n_exp):
    xs[:, k] = s + np.sqrt(np.power(var_w[k], ns2)) * w

# filtrado
s_ests = np.zeros((N2, n_exp))
for k in np.arange(n_exp):
    # inicialización: s[-1|-1]
    s_est = mu_s
    for n in ns2:
        s_pred = a * s_est
        s_est = s_pred + Ks[n, k] * (xs[n, k] - s_pred)
        # se guarda el resultado
        s_ests[n, k] = s_est


fs = 12

fig = plt.figure(0, figsize=(9, 4), frameon=False)
ax = plt.subplot2grid((8, 2), (0, 0), rowspan=8, colspan=1)
plt.plot([0, N1-1], [1, 1], 'k--', dashes=(4, 4))
plt.plot(ns1, Ks[:, 0], color = col1, lw=2)
plt.plot(ns1, Ks[:, 1], color = col2, lw=2)
plt.plot(ns1, Ks[:, 2], color = col3, lw=2)
plt.xlim(0, N1-1)
plt.ylim(0, 1.05)
plt.xlabel('$n$', fontsize=fs)
plt.ylabel('$K[n]$', fontsize=fs)

var_s_steady = var_u / (1 - a ** 2)
ax = plt.subplot2grid((8, 2), (0, 1), rowspan=8, colspan=1)
plt.plot([0, N1-1], [var_s_steady, var_s_steady], 'k--', dashes=(4, 4))
plt.plot(ns1, Ms[:, 0], color = col1, lw=2, label='$\sigma_n^2=({})^n$'.format(var_w[0]))
plt.plot(ns1, Ms[:, 1], color = col2, lw=2, label='$\sigma_n^2={}$'.format(var_w[1]))
plt.plot(ns1, Ms[:, 2], color = col3, lw=2, label='$\sigma_n^2=({})^n$'.format(var_w[2]))
plt.xlim(0, N1-1)
plt.ylim(0, 5.5)
plt.xlabel('$n$', fontsize=fs)
plt.ylabel('$M[n|n]$', fontsize=fs)
plt.annotate('$\mathrm{var}(s[n])=\dfrac{\sigma_u^2}{1-a^2}$', xytext=(3, var_s_steady-0.4), xycoords='data',
             xy=(28, var_s_steady), textcoords='data', color='k', fontsize=fs, va="top", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.3", color='k', relpos=(0.5, 1),
                             patchA=None, patchB=None, shrinkA=0, shrinkB=1))
leg = plt.legend(loc='center right', frameon=False, fontsize=fs)
plt.savefig('problem_13_11_gain_MSE.pdf', bbox_inches='tight')


fig = plt.figure(2, figsize=(9, 6), frameon=False)
ax = plt.subplot2grid((9, 1), (0, 0), rowspan=3, colspan=1)
k = 0
plt.plot(ns2, s, 'r')
plt.plot(ns2, xs[:, k], color = 'b')
plt.plot(ns2, s_ests[:, k], color = 'k')
plt.xlim(0, N2-1)
ax.set_xticklabels([])
plt.text(0.02, 0.8, '$\sigma_n^2=({})^n$'.format(var_w[k]), fontsize=fs, ha='left', va='baseline',
         transform = ax.transAxes)


ax = plt.subplot2grid((9, 1), (3, 0), rowspan=3, colspan=1)
k = 1
plt.plot(ns2, s, 'r')
plt.plot(ns2, xs[:, k], color = 'b')
plt.plot(ns2, s_ests[:, k], color = 'k')
plt.xlim(0, N2-1)
ax.set_xticklabels([])
plt.text(0.02, 0.8, '$\sigma_n^2={}$'.format(var_w[k]), fontsize=fs, ha='left', va='baseline',
         transform = ax.transAxes)


ax = plt.subplot2grid((9, 1), (6, 0), rowspan=3, colspan=1)
k = 2
plt.plot(ns2, s, 'r', label='$s[n]$')
plt.plot(ns2, xs[:, k], color = 'b', label='$x[n]$')
plt.plot(ns2, s_ests[:, k], color = 'k', label='$\hat{s}[n|n]$')
plt.xlim(0, N2-1)
plt.xlabel('$n$', fontsize=fs)
plt.text(0.02, 0.8, '$\sigma_n^2=({})^n$'.format(var_w[k]), fontsize=fs, ha='left', va='baseline',
         transform = ax.transAxes)
leg = plt.legend(loc=3, frameon=False, fontsize=fs, ncol=3)

plt.savefig('problem_13_11_filtering.pdf', bbox_inches='tight')

plt.show()