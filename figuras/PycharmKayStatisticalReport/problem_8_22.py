import matplotlib.pyplot as plt
import numpy as np
import math
import random

from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]


def sequential_lse(x, A0, var_A0, r):
    # numero de muestras de los datos
    N = x.shape[0]
    # inicialización de vectores con las estimaciones, ganancias
    # y varianzas del LS secuencial
    A_est = np.zeros((N, ))
    gains = np.zeros((N, ))
    vars = np.zeros((N, ))
    A_est[0] = A0
    vars[0] = var_A0
    for n in np.arange(1, N):
        gains[n] = vars[n - 1] / (vars[n - 1] + r ** n)
        A_est[n] = A_est[n - 1] + gains[n] * (x[n] - A_est[n - 1])
        vars[n] = (1 - gains[n]) * vars[n - 1]
    return gains, A_est, vars


# Parámetros
A = 10
rs = [0.95, 1, 1.05]
N = 100

# Generación de los datos
n_seq = len(rs)
xs = np.zeros((N, n_seq))
for j in np.arange(n_seq):
    random.seed(35)
    #random.seed(40)
    #random.seed(65)
    r = rs[j]
    for i in np.arange(N):
        xs[i, j] = random.normalvariate(A, math.sqrt(r ** i))

gains = np.zeros((N, n_seq))
A_est = np.zeros((N, n_seq))
vars = np.zeros((N, n_seq))

for j in np.arange(n_seq):
    r = rs[j]
    gains[:, j], A_est[:, j], vars[:, j] = sequential_lse(xs[:, j], xs[0, j], 1, rs[j])

n = np.arange(N)
print(A_est[0, 0])

ydata_max = 20
ydata_min = 0
est_max = 10.5
est_min = 9.9
var_max = 1
var_min = 0
gain_max = gains[1, 0]
gain_min = 0

fontsize = 13

fig = plt.figure(0, figsize=(10, 7), frameon=False)
j = 0
ax = plt.subplot2grid((12, 9), (0, 0), rowspan=3, colspan=3)
plt.plot(n, xs[:, j], 'k')
plt.ylim(ydata_min, ydata_max)
plt.xlim(0, N-1)
ax.set_xticklabels([])
plt.title('$r={:.2f}$'.format(rs[j]), fontsize=fontsize)
plt.ylabel('$x[n]$', fontsize=fontsize)
ax = plt.subplot2grid((12, 9), (3, 0), rowspan=3, colspan=3)
plt.plot(n, A_est[:, j], 'k')
plt.ylim(est_min, est_max)
plt.xlim(0, N-1)
plt.plot([0, N-1], [A, A], 'k--', lw=1)
ax.set_xticklabels([])
plt.ylabel('$\hat{A}[n]$', fontsize=fontsize)
ax = plt.subplot2grid((12, 9), (6, 0), rowspan=3, colspan=3)
plt.plot(n, vars[:, j], 'k')
plt.ylim(var_min, var_max)
plt.xlim(0, N-1)
ax.set_xticklabels([])
plt.ylabel('$\mathrm{var}(\hat{A}[n])$', fontsize=fontsize)
ax = plt.subplot2grid((12, 9), (9, 0), rowspan=3, colspan=3)
plt.plot(n[1:], gains[1:, j], 'k')
plt.ylim(gain_min, gain_max)
plt.xlim(0, N-1)
plt.ylabel('$K[n]$', fontsize=fontsize)
plt.xlabel('$n$', fontsize=fontsize)
gain_lim = 1 - rs[j]
plt.plot([0, N-1], [gain_lim, gain_lim], 'k--', lw=1)

j += 1
ax = plt.subplot2grid((12, 9), (0, 3), rowspan=3, colspan=3)
plt.plot(n, xs[:, j], 'k')
plt.ylim(ydata_min, ydata_max)
plt.xlim(0, N-1)
plt.title('$r={:.2f}$'.format(rs[j]))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax = plt.subplot2grid((12, 9), (3, 3), rowspan=3, colspan=3)
plt.plot(n, A_est[:, j], 'k')
plt.ylim(est_min, est_max)
plt.xlim(0, N-1)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.plot([0, N-1], [A, A], 'k--', lw=1)
ax = plt.subplot2grid((12, 9), (6, 3), rowspan=3, colspan=3)
plt.plot(n, vars[:, j], 'k')
plt.ylim(var_min, var_max)
plt.xlim(0, N-1)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax = plt.subplot2grid((12, 9), (9, 3), rowspan=3, colspan=3)
plt.plot(n[1:], gains[1:, j], 'k')
plt.ylim(gain_min, gain_max)
plt.xlim(0, N-1)
ax.set_yticklabels([])
plt.xlabel('$n$', fontsize=fontsize)

j += 1
ax = plt.subplot2grid((12, 9), (0, 6), rowspan=3, colspan=3)
plt.plot(n, xs[:, j], 'k')
plt.ylim(ydata_min, ydata_max)
plt.xlim(0, N-1)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.title('$r={:.2f}$'.format(rs[j]))
ax = plt.subplot2grid((12, 9), (3, 6), rowspan=3, colspan=3)
plt.plot(n, A_est[:, j], 'k')
plt.ylim(est_min, est_max)
plt.xlim(0, N-1)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.plot([0, N-1], [A, A], 'k--', lw=1)
ax = plt.subplot2grid((12, 9), (6, 6), rowspan=3, colspan=3)
plt.plot(n, vars[:, j], 'k')
plt.ylim(var_min, var_max)
plt.xlim(0, N-1)
ax.set_xticklabels([])
ax.set_yticklabels([])
var_lim = (rs[j]-1)/rs[j]
plt.plot([0, N-1], [var_lim, var_lim], 'k--', lw=1)
ax = plt.subplot2grid((12, 9), (9, 6), rowspan=3, colspan=3)
plt.plot(n[1:], gains[1:, j], 'k')
plt.ylim(gain_min, gain_max)
plt.xlim(0, N-1)
ax.set_yticklabels([])
plt.xlabel('$n$', fontsize=fontsize)

plt.savefig('problem_8_22.pdf', bbox_inches='tight')

plt.show()