import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


##########################################
# PARAMETROS - Esto puede ser modificado #
##########################################

# coeficientes del filtro de Kalman
a = 0.9
var_u = 1
var_n = 1
Mi = 1
N = 31

#####################
# FIN DE PARAMETROS #
#####################
# abscissa values
xmin = 0
xmax = N-1
ymin = 0.595
ymax = 0.648

ns = np.arange(N)

Ms = np.zeros((N,))
Mps = np.zeros((N,))
Ks = np.zeros((N,))

M = Mi
for n in ns:
    # predicción del MSE mínimo
    Mp = (a ** 2) * M + var_u
    # ganancia de Kalman
    K = Mp / ((var_n ** (n + 1)) + Mp)
    # MSE mínimo
    M = (1 - K) * Mp
    # almacenamiento de valores
    Ms[n] = M
    Mps[n] = Mp
    Ks[n] = K

# Calculo analítico de K[\infty] y M[infty]
p = [-var_u * var_n, var_u + var_n * (1 - a ** 2), a ** 2]
M_inf = np.max(poly.polyroots(p))

# axis parameters
xmin_ax = xmin
xmax_ax = xmax

ymax_ax = ymax
ymin_ax = ymin

# font size
fs = 12


fig = plt.figure(0, figsize=(5, 3), frameon=False)

ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

plt.plot(ns, Ms, 'ks-', markersize=5, zorder=2, label='$K[n],\,M[n|n]$')
plt.plot([0, N-1], [M_inf, M_inf], 'r--', dashes=(5, 6), label='$M[\infty]$')

leg = plt.legend(loc=1, frameon=False, fontsize=fs)
ax.set_xlabel('$n$', fontsize=fs)

# save as pdf image
plt.savefig('problem_13_17.pdf', bbox_inches='tight')

plt.show()

print(M_inf)
print(Ms)