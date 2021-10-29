import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
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

### Parámetros ###
a = 0.9
var_u = 1
var_w = 1

# número de frecuencias entre 0 y 0.5
nf = 512


##################

# resolución de la ecuación de ricatti de estado estacionario
# para calcular M[\infty]
p = [-var_u * var_w, var_u + var_w * (1 - a ** 2), a ** 2]
M_inf = np.max(poly.polyroots(p))
# cálculo de Mp[\infty]
Mp_inf = (a ** 2) * M_inf + var_u
# cálculo de K[\infty]
K_inf = Mp_inf / (var_w + Mp_inf)
print(M_inf)
print(Mp_inf)
print(K_inf)


# vector de frecuencias (normalizada)
f = np.arange(nf)/(2 * nf)
# respuesta en frecuencia del filtro de Kalman en estado estacionario
_, H_inf = signal.freqz(K_inf, [1, -a * (1 - K_inf)], worN=nf, whole=False)
# PSD de la señal s[n] en estado estacionario: proceso AR(1)
_, S = signal.freqz(var_u, [1, -a], worN=nf, whole=False)
Pss = np.square(np.abs(S))

H_inf_db = 20 * np.log10(np.abs(H_inf))
Pss_db = 10 * np.log10(Pss)

# filtro de Kalman como blanqueador
_, H_w = signal.freqz([1, -a], [1, -a * (1 - K_inf)], worN=nf, whole=False)
# PSD de los datos x[n]
Pxx = Pss + var_w
# PSD de la innovación
Pii = np.square(np.abs(H_w)) * Pxx
# para probar
Pii_2 = (var_u + var_w * (1 - 2 * a * np.cos(2 * np.pi * f) + a ** 2)) / \
        (1 - 2 * a * (1 - K_inf) * np.cos(2 * np.pi * f) + (a * (1 - K_inf)) ** 2)

H_w_db = 20 * np.log10(np.abs(H_w))
Pxx_db = 10 * np.log10(Pxx)
Pii_db = 10 * np.log10(Pii)

# fontsize
fs = 12

fig = plt.figure(0, figsize=(9, 4), frameon=False)
ax = plt.subplot2grid((8, 4), (0, 0), rowspan=8, colspan=2)
plt.xlim(0, 0.5)
plt.ylim(-17, 21)
plt.plot(f, H_inf_db, 'k', label='$|H_\infty(f)|$', lw=2)
plt.plot(f, Pss_db, 'r', label='$P_{ss}(f)$', lw=2)
ax.set_xlabel('${\\rm Frecuencia\;normalizada}$', fontsize=fs)
ax.set_ylabel('${\\rm Magnitud\;(dB)}$', fontsize=fs)
leg = plt.legend(loc=1, frameon=False, fontsize=fs)

ax = plt.subplot2grid((8, 4), (0, 2), rowspan=8, colspan=2)
plt.xlim(0, 0.5)
plt.ylim(-17, 21)
plt.plot(f, H_w_db, 'k', label='$|H_w(f)|$', lw=2)
plt.plot(f, Pxx_db, 'r', label='$P_{xx}(f)$', lw=2)
plt.plot(f, Pii_db, 'b', label='$P_{\\tilde{x}\\tilde{x}}(f)$', lw=2)
ax.set_xlabel('${\\rm Frecuencia\;normalizada}$', fontsize=fs)
leg = plt.legend(loc=1, frameon=False, fontsize=fs)
ax.set_yticklabels([])

plt.savefig('kalman_filter_steady_state_example.pdf', bbox_inches='tight')
plt.show()

