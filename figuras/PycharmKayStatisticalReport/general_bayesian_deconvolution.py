import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy import signal
import math
import matplotlib.colors as colors
from matplotlib import cm

from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# auxiliar function for plot ticks of equal length in x and y axis despite its scales.
def convert_display_to_data_coordinates(transData, length=10):
    # create a transform which will take from display to data coordinates
    inv = transData.inverted()
    # transform from display coordinates to data coordinates in x axis
    data_coords = inv.transform([(0, 0), (length, 0)])
    # get the length of the segment in data units
    yticks_len = data_coords[1, 0] - data_coords[0, 0]
    # transform from display coordinates to data coordinates in y axis
    data_coords = inv.transform([(0, 0), (0, length)])
    # get the length of the segment in data units
    xticks_len = data_coords[1, 1] - data_coords[0, 1]
    return xticks_len, yticks_len


#####################################
# PARAMETERS - This can be modified #
#####################################

# proceso AR(1)
a1 = -0.95
sigma_u = 1
# número de muestras
N = 50
# SNR
SNR = 5

#####################
# END OF PARAMETERS #
#####################

# Cálculo de la ACF de la señal
# Respuesta en frecuencia del filtro generador de s[n]
nfft = 512
w, S = signal.freqz([1], [1, a1], worN=nfft, whole=True)
# PSD de s[s]
P_ss = np.square(np.absolute(S))
# Autocorrelacion del ruido coloreado
r_ss = np.real(np.fft.ifft(P_ss))

# Para verificación: cálculo alternativo de la ACF
r_ss2 = sigma_u / (1 - a1 ** 2) * np.power(-a1, np.arange(N))
# print(r_ss[0: N] - r_ss2)

# Construcción de la matriz A (filtro de Wiener)
C_s = linalg.toeplitz(r_ss[: N])
# Potencia del ruido contaminante para la SNR dada
sigma_w = r_ss[0] / (pow(10, SNR / 10))
A = C_s @ np.linalg.inv(C_s + sigma_w * np.eye(N))


# Generación de las realizaciones de los procesos
np.random.seed(7)
# u[n]: WGN de potencia sigma_u
u = math.sqrt(sigma_u) * np.random.randn(N)
# s[n]: proceso AR(1)
s = signal.lfilter([1], [1, a1], u)
w = math.sqrt(sigma_w) * np.random.randn(N)
# x[n] = s[n] + w[n]
x = s + w
# hat_s: estimación de s mediante el filtro de Wiener
hat_s = A @ x

# vector de la frequencia normalizada
# numero de muestras en frecuencia (nnft/2 + 1)
nf = nfft // 2
f = np.arange(nf+1)/(2 * nf)  # f[0]=0, f[nfft//2+1]=0.5
nf = nf + 1
n = np.arange(N)

# conversion de P_ss a dB.
P_ss = 10 * np.log10(P_ss[:nf])

fs = 12
ms = 3

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col1 = scalarMap.to_rgba(0)
col2 = scalarMap.to_rgba(0.95)

fig = plt.figure(0, figsize=(9, 4), frameon=False)

ax = plt.subplot2grid((8, 3), (0, 0), rowspan=8, colspan=1)
plt.xlim(f[0], f[-1])
plt.ylim(-10, 30)
plt.plot(f, P_ss, 'k-', lw=2)

ax.set_xlabel('${\\rm Frecuencia\; normalizada,\;}f$', fontsize=fs)
ax.set_ylabel('$P_{ss}(f){\\rm\;\;(dB)}$', fontsize=fs)
xticks = np.arange(0, f[-1]+0.01, 0.1)
ax.set_xticks(xticks)

ax = plt.subplot2grid((8, 3), (0, 1), rowspan=8, colspan=2)
plt.xlim(0, N)
plt.ylim(-6.9, 6.9)
plt.plot(n, s, linestyle='-', marker='s', color='k', markersize=ms, lw=1, zorder=2, label='$s[n]$')
plt.plot(n, x, linestyle='-', marker='s', color=col2, markersize=ms, lw=1, zorder=1, label='$x[n]$')
plt.plot(n, hat_s, linestyle='-', marker='s', color=col1, markersize=ms, lw=1, zorder=3, label='$\hat{s}[n]$')
leg = plt.legend(fontsize=fs, frameon=False)
ax.set_xlabel('${\\rm N\\acute{u}mero\; de\; muestra,\;}n$', fontsize=fs)

# save as pdf image
plt.savefig('general_bayesian_deconvolution.pdf', bbox_inches='tight')


plt.show()



















