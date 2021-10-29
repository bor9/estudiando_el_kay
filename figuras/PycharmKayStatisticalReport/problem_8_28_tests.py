import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, linalg
from scipy.optimize import minimize

from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]


def ja(a):
    c = np.concatenate(([a[0]], np.zeros((N - (p + 1), ))))
    r = np.concatenate((a, [1], np.zeros((N - (p + 1), ))))
    AT = linalg.toeplitz(c, r)
    #epsilon = 1e-15
    epsilon  = 0
    return hd @ AT.T @ linalg.inv(AT @ AT.T + epsilon * np.eye(N - p)) @ AT @ hd


#respuesta al impulso deseada: sinc
N = 50  # numero par
fc = 0.2
nf = 1024
n = np.arange(-N/2, N/2+1)
N += 1
f = np.arange(nf)/(2 * nf)

# parámetros del filtro a diseñar
p = 10
q = p - 1

# respuesta al impulso
hd = 2 * fc * np.sinc(2 * fc * n)  # * np.hanning(N)
# respuesta en frecuencia
_, Hd = signal.freqz(hd, a=1, worN=nf, whole=False, plot=None)

# calculo de a empleando scipy.optimize.minimize
a0 = np.zeros((p, ))
# a0[0] = 0.132
# a0[0] = -0.14
# a0[0] = -1.6
# a0[0] = -3
# a0[0] = -6
# a0[3] = 12
a0[-1] = -3
res = minimize(ja, a0)
# res = minimize(ja, a0, method='Powell', options={'disp': True})
a_est = res.x
a_est = np.concatenate(([1], a_est[::-1]))


# cálculo de la matrix G empleando hat_a
delta = np.zeros((N, ))
delta[0] = 1
g = signal.lfilter([1], a_est, delta)
G = linalg.toeplitz(g, np.concatenate(([g[0]], np.zeros((p - 1, )))))
#b_est = linalg.inv(G.T @ G) @ G.T @ hd
epsilon = 1e-15
epsilon = 0
b_est = linalg.solve(G.T @ G + epsilon * np.eye(p), G.T @ hd)
print(G)
print(a_est)
print(b_est)

_, H_est = signal.freqz(b_est, a_est, worN=nf, whole=False, plot=None)
h_est = signal.lfilter(b_est, a_est, delta, axis=- 1, zi=None)


ms = 3
fs = 12
n = np.arange(N)
fig = plt.figure(0, figsize=(9, 5), frameon=False)
ax = plt.subplot2grid((8, 2), (0, 0), rowspan=6, colspan=1)
plt.xlim(0, N-1)
plt.ylim(np.amin(hd)-0.02, np.amax(hd)+0.02)
plt.plot(n, hd, linestyle='-', marker='s', color='k', markersize=ms, lw=1, label='${\\rm deseada}$')
plt.plot(n, h_est, linestyle='-', marker='s', color='r', markersize=ms, lw=1, label='${\\rm estimada}$')
leg = plt.legend(loc=1, frameon=False, fontsize=fs)
ax.set_xticklabels([])
ax.set_ylabel('${\\rm Respuesta\;al\;impulso}$', fontsize=fs)


ax = plt.subplot2grid((8, 2), (6, 0), rowspan=2, colspan=1)
e = hd-h_est
plt.xlim(0, N-1)
plt.ylim(np.amin(e)-0.001, np.amax(e)+0.001)
plt.plot(n, e, linestyle='-', marker='s', color='k', markersize=ms)
ax.set_xlabel(r'$n$', fontsize=fs)
ax.set_ylabel(r'$\epsilon[n]$', fontsize=fs)

ax = plt.subplot2grid((8, 2), (0, 1), rowspan=8, colspan=1)
plt.xlim(0, 0.5)
plt.ylim(-55, 18)
plt.plot(f, 10 * np.log10(np.abs(Hd)), 'k', label='${\\rm deseada}$')
plt.plot(f, 10 * np.log10(np.abs(H_est)), 'r', label='${\\rm estimada}$')
ax.set_xlabel('${\\rm Frecuencia\;normalizada}$', fontsize=fs)
ax.set_ylabel('${\\rm Respuesta\;en\;frecuencia\;(dB)}$', fontsize=fs)
leg = plt.legend(loc=1, frameon=False, fontsize=fs)


#plt.savefig('example_8_11.pdf', bbox_inches='tight')
plt.show()


