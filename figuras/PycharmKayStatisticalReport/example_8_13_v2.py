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
# factor de olvido
lamb = 0.9
# condiciones iniciales
theta_i = np.zeros((p, ))
sigma_i = 1e5 * np.eye(p)

# señal de referencia (frecuencia, amplitud y fase)
omega_r = 2 * math.pi * 0.1
a_r = 1
p_r = 0
# señal de interferencia (frecuencia, amplitud y fase)
omega_i = omega_r
n = np.arange(N)
a_i = 10 * (1 + 0.5 * np.cos(2 * math.pi * n / N))
a_i = np.empty((N, ))
a_i[: int(N/2)+1] = 12
a_i[int(N/2)+1:] = 4
p_i = math.pi / 4

# Construcción de las señales
# referencia e interferencia
x_i = a_i * np.cos(omega_i * n + p_i)
x_r = a_r * np.cos(omega_r * n + p_r)
# estimador en cada paso
theta_n = np.zeros((N, p))
error = np.zeros((N, ))
y_r = np.zeros((N, ))
# Procesamiento
theta = theta_i
sigma = sigma_i
x_r_pad = np.concatenate((np.zeros((p - 1,)), x_r))
for i in n:
    h = x_r_pad[i + (p - 1): i - 1 if i - 1 >= 0 else None: -1]
    e = x_i[i] - h @ theta
    K = (sigma @ h) / (lamb ** i + h @ sigma @ h)
    theta = theta + K * e
    sigma = (np.eye(p) - K[:, None] @ h[None, :]) @ sigma
    theta_n[i, :] = theta
    error[i] = x_i[i] - theta @ h
    y_r[i] = theta @ h


# valores verdaderos
h0 = a_i * math.sin(9*math.pi/20)/math.sin(math.pi/5)
h1 = -a_i * math.sin(math.pi/4) / math.sin(math.pi/5)
#h0 = a_i * math.cos(math.pi/4) - h1 * math.cos(math.pi/5)
#print("h[0] = {0:f}, h[1] = {1:f}".format(h0, h1))

ms = 3
fs = 12
ymax = 16

fig = plt.figure(0, figsize=(9, 6), frameon=False)

ax = plt.subplot2grid((9, 1), (0, 0), rowspan=3, colspan=1)
plt.xlim(0, N-1)
plt.ylim(-ymax, ymax)
plt.plot(n, x_i, linestyle='-', color='k', marker='s', markersize=ms, label='$x[n]$')
plt.plot(n, y_r, linestyle='-', color='r', marker='s', markersize=ms, label='$\hat{x}[n]$')
ax.set_xticklabels([])
leg = plt.legend(loc='center', bbox_to_anchor=(0.92, 0.83), frameon=False, fontsize=fs)
ax.set_ylabel('${\\rm Interferencia}'+'\n'+'${\\rm y\;estimaci\\acute{o}n}$', fontsize=fs)

ax = plt.subplot2grid((9, 1), (3, 0), rowspan=3, colspan=1)
plt.xlim(0, N-1)
plt.ylim(-ymax, ymax)
plt.plot(n, error, linestyle='-', marker='s', color='k', markersize=ms, lw=1)
ax.set_xticklabels([])
ax.set_ylabel(r'$\epsilon[n]=x[n]-\hat{x}[n]$', fontsize=fs)

ax = plt.subplot2grid((9, 1), (6, 0), rowspan=3, colspan=1)
# e = hd-h_est
plt.xlim(0, N-1)
plt.plot(n, theta_n[:, 0], linestyle='-', color='k', marker='s', markersize=ms, label='$\hat{h}_n[0]$')
plt.plot(n, theta_n[:, 1], linestyle='-', color='r', marker='s', markersize=ms, label='$\hat{h}_n[1]$')
plt.plot(n, h0, linestyle='--', lw=1, color='grey')
plt.plot(n, h1, linestyle='--', lw=1, color='grey')
ax.set_xlabel(r'$n$', fontsize=fs)
ax.set_ylabel('${\\rm Par\\acute{a}metros\;del\;filtro}$', fontsize=fs)
leg = plt.legend(loc='center', bbox_to_anchor=(0.4, 0.5), frameon=False, fontsize=fs)


plt.savefig('example_8_13.pdf', bbox_inches='tight')

fig = plt.figure(1, figsize=(9, 5), frameon=False)
plt.plot(n[p:], x_i[p:] - y_r[p:], linestyle='-', color='k', marker='s', markersize=ms)


plt.show()

