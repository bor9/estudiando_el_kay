import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace
import math

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rc('mathtext', fontset='cm')

#####################################
# PARAMETERS - This can be modified #
#####################################

# laplace pdf variance
var = 1
# laplace pdf mean
mu = 10
# samples number
N1 = 5
N2 = 6

#####################
# END OF PARAMETERS #
#####################

# sort samples with laplace distribution
np.random.seed(seed=12)
x2 = laplace.rvs(loc=4, scale=math.sqrt(2 * var), size=N2)
x2 = np.sort(x2)
x1 = np.delete(x2, 1)

mu_min = np.amin(x2) - 0.5
mu_max = np.amax(x2) + 0.5

mus = np.linspace(mu_min, mu_max, 500)

phi1 = np.zeros(mus.shape)
for xn in x1:
    phi1 += np.absolute(xn-mus)
phi2 = np.zeros(mus.shape)
for xn in x2:
    phi2 += np.absolute(xn-mus)
phi1 = -phi1
phi2 = -phi2

ymin1 = np.amin(phi2)
ymax1 = np.amax(phi1)

xmin_ax = mu_min
xmax_ax = mu_max
ymin_ax1 = ymin1 - 2
ymax_ax1 = ymax1 + 2

ymax_ax2 = N2 + 1
ymin_ax2 = -N2 - 1

# font size
fontsize = 14

fig = plt.figure(0, figsize=(10, 6), frameon=False)

ax = plt.subplot2grid((8, 8), (0, 0), rowspan=4, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax1, ymax_ax1)

plt.plot(mus, phi1, 'k-', lw=2)
plt.plot([x1, x1], [ymin_ax1 * np.ones(x1.shape), ymax_ax1 * np.ones(x1.shape)], 'k--', lw=1)
for i in np.arange(N1):
    plt.text(x1[i], ymax_ax1 + 0.8, r'$x_{{({})}}$'.format(i + 1), fontsize=fontsize, ha='center', va='baseline')
ax.set_xticklabels([])
plt.ylabel('$-\sum\limits_{n=0}^{N-1}|x[n]-\mu|$', fontsize=fontsize)
plt.text((xmin_ax+xmax_ax)/2, ymax_ax1 + 3.5, '$N\;\mathrm{{impar}}\;(N={})$'.format(N1), fontsize=fontsize,
         ha='center', va='baseline')

ax = plt.subplot2grid((8, 8), (4, 0), rowspan=4, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax2, ymax_ax2)
levels = N1 - 2 * np.arange(N1 + 1)

plt.plot([np.insert(x1, 0, [xmin_ax]), np.concatenate([x1, [xmax_ax]])], [levels, levels], 'k-', lw=2)
plt.plot([x1, x1], [ymin_ax2 * np.ones(x1.shape), ymax_ax2 * np.ones(x1.shape)], 'k--', lw=1)
plt.plot([xmin_ax, xmax_ax], [0, 0], 'k-', lw=1)
plt.xlabel('$\mu$', fontsize=fontsize)
plt.ylabel('$\sum\limits_{n=0}^{N-1}\mathrm{sig}(x[n]-\mu)$', fontsize=fontsize)

ax = plt.subplot2grid((8, 8), (0, 4), rowspan=4, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax1, ymax_ax1)

plt.plot(mus, phi2, 'k-', lw=2)
plt.plot([x2, x2], [ymin_ax1 * np.ones(x2.shape), ymax_ax1 * np.ones(x2.shape)], 'k--', lw=1)
for i in np.arange(N2):
    plt.text(x2[i], ymax_ax1 + 0.8, r'$x_{{({})}}$'.format(i + 1), fontsize=fontsize, ha='center', va='baseline')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.text((xmin_ax+xmax_ax)/2, ymax_ax1 + 3.5, '$N\;\mathrm{{par}}\;(N={})$'.format(N2), fontsize=fontsize,
         ha='center', va='baseline')

ax = plt.subplot2grid((8, 8), (4, 4), rowspan=4, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax2, ymax_ax2)
levels = N2 - 2 * np.arange(N2 + 1)

plt.plot([np.insert(x2, 0, [xmin_ax]), np.concatenate([x2, [xmax_ax]])], [levels, levels], 'k-', lw=2)
plt.plot([x2, x2], [ymin_ax2 * np.ones(x2.shape), ymax_ax2 * np.ones(x2.shape)], 'k--', lw=1)
plt.plot([xmin_ax, xmax_ax], [0, 0], 'k-', lw=1)
plt.xlabel('$\mu$', fontsize=fontsize)
ax.set_yticklabels([])

plt.savefig('example_mle_mean_laplace.pdf', bbox_inches='tight')
plt.show()




