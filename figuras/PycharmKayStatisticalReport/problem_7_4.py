import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')

#####################################
# PARAMETERS - This can be modified #
#####################################

# exponential PDF
sigma_sqr = 1
lambd = 1/sigma_sqr

#####################
# END OF PARAMETERS #
#####################

# abscissa values
nmin = 70
nmax = 140

n = np.linspace(nmin, nmax, 300)

# axis parameters
dx = 10
xmin_ax = nmin
xmax_ax = nmax

var1 = 2 / n
var2 = 1 / n + 100 / np.square(n)

ymax_ax = 0.03
ymin_ax = 0.013

# ticks labels margin
xtm = 0.0016
ytm = 1.5
# font size
fontsize = 11
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)


fig = plt.figure(0, figsize=(5, 3), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)


plt.plot(n, var1, color=col10, linewidth=2, label='$\mathrm{var}(\hat{\\theta}_1)$')
plt.plot(n, var2, color=col20, linewidth=2, label='$\mathrm{var}(\hat{\\theta}_2)$')

plt.plot([100, 100], [ymin_ax, 0.02],'k--', linewidth=1)
plt.plot([xmin_ax, 100], [0.02, 0.02],'k--', linewidth=1)


xticks = np.arange(nmin, nmax+1, 10)
plt.xticks(xticks, [])
for t in xticks:
    ax.text(t, ymin_ax-xtm, '${:d}$'.format(t), fontsize=fontsize, ha='center', va='baseline')
yticks = np.arange(0.015, 0.031, 0.005)
plt.yticks(yticks, [])
for t in yticks:
    ax.text(nmin-ytm, t-0.0002, '${:.3f}$'.format(t), fontsize=fontsize, ha='right', va='center')

# labels
plt.text((xmin_ax+xmax_ax)/2, ymin_ax-1.8*xtm, '$N$', fontsize=fontsize+1, ha='center', va='baseline')
plt.text(nmin-11, (ymin_ax+ymax_ax)/2, '$\mathrm{Varianza}$', fontsize=fontsize+1, ha='center', va='center',
         rotation=90)

leg = plt.legend(loc=1, fontsize=fontsize+1, frameon=False)


# save as pdf image
plt.savefig('problem_7_4.pdf', bbox_inches='tight')

plt.show()

