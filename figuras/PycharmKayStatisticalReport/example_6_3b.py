import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import math

from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')


###########################################
# PARAMETEROS - Esto puede ser modificado #
###########################################

# coordinates
# antenna 1
xi = 0
yi = 0
# antenna 1
x1 = -4
y1 = -1
# antenna N
xN = 4
yN = -1
# nominal
xn = 1
yn = 7
# source
xs = 3.5
ys = 8.5

# plot axis max values
xmin = -5
xmax = 6

ymin = -2
ymax = 10

#####################
# END OF PARAMETERS #
#####################

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fontsize = 10
markersize = 6

fig = plt.figure(0, figsize=(3, 3), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# antennas
plt.plot(x1, y1, 'k.', markersize=markersize)
plt.plot(xi, yi, 'k.', markersize=markersize)
plt.plot(xN, yN, 'k.', markersize=markersize)
# nominal position
plt.plot(xn, yn, 'k.', markersize=markersize)
# source
plt.plot(xs, ys, 'k.', markersize=markersize)
# Rni
plt.plot([xi, xn], [yi, yn], 'k', linewidth=1)
# Ri
plt.plot([xi, xs], [yi, ys], 'k', linewidth=1)
# delta xs
plt.plot([xn, xs], [yn, yn], 'k--', linewidth=1, dashes=(4, 2))
# delta ys
plt.plot([xs, xs], [yn, ys], 'k--', linewidth=1, dashes=(4, 2))

# angle
alpha_i = math.atan((yn-yi)/(xn-xi))
alphas = np.linspace(0, alpha_i, 20)
d = 1
plt.plot(d*np.cos(alphas), d*np.sin(alphas), 'k', linewidth=1)
plt.plot([xi, xi+1.5], [yi, yi], 'k--', linewidth=1, dashes=(4, 2))

# dots
l = np.array(((xi+x1)/2, (yi+y1)/2))
angle = math.atan((yi-y1)/(xi-x1))*180/math.pi
trans_angle = plt.gca().transData.transform_angles(np.array((angle,)), l.reshape((1, 2)))[0]
plt.text(l[0], l[1], '$\cdots$', fontsize=16, ha='center', va='center', rotation=trans_angle)
l = np.array(((xi+xN)/2, (yi+yN)/2))
trans_angle = plt.gca().transData.transform_angles(np.array((-angle,)), l.reshape((1, 2)))[0]
plt.text(l[0], l[1], '$\cdots$', fontsize=16, ha='center', va='center', rotation=trans_angle)

# labels
plt.text((xi+xn)/2-0.1, (yi+yn)/2, '$R_{n_i}$', fontsize=fontsize, ha='right', va='center')
plt.text((xi+xs)/2+0.2, (yi+ys)/2, '$R_i$', fontsize=fontsize, ha='left', va='center')
plt.text(xn+0.8, yn, '$\delta x_s$', fontsize=fontsize, ha='left', va='bottom')
plt.text(xs+0.15, (yn+ys)/2-0.2, '$\delta y_s$', fontsize=fontsize, ha='left', va='center')
plt.text(xi+0.9, yi+0.9, '$\\alpha_i$', fontsize=fontsize, ha='left', va='center')

plt.text(x1, y1-0.4, '$0$', fontsize=fontsize, ha='center', va='top')
plt.text(xi, yi-0.4, '$i$', fontsize=fontsize, ha='center', va='top')
plt.text(xN, yN-0.4, '$N-1$', fontsize=fontsize, ha='center', va='top')
plt.text(xi, y1-0.4, '${\\rm Antenas}$', fontsize=fontsize, ha='center', va='top')

plt.text(xn-0.1, yn, '${\\rm Posición\;nominal}$\n$(x_n,\,y_n)$', fontsize=fontsize, ha='right', va='bottom')
plt.text(xs+0.1, ys, '${\\rm Fuente}$\n$(x_s,\,y_s)$', fontsize=fontsize, ha='left', va='bottom')

# axis
xi = xmin
xf = xmin+2
yi = 2
yf = yi + 2
plt.annotate("", xytext=(xi, yi), xycoords='data', xy=(xf, yi), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=3, headlength=5, facecolor='black'))
plt.annotate("", xytext=(xi, yi), xycoords='data', xy=(xi, yf), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=3, headlength=5, facecolor='black'))

plt.text(xf, yi-0.1, '$x$', fontsize=fontsize, ha='center', va='top')
plt.text(xi+0.25, yf, '$y$', fontsize=fontsize, ha='left', va='center')


plt.axis('off')

# save as pdf image
plt.savefig('example_6_3b.pdf', bbox_inches='tight')

plt.show()

