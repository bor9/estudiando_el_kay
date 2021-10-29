import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math

from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')

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



###########################################
# PARAMETEROS - Esto puede ser modificado #
###########################################

# coordinates
# antenna 1
x1 = 0
y1 = 0
# antenna 0
x0 = -4.5
y0 = 0
# antenna 2
x2 = 4.5
y2 = 0
# nominal
xn = 0
yn = 6
# source
xs = -2
ys = 7

# plot axis max values
xmin = -5
xmax = 6

ymin = -2
ymax = 7.5

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

fig = plt.figure(0, figsize=(3, 2), frameon=False)
ax = fig.add_subplot(111)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# for right angle
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=12)


# antennas
plt.plot(x1, y1, 'k.', markersize=markersize)
plt.plot(x0, y0, 'k.', markersize=markersize)
plt.plot(x2, y2, 'k.', markersize=markersize)
# nominal position
plt.plot(xn, yn, 'k.', markersize=markersize)
# source
plt.plot(xs, ys, 'k.', markersize=markersize)
# Rni
plt.plot([x1, xn], [y1, yn], 'k', linewidth=1)
plt.plot([x0, xn], [y0, yn], 'k', linewidth=1)
plt.plot([x2, xn], [y2, yn], 'k', linewidth=1)

# angle
alpha_i = math.atan((yn-y0)/(xn-x0))
alphas = np.linspace(0, alpha_i, 20)
d = 1
plt.plot(d*np.cos(alphas)+x0, d*np.sin(alphas), 'k', linewidth=0.5)
plt.plot([x0, x0+1.5], [y0, y0], 'k--', linewidth=1, dashes=(4, 2))

plt.plot(-d*np.cos(alphas)+x2, d*np.sin(alphas), 'k', linewidth=0.5)
plt.plot([x2, x2-1.5], [y2, y2], 'k--', linewidth=1, dashes=(4, 2))
plt.plot([x1, x1+1.5], [y1, y1], 'k--', linewidth=1, dashes=(4, 2))

# right angle
d = 0.5
plt.plot([0, ytl], [xtl, xtl], 'k', linewidth=0.5)
plt.plot([ytl, ytl], [0, xtl], 'k', linewidth=0.5)

# labels
plt.text(x0+1.1, y0+0.6, '$\\alpha$', fontsize=fontsize, ha='left', va='center')
plt.text(x2-1.1, y2+0.6, '$\\alpha$', fontsize=fontsize, ha='right', va='center')


plt.text(x0, y0-0.5, '$0$', fontsize=fontsize, ha='center', va='top')
plt.text(x1, y1-0.5, '$1$', fontsize=fontsize, ha='center', va='top')
plt.text(x2, y2-0.5, '$2$', fontsize=fontsize, ha='center', va='top')

plt.text(xn+0.5, yn, '${\\rm Posición\;nominal}$', fontsize=fontsize, ha='left', va='center')
plt.text(xs-0.4, ys, '${\\rm Fuente}$', fontsize=fontsize, ha='right', va='center')

ya = -1.8
delta = 0.05
ax.annotate('', xy=(x0-delta, ya), xytext=(x1+delta, ya), ha='left', arrowprops=dict(arrowstyle='<->',
                                                                                     shrinkA=0, shrinkB=0))
ax.annotate('', xy=(x1-delta, ya), xytext=(x2+delta, ya), ha='left', arrowprops=dict(arrowstyle='<->',
                                                                                     shrinkA=0, shrinkB=0))

plt.text((x0+x1)/2, ya, '$d$', fontsize=fontsize, ha='center', va='center',
         bbox=dict(fc='white', ec='white', alpha=1))
plt.text((x1+x2)/2, ya, '$d$', fontsize=fontsize, ha='center', va='center',
         bbox=dict(fc='white', ec='white', alpha=1))

plt.axis('off')

# save as pdf image
plt.savefig('example_6_3c.pdf', bbox_inches='tight')

plt.show()

