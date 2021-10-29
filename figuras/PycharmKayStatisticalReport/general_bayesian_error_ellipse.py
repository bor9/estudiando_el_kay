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

# parámetros de la elipse
a = 4
b = 2.5

#####################
# END OF PARAMETERS #
#####################

# foco
c = math.sqrt(a ** 2 - b ** 2)

# puntos (x, y) de la elipse
t = np.linspace(0, 2 * math.pi, 300)
xe = a * np.cos(t)
ye = b * np.sin(t)

xmax = a
xmin = -xmax
ymax = b
ymin = -ymax

# axis parameters
dx = 1.5
xmax_ax = xmax + dx
xmin_ax = xmin - dx
ymax_ax = ymax + dx
ymin_ax = ymin - dx

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.55
# font size
fontsize = 14
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0.1)
col20 = scalarMap.to_rgba(1)


f = plt.figure(0, figsize=(5, 3), frameon=False)
ax = f.add_subplot(111)

plt.axis('equal')
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

f.canvas.draw()
ymin_ax, ymax_ax = ax.get_ylim()


# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)

# elipse
plt.plot(xe, ye, 'k-', lw=2.5)
# focos
plt.plot(c, 0, 'k.', markersize=10)
plt.plot(-c, 0, 'k.', markersize=10)
# parámetros
plt.plot([0, c], [0, 0], 'g', zorder=1, lw=2)
plt.plot([-a, 0], [0, 0], color='r', zorder=1, lw=2)
plt.plot([0, 0], [0, b], color='b', zorder=1, lw=2)
plt.plot([0, c], [b, 0], color='r', zorder=1, lw=2)

# labels
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.3, ymax_ax, '$y$', fontsize=fontsize, ha='right', va='center')
plt.text(-c, -0.8, '$F_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(c, -0.8, '$F_1$', fontsize=fontsize, ha='center', va='baseline')

plt.text(-a/2, 0.25, '$a$', fontsize=fontsize, color='r', ha='center', va='baseline')
plt.text(c/2, xtm, '$c$', fontsize=fontsize, color='g', ha='center', va='baseline')
plt.text(-0.25, b/2, '$b$', fontsize=fontsize, color='b', ha='right', va='center')
plt.text(2+0.2, b/2+0.2, '$a$', fontsize=fontsize, color='r', ha='right', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('general_bayesian_error_ellipse.pdf', bbox_inches='tight')

plt.show()

