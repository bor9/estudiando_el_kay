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


def coordinates_change(x, y):
    xp = x * np.cos(theta) - y * np.sin(theta)
    yp = x * np.sin(theta) + y * np.cos(theta)
    return xp, yp



#####################################
# PARAMETERS - This can be modified #
#####################################

# parámetros de la elipse
a = 4
b = 2.5
theta = math.pi / 6

#####################
# END OF PARAMETERS #
#####################

# foco
c = math.sqrt(a ** 2 - b ** 2)

# puntos (x, y) de la elipse
t = np.linspace(0, 2 * math.pi, 300)
xe = a * np.cos(t)
ye = b * np.sin(t)

xe_rot = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
ye_rot = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)


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


f = plt.figure(0, figsize=(10, 4), frameon=False)
ax = plt.subplot2grid((1, 8), (0, 0), rowspan=1, colspan=4)

plt.axis('equal')
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

f.canvas.draw()
ymin_ax, ymax_ax = ax.get_ylim()
xmin_ax, xmax_ax = ax.get_xlim()


# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)

# rotated axis arrows
xpi, ypi = coordinates_change(xmin_ax, 0)
xpf, ypf = coordinates_change(xmax_ax, 0)
plt.annotate("", xytext=(xpi, ypi), xycoords='data', xy=(xpf, ypf), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)
xpi, ypi = coordinates_change(0, ymax_ax)
xpf, ypf = coordinates_change(0, ymin_ax)
plt.annotate("", xytext=(xpf, ypf), xycoords='data', xy=(xpi, ypi), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)

# elipse
plt.plot(xe_rot, ye_rot, 'k-', lw=2.5)
# angle
t1 = np.linspace(0, theta, 30)
lt = 1
xt = lt * np.cos(t1)
yt = lt * np.sin(t1)
plt.plot(xt, yt, 'k', lw=1)

# labels
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.3, ymax_ax, '$y$', fontsize=fontsize, ha='right', va='center')
xp, yp = coordinates_change(xmax_ax, xtm)
plt.text(xp, yp, '$x\'$', fontsize=fontsize, ha='center', va='baseline')
xp, yp = coordinates_change(-0.3, ymax_ax)
plt.text(xp, yp, '$y\'$', fontsize=fontsize, ha='center', va='baseline')
plt.text(lt+0.1, 0.15, '$\\theta$', fontsize=fontsize, ha='left', va='baseline')

plt.axis('off')

# EJES ROTADOS

ax = plt.subplot2grid((1, 8), (0, 4), rowspan=1, colspan=4)

plt.axis('equal')
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

f.canvas.draw()
ymin_ax, ymax_ax = ax.get_ylim()
xmin_ax, xmax_ax = ax.get_xlim()


# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)

# rotated axis arrows
xpi, ypi = coordinates_change(xmin_ax, 0)
xpf, ypf = coordinates_change(xmax_ax, 0)
plt.annotate("", xytext=(xpi, ypi), xycoords='data', xy=(xpf, ypf), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)
xpi, ypi = coordinates_change(0, ymax_ax)
xpf, ypf = coordinates_change(0, ymin_ax)
plt.annotate("", xytext=(xpf, ypf), xycoords='data', xy=(xpi, ypi), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)

# angles
# theta
plt.plot(xt, yt, 'k', lw=1)
# alpha
xp = 2.5
yp = 3
plt.plot(xp, yp, 'k.', markersize=8)
alpha = math.atan(yp / xp)
plt.plot([0, xp], [0, yp], 'k', lw=2)
t2 = np.linspace(0, alpha, 30)
la = 1.8
xa = la * np.cos(t2)
ya = la * np.sin(t2)
plt.plot(xa, ya, 'k', lw=1)
# coordenadas en el eje original
plt.plot([xp, xp], [0, yp], 'k--', lw=1)
plt.plot([0, xp], [yp, yp], 'k--', lw=1)
# coordenadas en el eje rotado
r = math.sqrt(xp ** 2 + yp ** 2)
xpxp = r * np.cos(alpha-theta)
xpxpx, xpxpy = coordinates_change(xpxp, 0)
plt.plot([xpxpx, xp], [xpxpy, yp], 'k--', lw=1)
ypyp = r * np.sin(alpha-theta)
ypypx, ypypy = coordinates_change(0, ypyp)
plt.plot([ypypx, xp], [ypypy, yp], 'k--', lw=1)

# labels
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.3, ymax_ax, '$y$', fontsize=fontsize, ha='right', va='center')
xx, yy = coordinates_change(xmax_ax, xtm)
plt.text(xx, yy, '$x\'$', fontsize=fontsize, ha='center', va='baseline')
xx, yy = coordinates_change(-0.3, ymax_ax)
plt.text(xx, yy, '$y\'$', fontsize=fontsize, ha='center', va='baseline')
plt.text(lt+0.1, 0.15, '$\\theta$', fontsize=fontsize, ha='left', va='baseline')
plt.text(la+0.1, 0.4, '$\\alpha$', fontsize=fontsize, ha='left', va='baseline')
plt.text(xp, xtm, '$x_P$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.3, yp, '$y_P$', fontsize=fontsize, ha='right', va='center')
plt.text(xpxpx, xpxpy, '$x\'_P$', fontsize=fontsize, ha='left', va='top')
plt.text(ypypx, ypypy, '$y\'_P$', fontsize=fontsize, ha='right', va='top')
plt.text(r / 2 * np.cos(alpha) - 0.15, r / 2 * np.sin(alpha) + 0.15, '$r$', fontsize=fontsize, ha='center', va='center')
plt.text(xp + 0.3, yp + 0.3, '$P$', fontsize=fontsize, ha='center', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('general_bayesian_error_rotated_ellipse.pdf', bbox_inches='tight')

plt.show()

