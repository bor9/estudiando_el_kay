import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
import matplotlib.colors as colors

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


##########################################
# PARAMETROS - Esto puede ser modificado #
##########################################

# coeficientes del polinomio - a estimar
coefs = [0.0001, -0.016, 0.73, 1]
# numero de muestras
N = 100
# varianza del ruido
var_w = 10

#####################
# FIN DE PARAMETROS #
#####################
# abscissa values
xmin = 0
xmax = 100
ymin = -2
ymax = 19

n = np.arange(N)
f = np.polyval(coefs, n)
x = f + np.random.normal(0, math.sqrt(10), N)

# calculo del estimador
p =len(coefs)
H = np.vander(n, p, increasing=True)
invHTH = inv(np.transpose(H).dot(H))
coefs_est = np.flipud(invHTH.dot(np.transpose(H).dot(x)))
coefs_var = np.flipud(np.diagonal(var_w * invHTH))
print(coefs_est)
print(coefs_var)
f_est = np.polyval(coefs_est, n)


# axis parameters
dx = 7
xmin_ax = xmin - dx
xmax_ax = xmax + 9

dy = 0
ymax_ax = ymax
ymin_ax = ymin


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -1.7
ytm = -2
# font size
fontsize = 12

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(9, 4), frameon=False)

# PLOT OF P_ww
ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(n, f, color=col10, linewidth=2, label='$\mathrm{polinomio}$')
plt.plot(n, x, 'k.', markersize=5, label='$x[n]$')
plt.plot(n, f_est, color=col20, linewidth=2, label='$\mathrm{polinomio\;estimado}$')

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
for i in np.arange(20, xmax+1, 20):
    plt.plot([i, i], [0, xtl], 'k')
    plt.text(i, xtm, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

yticks = [5, 10, 15]
for i in yticks:
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ytm, i, '${}$'.format(i), fontsize=fontsize, ha='right', va='center')

# legend
leg = plt.legend(loc=(0.7, 0.9), fontsize=12, frameon=True)

plt.axis('off')

# save as pdf image
plt.savefig('example_4_1.pdf', bbox_inches='tight')

plt.show()

