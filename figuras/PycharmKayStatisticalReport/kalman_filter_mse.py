import matplotlib.pyplot as plt
import numpy as np
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

# coeficientes del filtro de Kalman
a = 0.99
var_u = 0.1
var_n = 0.9
Mi = 1
N = 31

#####################
# FIN DE PARAMETROS #
#####################
# abscissa values
xmin = 0
xmax = N
ymin = 0
ymax = 1.3

ns = np.arange(N)

Ms = np.zeros((N,))
Mps = np.zeros((N,))
Ks = np.zeros((N,))

M = Mi
for n in ns:
    # predicción del MSE mínimo
    Mp = (a ** 2) * M + var_u
    # ganancia de Kalman
    K = Mp / ((var_n ** (n + 1)) + Mp)
    # MSE mínimo
    M = (1 - K) * Mp
    # almacenamiento de valores
    Ms[n] = M
    Mps[n] = Mp
    Ks[n] = K


# axis parameters
xmin_ax = xmin - 2
xmax_ax = xmax + 1

dy = 0.2
ymax_ax = ymax + 0.1
ymin_ax = ymin - dy


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.15
ytm = -0.5
# font size
fontsize = 12

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(9, 3), frameon=False)

# PLOT OF P_ww
ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

nn = np.zeros((2 * N + 1,))
nn[1::2] = ns
nn[2::2] = ns
nn[0] = -1
Mm = np.zeros((2 * N + 1,))
Mm[1::2] = Mps
Mm[2::2] = Ms
Mm[0] = Mi


# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data', zorder=1,
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data', zorder=1,
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))


plt.plot(ns, Ms, 'ks', markersize=4, label='$M[n|n]$', zorder=2)
plt.plot(ns, Mps, 'rs', markersize=4, label='$M[n|n-1]$', zorder=2)
plt.plot(nn, Mm, 'k-', lw=2, zorder=1)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
for i in np.arange(10, xmax+1, 10):
    plt.plot([i, i], [0, xtl], 'k')
    plt.text(i, xtm, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(0.2, ymax, 0.2):
    plt.plot([0, ytl], [i, i], 'k')

for i in np.arange(0.2, 0.8, 0.2):
    plt.text(ytm, i, '${:.1f}$'.format(i), fontsize=fontsize, ha='right', va='center')
i = 1.2
plt.text(ytm, i, '${:.1f}$'.format(i), fontsize=fontsize, ha='right', va='center')



# legend
leg = plt.legend(fontsize=12, frameon=False)

plt.axis('off')

# save as pdf image
plt.savefig('kalman_filter_mse.pdf', bbox_inches='tight')

plt.show()

