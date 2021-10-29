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

# numero de muestras
N = 50
# transición de estados
a = 0.9
# varianza del ruido de exitación
var_u = 1
# Error MSE inicial M[-1|-1]
Mi = 1

#####################
# FIN DE PARAMETROS #
#####################

ns = np.arange(N)
Ms = np.zeros((N, ))

M_prev = Mi
for n in ns:
    var_n = n + 1
    M = (var_n * ((a ** 2) * M_prev + var_u)) / (var_n + (a ** 2) * M_prev + var_u)
    M_prev = M
    Ms[n] = M

# abscissa values
xmin = 0
xmax = N
ymin = 0
ymax = 4


# axis parameters
dx = 7
xmin_ax = xmin - dx
xmax_ax = xmax + 9

dy = 0.8
ymax_ax = ymax + dy
ymin_ax = ymin - dy


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.5
ytm = -1.5
# font size
fontsize = 12

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(5, 3), frameon=False)

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

plt.plot(ns, Ms, color=col10, linewidth=2, label='$\mathrm{polinomio}$')

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, ymax_ax, '$M[n|n]$', fontsize=fontsize, ha='left', va='center')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
for i in np.arange(10, N+5, 10):
    plt.plot([i, i], [0, xtl], 'k')
    plt.text(i, xtm, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(0.5, ymax+0.5, 0.5):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ytm, i, '${}$'.format(i), fontsize=fontsize, ha='right', va='center')


plt.axis('off')

# save as pdf image
plt.savefig('problem_13_23.pdf', bbox_inches='tight')

plt.show()

