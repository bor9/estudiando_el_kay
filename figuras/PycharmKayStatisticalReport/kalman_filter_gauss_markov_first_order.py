import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal
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
N = 200
# varianza del ruido
var_u = 0.1
# estadística de las condiciones iniciales
mu_s = 5
var_s = 1
# parámetro del proceso Gauss-Markov
a = 0.98

#####################
# FIN DE PARAMETROS #
#####################
# abscissa values
xmin = 0
xmax = N
ymin = -1
ymax = 7

n = np.arange(N)
# condición inicial s[-1]
np.random.seed(5)
s_i = np.random.normal(mu_s, math.sqrt(var_s), 1)
# muestras de ruido
u = np.random.normal(0, math.sqrt(var_u), N)
# proceso de gauss-markov
# las condiciones iniciales pueden calcularse como:
# z_i = signal.lfiltic([1], [1, -a], s_i)
s, z_f = signal.lfilter([1], [1, -a], u, zi=a * s_i)

# filtrado con un for - para comparar con el resultado de lfilter con condiciones iniciales
s2 = np.zeros((N,))
s_prev = s_i
for i in n:
    s2[i] = a * s_prev + u[i]
    s_prev = s2[i]

print(np.amax(np.abs(s - s2)))

# axis parameters
dx = 6
xmin_ax = xmin - dx
xmax_ax = xmax + 12

dy = 0.6
ymax_ax = ymax
ymin_ax = ymin - dy


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.7
ytm = -2
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

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(n, s, 'k-', lw=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(3, ymax_ax, '$s[n]$', fontsize=fontsize, ha='left', va='center')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
for i in np.arange(10, xmax+1, 10):
    plt.plot([i, i], [0, xtl], 'k')
for i in np.arange(20, xmax+1, 20):
    plt.text(i, xtm, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(1, ymax, 1):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ytm, i, '${}$'.format(i), fontsize=fontsize, ha='right', va='center')


plt.axis('off')

# save as pdf image
plt.savefig('kalman_filter_gauss_markov_first_order.pdf', bbox_inches='tight')

plt.show()

