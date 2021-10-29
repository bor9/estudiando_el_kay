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

# parametros del proceso de Gauss-Markov
a = 0.98
var_u = 0.1
mu_s_i = 5
var_s_i = 1
N = 150

#####################
# END OF PARAMETERS #
#####################

n = np.arange(N)

# evolución de la esperanza
mu_s = mu_s_i * (a ** (n + 1))
# evolución de la varianza
var_s = (a ** (2 * n + 2)) * var_s_i + (1 - a ** (2 * n + 2)) / (1 - (a ** 2)) * var_u
var_s_steady = (1 / (1 - a ** 2)) * var_u

# axis parameters
xmin = 0
xmax = N
dx = 15
xmin_ax = xmin - dx
xmax_ax = xmax + dx


ymin = 0
ymax = mu_s_i + 0.5
dy = 0.5
ymin_ax = ymin - dy
ymax_ax = ymax + dy


ymax_ax_2 = 3.5
ymin_ax_2 = ymax_ax_2 * ymin_ax / ymax_ax


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# ticks labels margin
xtm = -0.7
ytm = -5
# font size
fontsize = 12
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 3), frameon=False)

# gráfica de la evolución de la esperanza
ax = plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(n, mu_s, 'k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, ymax_ax, '$E(s[n])$', fontsize=fontsize, ha='left', va='center')

for i in np.arange(25, N+5, 25):
    plt.plot([i, i], [0, xtl], 'k')
    plt.text(i, xtm, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(1, ymax-0.5, 1):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ytm, i, '${:.0f}$'.format(i), fontsize=fontsize, ha='right', va='center')
i = 5
plt.plot([0, ytl], [i, i], 'k')
plt.text(ytm, i, '$\mu_s={:.0f}$'.format(i), fontsize=fontsize, ha='right', va='center')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.axis('off')

# gráfica de la evolución de la varianza

xtm2 = ymax_ax_2 * xtm / ymax_ax

ax = plt.subplot2grid((1, 4), (0, 2), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax_2, ymax_ax_2)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)


# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax_2), xycoords='data', xy=(0, ymax_ax_2), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(n, var_s, 'k', linewidth=2)
plt.plot([0, N-1], [var_s_steady, var_s_steady], 'k--', dashes=(5, 5), linewidth=1, zorder=0)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm2, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, ymax_ax_2, '${\\rm var}(s[n])$', fontsize=fontsize, ha='left', va='center')

for i in np.arange(25, N+5, 25):
    plt.plot([i, i], [0, xtl], 'k')
    plt.text(i, xtm2, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(2, ymax_ax_2, 1):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ytm, i, '${:.0f}$'.format(i), fontsize=fontsize, ha='right', va='center')
i = 1
plt.plot([0, ytl], [i, i], 'k')
plt.text(ytm, i, '$\sigma_s^2={:.0f}$'.format(i), fontsize=fontsize, ha='right', va='center')


plt.text(ytm, var_s_steady, '$\dfrac{\sigma_u^2}{1-a^2}$', fontsize=fontsize, ha='right', va='center')
plt.text(ytm, xtm2, '$0$', fontsize=fontsize, ha='right', va='baseline')

plt.axis('off')

# save as pdf image
plt.savefig('problem_13_3_mean_variance.pdf', bbox_inches='tight')


#######################
## Correlación y PSD ##
#######################

# autocorrelación
K = 150
k = np.arange(K)
r_ss = var_u / (1 - a ** 2) * (a ** k)

# PSD
fmax = 0.03
f = np.linspace(0, fmax, 300)
P_ss = var_u / (1 - 2 * a * np.cos(2 * math.pi * f) + a ** 2)


# axis parameters
kmin = 0
kmax = K
dx = 15
kmin_ax = xmin - dx
kmax_ax = xmax + dx

ymin = 0
ymax = var_s_steady + 0.5
dy = 0.3
ymin_ax = ymin - dy
ymax_ax = ymax + dy

# ticks labels margin
xtm = -0.4
ytm = -5

fig = plt.figure(1, figsize=(10, 3), frameon=False)

# gráfica de la evolución de la esperanza
ax = plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=2)

plt.xlim(kmin_ax, kmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(k, r_ss, 'k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$k$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, ymax_ax, '$r_{ss}[k]$', fontsize=fontsize, ha='left', va='center')

for i in np.arange(25, N+5, 25):
    plt.plot([i, i], [0, xtl], 'k')
    plt.text(i, xtm, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(1, ymax, 1):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ytm, i, '${:.0f}$'.format(i), fontsize=fontsize, ha='right', va='center')

plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, var_s_steady, '$\dfrac{\sigma_u^2}{1-a^2}$', fontsize=fontsize, ha='right', va='center')
plt.plot([0, ytl], [var_s_steady, var_s_steady], 'k')


plt.axis('off')


# gráfica de la PSD

ymax_ax_2 = 335
ymin_ax_2 = ymax_ax_2 * ymin_ax / ymax_ax

xtm2 = ymax_ax_2 * xtm / ymax_ax

ax = plt.subplot2grid((1, 4), (0, 2), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax_2, ymax_ax_2)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax_2), xycoords='data', xy=(0, ymax_ax_2), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))


fact = (K - 1) / fmax
plt.plot(f * fact, P_ss, 'k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm2, '$f$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, ymax_ax_2, '$P_{ss}(f)$', fontsize=fontsize, ha='left', va='center')

for i in np.arange(0.01, fmax + 0.001, 0.01):
    plt.plot([i * fact, i * fact], [0, xtl], 'k')
    plt.text(i * fact, xtm2, '${:.2f}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(100, ymax_ax_2, 100):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ytm, i, '${:.0f}$'.format(i), fontsize=fontsize, ha='right', va='center')

plt.text(ytm, xtm2, '$0$', fontsize=fontsize, ha='right', va='baseline')

P_ss_0 = var_u / ((1 - a) ** 2)
plt.text(ytm, P_ss_0, '$\dfrac{\sigma_u^2}{(1-a)^2}$', fontsize=fontsize, ha='right', va='center')
plt.plot([0, ytl], [P_ss_0, P_ss_0], 'k')

plt.axis('off')

# save as pdf image
plt.savefig('problem_13_3_ACF_PSD.pdf', bbox_inches='tight')


plt.show()


