import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import uniform
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

# media altura y peso
mu_a = 170
mu_p = 66
delta_a = 20
delta_p = 20
rho = 0.7
N = 200

#####################
# END OF PARAMETERS #
#####################

# distribución normal
# varianza de la altura y el peso
var_a = (delta_a / 3) ** 2
var_p = (delta_p / 3) ** 2
# vector de media y matriz de covarianza
cov_ap = rho * math.sqrt(var_a * var_p)
C = [[var_p, cov_ap], [cov_ap, var_a]]
print(C)
mu = [mu_p, mu_a]

nrv = multivariate_normal(mu, C)
ns = nrv.rvs(N)
nw = ns[:, 0]
na = ns[:, 1]

# distribución uniforme
urv = uniform(loc=mu_p - delta_p, scale=2 * delta_p)
up = urv.rvs(N)
urv = uniform(loc=mu_a - delta_a, scale=2 * delta_a)
ua = urv.rvs(N)

# abscissa values
xmin = mu_a - 2 * delta_a
xmax = mu_a + 2 * delta_a
ymin = mu_p - 2 * delta_p
ymax = mu_p + 2 * delta_p


# axis parameters
dx = 3
xmin_ax = xmin - dx
xmax_ax = xmax + dx

dx = 5
ymin_ax = ymin - dx
ymax_ax = ymax + dx

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = 8
ytm = 2
# font size
fontsize = 12
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 3), frameon=False)
ax = plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, ymin), xycoords='data', xy=(xmax_ax, ymin), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(xmin, ymin_ax), xycoords='data', xy=(xmin, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(na, nw, 'k.', ms=5)

# xlabel
plt.text(xmax_ax, ymin-xtm, '$h\;(\\mathrm{cm})$', fontsize=fontsize, ha='center', va='baseline')
# ylabel
plt.text(xmin+ytm, ymax_ax, '$P\;(\\mathrm{kg})$', fontsize=fontsize, ha='left', va='center')

# xtickslabels
xt = np.arange(140, 210, 10)
plt.plot([xt, xt], [ymin, ymin + xtl], 'k', lw=1)
for x in xt:
    plt.text(x, ymin-xtm, '${}$'.format(x), fontsize=fontsize, ha='center', va='baseline')

# ytickslabels
yt = np.arange(40, 120, 20)
plt.plot([xmin, xmin + ytl], [yt, yt], 'k', lw=1)
for y in yt:
    plt.text(xmin-ytl, y, '${}$'.format(y), fontsize=fontsize, ha='right', va='center')

plt.axis('off')


##
ax = plt.subplot2grid((1, 4), (0, 2), rowspan=1, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, ymin), xycoords='data', xy=(xmax_ax, ymin), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(xmin, ymin_ax), xycoords='data', xy=(xmin, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(ua, up, 'k.', ms=5)

# xlabel
plt.text(xmax_ax, ymin-xtm, '$h\;(\\mathrm{cm})$', fontsize=fontsize, ha='center', va='baseline')
# ylabel
plt.text(xmin+ytm, ymax_ax, '$P\;(\\mathrm{kg})$', fontsize=fontsize, ha='left', va='center')

# xtickslabels
xt = np.arange(140, 210, 10)
plt.plot([xt, xt], [ymin, ymin + xtl], 'k', lw=1)
for x in xt:
    plt.text(x, ymin-xtm, '${}$'.format(x), fontsize=fontsize, ha='center', va='baseline')

# ytickslabels
yt = np.arange(40, 120, 20)
plt.plot([xmin, xmin + ytl], [yt, yt], 'k', lw=1)
for y in yt:
    plt.text(xmin-ytl, y, '${}$'.format(y), fontsize=fontsize, ha='right', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('problem_10_11.pdf', bbox_inches='tight')

plt.show()

