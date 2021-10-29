import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import math
from scipy.stats import expon

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

# exponential PDF
theta = 2
lambd = 0.4
N = 20

#####################
# END OF PARAMETERS #
#####################

# se sortean N muestras de la PDF exponencial con parámetro lambda=theta
np.random.seed(seed=43)
xn = expon.rvs(loc=0, scale=1/theta, size=N)

# abscissa values
tmin = 0
tmax = 7

thetas = np.linspace(tmin, tmax, 300)
priori_pdf = lambd * np.exp(-lambd * thetas)
post_pdf = np.power(thetas, N) * np.exp(-thetas * np.sum(xn))
# post_pdf = np.power(thetas, N) * np.exp(-thetas * 0.37 * N)

# media muestral
sm = np.mean(xn)
post_pdf_sm = (1 / (sm ** N)) * math.exp(-N)
# post_pdf = np.power(1/sm, N) * np.exp(-N)

# axis parameters
dx = 0.4
xmin_ax = tmin - dx
xmax_ax = tmax + dx

dy = 0.1
ymax_ax = np.amax(post_pdf) * (1 + dy)
ymin_ax = -np.amax(post_pdf) * dy

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.15
ytm = -0.2
# font size
fontsize = 14
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)


fig = plt.figure(0, figsize=(5, 3), frameon=False)
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

plt.plot(thetas, priori_pdf, color=col10, lw=2, label='$p(\\theta)$')
plt.plot(thetas, post_pdf, color=col20, lw=2, label='$p(\mathbf{x|}\\theta)$')

plt.plot([1/sm, 1/sm], [0, post_pdf_sm], 'k--', lw=1)

plt.plot([0, ytl], [lambd, lambd], 'k', lw=1)

# labels
plt.text(xmax_ax, xtm, '$\\theta$', fontsize=fontsize, ha='right', va='baseline')
plt.text(1/sm, xtm, '$1/\\bar{x}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, lambd, '$\lambda$', fontsize=fontsize, ha='right', va='center')


leg = plt.legend(loc=1, frameon=False, fontsize=fontsize)

plt.axis('off')

# save as pdf image
plt.savefig('example_11_2.pdf', bbox_inches='tight')

plt.show()

