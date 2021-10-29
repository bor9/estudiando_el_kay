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

# exponential PDF
sigma = 1
sigma_A = 1
sigma_B = 1
M = 20

#####################
# END OF PARAMETERS #
#####################

n = np.arange(-M, M+1)
bsme = 1 / (1 / sigma_A + (2 * M + 1) / sigma) + np.square(n) / (1 / sigma_B + np.sum(np.square(n)) / sigma)


# abscissa values
xmin = -M
xmax = M

# axis parameters
xmin_ax = xmin - xmax / 4
xmax_ax = xmax + xmax / 4

ym = np.amax(bsme)
ymax_ax = ym + ym / 5
ymin_ax = -ym / 20

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.012
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

plt.plot(n, bsme, color=col10, linewidth=2)
plt.plot([M, M], [0, bsme[-1]], 'k--', lw=1)
plt.plot([-M, -M], [0, bsme[-1]], 'k--', lw=1)

# labels
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(M, xtm, '$M$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-M, xtm, '$-M$', fontsize=fontsize, ha='center', va='baseline')
plt.text(1.4, ymax_ax, '${\\rm Bsme}(\hat{s}[n])$', fontsize=fontsize, ha='left', va='center')


plt.axis('off')

# save as pdf image
plt.savefig('problem_11_17.pdf', bbox_inches='tight')

plt.show()

