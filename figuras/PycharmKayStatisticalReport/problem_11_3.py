import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
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
x = 1

#####################
# END OF PARAMETERS #
#####################

# abscissa values
xmin = -0.5
xmax = 5

theta = np.linspace(x, xmax, 100)
# exponential density
exp_pdf = np.exp(-(theta - x))

# axis parameters
dx = 0.4
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ymax_ax = 1.2
ymin_ax = -0.15

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.15
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


plt.plot([xmin, x], [0, 0], color=col10, linewidth=2, zorder=10)
plt.plot(theta, exp_pdf, color=col10, linewidth=2, zorder=10)

plt.plot([0, x], [1, 1], 'k--', lw=1)
plt.plot([x, x], [0, 1], 'k--', lw=1)
plt.plot([x+1, x+1], [0, np.exp(-x)], 'k--', lw=1)


# labels
plt.text(xmax_ax, xtm, '$\\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0.2, ymax_ax, '$p(\\theta|x)$', fontsize=fontsize, ha='left', va='center')

# ticks labels
plt.text(x, xtm, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(x+1, xtm, '$x+1$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.1, 1, '$1$', fontsize=fontsize, ha='right', va='center')

plt.text(x, 2.1 * xtm, '$\hat{\\theta}_{\\rm MMSE}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(x+1, 2.1 * xtm, '$\hat{\\theta}_{\\rm MAP}$', fontsize=fontsize, ha='center', va='baseline')


plt.axis('off')

# save as pdf image
plt.savefig('problem_11_3.pdf', bbox_inches='tight')

plt.show()

