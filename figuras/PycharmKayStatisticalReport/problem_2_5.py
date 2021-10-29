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
sigma_sqr = 1
lambd = 1/sigma_sqr

#####################
# END OF PARAMETERS #
#####################

# abscissa values
xmin = 0
xmax = 5 * sigma_sqr

x = np.linspace(xmin, xmax, 300)
# exponential density
exp_pdf = lambd * np.exp(-lambd * x)

# axis parameters
xmin_ax = xmin - xmax / 8
xmax_ax = xmax + xmax / 15

ym = exp_pdf[0]
ymax_ax = ym + ym / 5
ymin_ax = -ym / 5

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.18
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

plt.plot(x, exp_pdf, color=col10, linewidth=2)

# labels
plt.plot([sigma_sqr, sigma_sqr], [0, lambd * math.exp(-1)], 'k--')
plt.text(xmax_ax, xtm, '$\hat{\sigma^2}$', fontsize=fontsize, ha='right', va='baseline')
plt.text(sigma_sqr, xtm, '$\sigma^2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0.2, ymax_ax, '$p(\hat{\sigma^2})$', fontsize=fontsize, ha='left', va='center')
plt.plot([0, ytl], [1/sigma_sqr, 1/sigma_sqr], 'k')
plt.text(-0.15, 1/sigma_sqr, '$\\dfrac{1}{\sigma^2}$', fontsize=fontsize, ha='right', va='center')


plt.axis('off')

# save as pdf image
plt.savefig('problem_2_5.pdf', bbox_inches='tight')

plt.show()

