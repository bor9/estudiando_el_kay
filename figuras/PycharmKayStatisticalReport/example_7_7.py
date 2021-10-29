import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace
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

# parametros de la pdf de laplace
mean1 = 0
mean2 = 2
var = 0.5

#####################
# END OF PARAMETERS #
#####################

b = math.sqrt(var/2)

# abscissa values
xmin = -4
xmax = 4

x = np.linspace(xmin, xmax, 300)
# normal distribution and density values in x
pdf_w = laplace.pdf(x, loc=mean1, scale=b)
pdf_x = laplace.pdf(x, loc=mean2, scale=b)

# axis parameters
dx = xmax / 8
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ym = np.amax(pdf_w)
ymax_ax = ym + ym / 3
ymin_ax = -ym / 10

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.25
ytm = 0.3
# font size
fontsize = 14
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 2), frameon=False)

# PLOT OF F(x | x < a)
ax = plt.subplot2grid((1, 8), (0, 0), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, pdf_w, 'k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$w[n]$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, ymax_ax, '$p(w[n])$', fontsize=fontsize, ha='left', va='center')


plt.axis('off')


# PLOT OF F(x | x < a)
ax = plt.subplot2grid((1, 8), (0, 4), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)


# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, pdf_x, 'k', linewidth=2)
plt.plot([mean2, mean2], [0, xtl], 'k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$x[n]$', fontsize=fontsize, ha='center', va='baseline')
plt.text(mean2, xtm, '$A$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, ymax_ax, '$p(x[n];\,A)$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('example_7_7.pdf', bbox_inches='tight')

plt.show()

