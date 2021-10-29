import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
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

# normal pdf standard deviation
sigma1 = 1
sigma2 = sigma1 / 10
# normal pdf mean
h1 = 3
h2 = h1 / 2

# maximum deviation from the mean where to plot each gaussian
max_mean_dev = 3 * sigma1

#####################
# END OF PARAMETERS #
#####################

# abscissa values
xmin = h2 - max_mean_dev
xmax = h1 + max_mean_dev

x = np.linspace(xmin, xmax, 300)
# normal distribution and density values in x
pdf_h1 = norm.pdf(x, h1, sigma1)
pdf_h1_avg = norm.pdf(x, h1, math.sqrt(sigma2))
pdf_h2 = norm.pdf(x, h2, sigma1)
pdf_h2_avg = norm.pdf(x, h2, math.sqrt(sigma2))

# axis parameters
dx = xmax / 20
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ym = np.amax(pdf_h1_avg)
ymax_ax = ym + ym / 10
ymin_ax = -ym / 10

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.03
# font size
fontsize = 14
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 3), frameon=False)

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

plt.plot(x, pdf_h1, color=col10, linewidth=2)
plt.plot(x, pdf_h1_avg, color=col20, linewidth=2)

# xlabels and xtickslabels
plt.plot([h1, h1], [0, xtl], 'k')
plt.text(h1, xtm, '$h$', fontsize=fontsize, ha='center', va='top')
plt.text(xmin_ax, ymax_ax-0.1, '$\\alpha=1$', fontsize=fontsize, ha='left', va='baseline')


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

plt.plot(x, pdf_h2, color=col10, linewidth=2)
plt.plot(x, pdf_h2_avg, color=col20, linewidth=2)

# xlabels and xtickslabels
plt.plot([h1, h1], [0, xtl], 'k')
plt.text(h1, xtm, '$h$', fontsize=fontsize, ha='center', va='top')
plt.plot([h2, h2], [0, xtl], 'k')
plt.text(h2, xtm, '$\\dfrac{h}{2}$', fontsize=fontsize, ha='center', va='top')


plt.text(xmin_ax, ymax_ax-0.1, '$\\alpha=\\dfrac{1}{2}$', fontsize=fontsize, ha='left', va='baseline')


# legend
leg = plt.legend(['$p(\hat{h}_i)$', '$p(\hat{h})$'], loc=1, fontsize=fontsize)
leg.get_frame().set_facecolor(0.97*np.ones((3,)))
leg.get_frame().set_edgecolor(0.97*np.ones((3,)))

plt.axis('off')

# save as pdf image
plt.savefig('problem_2_4.pdf', bbox_inches='tight')

plt.show()

