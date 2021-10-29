import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

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

# normal pdf variances
sigma = 1
mu1 = 1.5
mu2 = -1.5

#####################
# END OF PARAMETERS #
#####################

# abscissa values
xmin = -3
xmax = 5

x1 = np.linspace(xmin, 0, 150)
x2 = np.linspace(0, xmax, 300)
# normal distribution and density values in x
pdf11 = norm.pdf(x1, mu1, math.sqrt(sigma))
pdf12 = norm.pdf(x2, mu1, math.sqrt(sigma))
pdf21 = norm.pdf(x1, mu2, math.sqrt(sigma))
pdf22 = norm.pdf(x2, mu2, math.sqrt(sigma))

# axis parameters
dx = 0.5
xmin_ax = xmin - dx
xmax_ax = xmax + dx

pdf_max = norm.pdf(mu1, mu1, math.sqrt(sigma))
ymax_ax = pdf_max + pdf_max / 6
ymin_ax = -pdf_max / 6

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.06
ytm = 0.3
# font size
fontsize = 14
# colors from coolwarm
grey = [0.7, 0.7, 0.7]

fig = plt.figure(0, figsize=(10, 3), frameon=False)

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

plt.plot(x1, pdf11, color=grey, linewidth=2)
plt.plot(x2, pdf12, color='k', linewidth=2)

plt.plot([mu1, mu1], [0, pdf_max], 'k--', lw=1)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$A$', fontsize=fontsize, ha='center', va='baseline')
plt.text(mu1, -0.07, '$\\bar{x}-\dfrac{\sigma^2}{N}\lambda$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.1, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, '$p(\mathbf{x}|A)p(A)$', fontsize=fontsize, ha='left', va='center')

plt.text(xmin_ax, ymax_ax, '$\\bar{x}>\dfrac{\sigma^2}{N}\lambda$', fontsize=fontsize, ha='left', va='baseline')

plt.axis('off')


##
ax = plt.subplot2grid((1, 8), (0, 4), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x1, pdf21, color=grey, linewidth=2)
plt.plot(x2, pdf22, color='k', linewidth=2)

plt.plot([mu2, mu2], [0, pdf_max], 'k--', lw=1)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$A$', fontsize=fontsize, ha='center', va='baseline')
plt.text(mu2, -0.07, '$\\bar{x}-\dfrac{\sigma^2}{N}\lambda$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-0.1, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, '$p(\mathbf{x}|A)p(A)$', fontsize=fontsize, ha='left', va='center')

plt.text(xmin_ax, ymax_ax, '$\\bar{x}<\dfrac{\sigma^2}{N}\lambda$', fontsize=fontsize, ha='left', va='baseline')

plt.axis('off')

# save as pdf image
plt.savefig('problem_11_4.pdf', bbox_inches='tight')

plt.show()

