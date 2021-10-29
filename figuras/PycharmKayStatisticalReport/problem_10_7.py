import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
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

# varianza (sqrt{sigma^2/N})
sigma_sqr = 1
# normal pdf mean
A01 = 3
A02 = 10

#####################
# END OF PARAMETERS #
#####################

N = 500

xmin1 = -A01
xmax1 = A01
x1 = np.linspace(xmin1, xmax1, N)
hat_A1 = x1 + (norm.pdf(-A01 - x1, 0, 1) - norm.pdf(A01 - x1, 0, 1)) / \
         (norm.cdf(A01 - x1, 0, 1) - norm.cdf(-A01 - x1, 0, 1))

xmin2 = -A02
xmax2 = A02
x2 = np.linspace(xmin2, xmax2, N)
hat_A2 = x2 + (norm.pdf(-A02 - x2, 0, 1) - norm.pdf(A02 - x2, 0, 1)) / \
         (norm.cdf(A02 - x2, 0, 1) - norm.cdf(-A02 - x2, 0, 1))

# axis parameters
dx = xmax1 / 6
xmin_ax = xmin1 - dx
xmax_ax = xmax1 + dx

ymax_ax = xmax_ax
ymin_ax = xmin_ax

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.16
ytm = 0.4
# font size
fontsize1 = 10
fontsize2 = 12
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 4), frameon=False)

# PLOT OF F(x | x < a)
ax = plt.subplot2grid((1, 8), (0, 0), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)
# horizontal and vertical margins
xtm, _ = convert_display_to_data_coordinates(ax.transData, length=18)
# horizontal and vertical margins
_, ytm = convert_display_to_data_coordinates(ax.transData, length=6)


# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x1, x1, color='k', linewidth=2, label='\\bar{x}')
plt.plot(x1, hat_A1, color='r', linewidth=2, label='\hat{A}')

# xlabels and xtickslabels
ticks = np.array([1, 2, 3])
plt.plot([ticks, ticks], [0, xtl], 'k')
plt.plot([-ticks, -ticks], [0, xtl], 'k')

plt.plot([0, ytl], [ticks, ticks], 'k')
plt.plot([0, ytl], [-ticks, -ticks], 'k')

for t in ticks:
    plt.text(t, -xtm, '${}$'.format(t), fontsize=fontsize1, ha='center', va='baseline')
    plt.text(-t, -xtm, '${}$'.format(-t), fontsize=fontsize1, ha='center', va='baseline')
    plt.text(-ytm, t, '${}$'.format(t), fontsize=fontsize1, ha='right', va='center')
    plt.text(-ytm, -t, '${}$'.format(-t), fontsize=fontsize1, ha='right', va='center')
plt.text(ytm / 2, -xtm, '$0$', fontsize=fontsize1, ha='left', va='baseline')

plt.text(xmax_ax, -xtm, '$\\bar{x}$', fontsize=fontsize2, ha='center', va='baseline')
plt.legend(loc='center', fontsize=fontsize2, frameon=False, bbox_to_anchor=(0.8, 0.2))
plt.text(xmin_ax + xtm, 6 * xtm, '$A_0={}$'.format(A01), fontsize=fontsize2, ha='left', va='baseline')

plt.axis('off')

# axis parameters
dx = xmax2 / 6
xmin_ax = xmin2 - dx
xmax_ax = xmax2 + dx

ymax_ax = xmax_ax
ymin_ax = xmin_ax

ax = plt.subplot2grid((1, 8), (0, 4), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)
# horizontal and vertical margins
xtm, _ = convert_display_to_data_coordinates(ax.transData, length=18)
# horizontal and vertical margins
_, ytm = convert_display_to_data_coordinates(ax.transData, length=6)


# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x2, x2, color='k', linewidth=2)
plt.plot(x2, hat_A2, color='r', linewidth=2)

# xlabels and xtickslabels
# ticks = np.array([2, 4, 6, 8, 10])
ticks = np.array([2.5, 5, 7.5, 10])
plt.plot([ticks, ticks], [0, xtl], 'k')
plt.plot([-ticks, -ticks], [0, xtl], 'k')

plt.plot([0, ytl], [ticks, ticks], 'k')
plt.plot([0, ytl], [-ticks, -ticks], 'k')

for t in ticks:
    plt.text(t, -xtm, '${}$'.format(t), fontsize=fontsize1, ha='center', va='baseline')
    plt.text(-t, -xtm, '${}$'.format(-t), fontsize=fontsize1, ha='center', va='baseline')
    plt.text(-ytm, t, '${}$'.format(t), fontsize=fontsize1, ha='right', va='center')
    plt.text(-ytm, -t, '${}$'.format(-t), fontsize=fontsize1, ha='right', va='center')
plt.text(ytm / 2, -xtm, '$0$', fontsize=fontsize1, ha='left', va='baseline')

plt.text(xmax_ax, -xtm, '$\\bar{x}$', fontsize=fontsize2, ha='center', va='baseline')
plt.text(xmin_ax + xtm, 6 * xtm, '$A_0={}$'.format(A02), fontsize=fontsize2, ha='left', va='baseline')

plt.axis('off')

# save as pdf image
plt.savefig('problem_10_7.pdf', bbox_inches='tight')

plt.show()

