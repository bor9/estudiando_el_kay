import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as colors
from scipy.stats import norm
from scipy import optimize

from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')


def j(z):
    return epsilon2 * norm.cdf(z-x) + (1 - epsilon2) * norm.cdf(z+x) - 1/2


def fun_pdf1(theta):
    return epsilon1 / math.sqrt(2 * math.pi) * np.exp(-0.5 * np.square(theta - x)) + \
           (1 - epsilon1) / math.sqrt(2 * math.pi) * np.exp(-0.5 * np.square(theta + x))


def fun_pdf2(theta):
    return epsilon2 / math.sqrt(2 * math.pi) * np.exp(-0.5 * np.square(theta - x)) + \
           (1 - epsilon2) / math.sqrt(2 * math.pi) * np.exp(-0.5 * np.square(theta + x))


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

# parametros de la pdf
epsilon1 = 1/2
epsilon2 = 3/4
x = 2
N = 400

#####################
# END OF PARAMETERS #
#####################


# abscissa values
xmin = -5
xmax = 5

theta = np.linspace(xmin, xmax, N)
# normal distribution and density values in x
pdf1 = epsilon1 / math.sqrt(2 * math.pi) * np.exp(-0.5 * np.square(theta - x)) + \
       (1 - epsilon1) / math.sqrt(2 * math.pi) * np.exp(-0.5 * np.square(theta + x))
pdf2 = epsilon2 / math.sqrt(2 * math.pi) * np.exp(-0.5 * np.square(theta - x)) + \
       (1 - epsilon2) / math.sqrt(2 * math.pi) * np.exp(-0.5 * np.square(theta + x))

# cálculo de la mediana
z0 = 0
pdf2_root = optimize.root(j, z0)
median_pdf2 = pdf2_root.x[0]
print(median_pdf2)


# axis parameters
dx = 0.6
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ym = np.amax(pdf2)
ymax_ax = ym + ym / 3
ymin_ax = -ym / 5

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.06
ytm = 0.3
# font size
fontsize = 13
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col1 = scalarMap.to_rgba(0)
col2 = scalarMap.to_rgba(0.2)
col3 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 2), frameon=False)

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

plt.plot(theta, pdf1, 'k', lw=2)
plt.plot([x, x], [0, xtl], 'k', lw=1)
plt.plot([-x, -x], [0, xtl], 'k', lw=1)

# media, mediana y modo
mean_pdf1 = 0
median_pdf1 = 0
plt.plot([mean_pdf1, mean_pdf1], [0, fun_pdf1(mean_pdf1)], color=col1)
plt.plot([median_pdf1, median_pdf1], [0, fun_pdf1(median_pdf1)], color=col2)
plt.plot([-x, -x], [0, fun_pdf1(-x)], color=col3)
plt.plot([x, x], [0, fun_pdf1(x)], color=col3)


# xlabels and xtickslabels
plt.text(x, xtm, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-x, xtm, '$-x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, '$\\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, ymax_ax, '$p(\\theta|x)$', fontsize=fontsize, ha='left', va='center')

plt.text(xmin_ax, ymax_ax - 0.05, '$\epsilon=\\dfrac{1}{2}$', fontsize=fontsize, ha='left', va='baseline')

plt.axis('off')

#
ax = plt.subplot2grid((1, 8), (0, 4), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)


# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(theta, pdf2, 'k', lw=2)
plt.plot([x, x], [0, xtl], 'k', lw=1)
plt.plot([-x, -x], [0, xtl], 'k', lw=1)

# media, mediana y modo
mean_pdf2 = x / 2
mode_pdf2 = x
plt.plot([mean_pdf2, mean_pdf2], [0, fun_pdf2(mean_pdf2)], color=col1, label='${\\rm media}$')
plt.plot([median_pdf2, median_pdf2], [0, fun_pdf2(median_pdf2)], color=col2, label='${\\rm mediana}$')
plt.plot([mode_pdf2, mode_pdf2], [0, fun_pdf2(mode_pdf2)], color=col3, label='${\\rm moda}$')

plt.plot([mean_pdf2, mean_pdf2], [0, xtl], 'k', lw=1)
plt.text(x/2, 1.35 * xtm, '$\\dfrac{x}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([median_pdf2, median_pdf2], [0, xtl], 'k', lw=1)

# xlabels and xtickslabels
plt.text(x, xtm, '$x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-x, xtm, '$-x$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, '$\\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, ymax_ax, '$p(\\theta|x)$', fontsize=fontsize, ha='left', va='center')

plt.text(xmax_ax, ymax_ax - 0.05, '$\epsilon=\\dfrac{3}{4}$', fontsize=fontsize, ha='right', va='baseline')

plt.axis('off')

leg = plt.legend(loc=(0, 0.5), frameon=False, fontsize=12)

# save as pdf image
plt.savefig('problem_11_2.pdf', bbox_inches='tight')


plt.show()

