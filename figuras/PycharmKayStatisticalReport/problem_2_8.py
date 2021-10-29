import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

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

# normal pdf variances
sigma_sqr = 5
N1 = 2
N2 = 40
# normal pdf mean
A = 6
epsilon = 1.7

#####################
# END OF PARAMETERS #
#####################

A_mean = A / 2
var1 = sigma_sqr / N1
var2 = sigma_sqr / N2

# abscissa values
xmin = -0.5
xmax = A + epsilon + 1

x = np.linspace(xmin, xmax, 300)
# normal distribution and density values in x
pdf_var1 = norm.pdf(x, A_mean, math.sqrt(var1))
pdf_var2 = norm.pdf(x, A_mean, math.sqrt(var2))

# axis parameters
dx = xmax / 20
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ym = np.amax(pdf_var2)
ymax_ax = ym + ym / 8
ymin_ax = -ym / 8

# areas to fill limits
pdf_xinf = np.linspace(xmin, A - epsilon, 50)
pdf_xsup = np.linspace(A + epsilon, xmax, 50)
pdf1_inf = norm.pdf(pdf_xinf, A_mean, math.sqrt(var1))
pdf1_sup = norm.pdf(pdf_xsup, A_mean, math.sqrt(var1))
pdf2_inf = norm.pdf(pdf_xinf, A_mean, math.sqrt(var2))
pdf2_sup = norm.pdf(pdf_xsup, A_mean, math.sqrt(var2))


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.16
ytm = 0.4
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

plt.plot(x, pdf_var1, color='k', linewidth=2)

# filled areas
ax.fill_between(pdf_xinf, 0, pdf1_inf, color=col10)
ax.fill_between(pdf_xsup, 0, pdf1_sup, color=col10)

# xlabels and xtickslabels
plt.plot([A, A], [0, xtl], 'k')
plt.plot([A - epsilon, A - epsilon], [0, xtl], 'k')
plt.plot([A + epsilon, A + epsilon], [0, xtl], 'k')
plt.plot([A_mean, A_mean], [0, xtl], 'k')
plt.text(A, xtm, '$A$', fontsize=fontsize, ha='center', va='baseline')
plt.text(A - epsilon, xtm, '$A-\epsilon$', fontsize=fontsize, ha='center', va='baseline')
plt.text(A + epsilon, xtm, '$A+\epsilon$', fontsize=fontsize, ha='center', va='baseline')
plt.text(A_mean, xtm, '$A/2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, '$\check{A}$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, '$p(\check{A})=\mathcal{N}(A/2,\,\sigma^2/4N_1)$',
         fontsize=fontsize, ha='left', va='center')

plt.text(xmax_ax, ymax_ax, '$N_1<N_2$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

# legend
handles = []
labels = [r'$\Pr\big\{|\check{A}-A|>\epsilon\big\}$']
handles.append(Rectangle((0, 0), 1, 1, color=col10, linewidth=0))
leg = plt.legend(handles=handles,
                 labels=labels,
                 framealpha=1,
                 loc='center left',
                 bbox_to_anchor=(0.3, 0.7),
                 fontsize=12,
                 frameon=False)


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

plt.plot(x, pdf_var2, color='k', linewidth=2)

# filled areas
ax.fill_between(pdf_xinf, 0, pdf2_inf, color=col10)
ax.fill_between(pdf_xsup, 0, pdf2_sup, color=col10)

# xlabels and xtickslabels
plt.plot([A, A], [0, xtl], 'k')
plt.plot([A - epsilon, A - epsilon], [0, xtl], 'k')
plt.plot([A + epsilon, A + epsilon], [0, xtl], 'k')
plt.plot([A_mean, A_mean], [0, xtl], 'k')
plt.text(A, xtm, '$A$', fontsize=fontsize, ha='center', va='baseline')
plt.text(A - epsilon, xtm, '$A-\epsilon$', fontsize=fontsize, ha='center', va='baseline')
plt.text(A + epsilon, xtm, '$A+\epsilon$', fontsize=fontsize, ha='center', va='baseline')
plt.text(A_mean, xtm, '$A/2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, '$\check{A}$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, '$p(\check{A})=\mathcal{N}(A/2,\,\sigma^2/4N_2)$',
         fontsize=fontsize, ha='left', va='center')


plt.axis('off')

# save as pdf image
plt.savefig('problem_2_8.pdf', bbox_inches='tight')

plt.show()

