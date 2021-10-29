import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import math
from scipy.stats import norm

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

A = 0
var_w = 1
# numero de muestras
Ns = [10, 1000]
# numero de experimentos
M = 10000


#####################
# END OF PARAMETERS #
#####################

estimaciones = np.zeros((M, 2))

np.random.seed(13)
for i in np.arange(len(Ns)):
    for j in np.arange(M):
        x = np.random.normal(loc=A, scale=math.sqrt(var_w), size=(Ns[i], ))
        mean_x = np.mean(x)
        var_x = np.var(x)
        estimaciones[j, i] = mean_x / math.sqrt(var_x / Ns[i])


# abscissa values
dA = 4.2
xmin = A - dA
xmax = A + dA

nbins = 60
bin_edges = np.linspace(xmin, xmax, nbins)
hist1, bins = np.histogram(estimaciones[:, 0], bin_edges, density=False)
hist2, bins = np.histogram(estimaciones[:, 1], bin_edges, density=False)
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

# pdf teorica
xpdf = np.linspace(xmin, xmax, 100)
pdf = norm.pdf(xpdf, 0, math.sqrt(1))

# axis parameters
dx = 0.2
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ymax_ax = np.amax(pdf) + 0.1
ymin_ax = -0.05

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.07
ytm = 0.1
# font size
fontsize = 12


fig = plt.figure(0, figsize=(9, 5), frameon=False)

# PLOT OF P_ww
ax = fig.add_subplot(211)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(xmin, ymin_ax), xycoords='data', xy=(xmin, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.bar(center, hist1 / (M * width), align='center', width=0.9 * width, ec='k', fc='b')
plt.plot(xpdf, pdf , 'r-', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm-0.02, '$\\frac{\sqrt{N}\\bar{x}}{\hat{\sigma}}$', fontsize=14, ha='center', va='baseline')

xts = np.arange(-4, 5, 1)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:d}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(0.1, 0.5, 0.1)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [yt, yt], 'k')
    plt.text(xmin-ytm, yt, '${:.1f}$'.format(yt), fontsize=fontsize, ha='right', va='center')

# ylabel
plt.text(xmin+2*ytm, ymax_ax-0.02, '$p\left(\\frac{\sqrt{N}\\bar{x}}{\hat{\sigma}}\\right)$', fontsize=14, ha='left',
         va='baseline')

# legends
i = 0
plt.text(-2.5, ymax_ax-0.02, '$N={:d}$'.format(Ns[i]), fontsize=fontsize, ha='left', va='baseline')

plt.text(2.2, ymax_ax-0.02, '${\\rm Media: }$', fontsize=fontsize, ha='left', va='baseline')
plt.text(4.1, ymax_ax-0.02, '${0:.5f}$'.format(np.mean(estimaciones[:, i])), fontsize=fontsize, ha='right',
         va='baseline')

plt.text(2.2, ymax_ax-0.08, '${\\rm Varianza: }$', fontsize=fontsize, ha='left', va='baseline')
plt.text(4.1, ymax_ax-0.08, '${0:.5f}$'.format(np.var(estimaciones[:, i])), fontsize=fontsize, ha='right', va='baseline')

plt.axis('off')

ax = fig.add_subplot(212)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(xmin, ymin_ax), xycoords='data', xy=(xmin, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.bar(center, hist2 / (M * width), align='center', width=0.9*width, ec='k', fc='b')
plt.plot(xpdf, pdf, 'r-', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm-0.02, '$\\frac{\sqrt{N}\\bar{x}}{\hat{\sigma}}$', fontsize=14, ha='center', va='baseline')

xts = np.arange(-4, 5, 1)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:d}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(0.1, 0.5, 0.1)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [yt, yt], 'k')
    plt.text(xmin-ytm, yt, '${:.1f}$'.format(yt), fontsize=fontsize, ha='right', va='center')

# ylabel
plt.text(xmin+2*ytm, ymax_ax-0.02, '$p\left(\\frac{\sqrt{N}\\bar{x}}{\hat{\sigma}}\\right)$', fontsize=14, ha='left',
         va='baseline')

# legends
i = 1
plt.text(-2.5, ymax_ax-0.02, '$N={:d}$'.format(Ns[i]), fontsize=fontsize, ha='left', va='baseline')

plt.text(2.2, ymax_ax-0.02, '${\\rm Media: }$', fontsize=fontsize, ha='left', va='baseline')
plt.text(4.1, ymax_ax-0.02, '${0:.5f}$'.format(np.mean(estimaciones[:, i])), fontsize=fontsize, ha='right',
         va='baseline')

plt.text(2.2, ymax_ax-0.08, '${\\rm Varianza: }$', fontsize=fontsize, ha='left', va='baseline')
plt.text(4.1, ymax_ax-0.08, '${0:.5f}$'.format(np.var(estimaciones[:, i])), fontsize=fontsize, ha='right',
         va='baseline')

plt.axis('off')

# save as pdf image
plt.savefig('problem_7_14.pdf', bbox_inches='tight')

plt.show()

