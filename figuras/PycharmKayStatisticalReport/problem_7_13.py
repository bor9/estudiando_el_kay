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

A = 1
var_w = 0.1
# numero de muestras
N = 50
# numero de experimentos
M1 = 1000
M2 = 10000


#####################
# END OF PARAMETERS #
#####################

estimaciones1 = np.zeros((M1, ))
estimaciones2 = np.zeros((M2, ))

np.random.seed(3)
for i in np.arange(M1):
    x = np.random.normal(loc=A, scale=math.sqrt(var_w), size=(N, ))
    mean = np.mean(x)
    estimaciones1[i] = mean
    estimaciones2[i] = mean
for i in np.arange(M1, M2):
    x = np.random.normal(loc=A, scale=math.sqrt(var_w), size=(N,))
    estimaciones2[i] = np.mean(x)

# media y varianza
mean1 = np.mean(estimaciones1)
var1 = np.var(estimaciones1)
mean2 = np.mean(estimaciones2)
var2 = np.var(estimaciones2)


# abscissa values
dA = 0.16
xmin = A - dA
xmax = A + dA

nbins = 60
bin_edges = np.linspace(xmin, xmax, nbins)
hist1, bins = np.histogram(estimaciones1, bin_edges, density=False)
hist2, bins = np.histogram(estimaciones2, bin_edges, density=False)
width = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

# pdf teorica
xpdf = np.linspace(xmin, xmax, 100)
pdf = norm.pdf(xpdf, A, math.sqrt(var_w/N))

# axis parameters
dx = 0.01
xmin_ax = xmin - dx
xmax_ax = xmax + dx


ymax_ax = np.amax(pdf) + 3
ymin_ax = -1.5

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -1.8
ytm = 0.003
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

plt.bar(center, hist1 / (M1 * width), align='center', width=0.9 * width, ec='k', fc='b')
plt.plot(xpdf, pdf , 'r-', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$\\bar{x}$', fontsize=fontsize, ha='center', va='baseline')

xts = np.arange(0.85, 1.2, 0.05)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:.2f}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(2.5, 11, 2.5)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [yt, yt], 'k')
    plt.text(xmin-ytm, yt, '${:.1f}$'.format(yt), fontsize=fontsize, ha='right', va='center')

# ylabel
plt.text(xmin+2*ytm, ymax_ax, '$p(\\bar{x};\,A)$', fontsize=fontsize, ha='left', va='baseline')

# legends
plt.text(0.9, ymax_ax, '$M={:d}$'.format(M1), fontsize=fontsize, ha='left', va='baseline')

plt.text(1.1, ymax_ax, '${\\rm Media: }$', fontsize=fontsize, ha='left', va='baseline')
plt.text(1.14, ymax_ax, '${0:.5f}$'.format(mean1), fontsize=fontsize, ha='left', va='baseline')

plt.text(1.1, ymax_ax-1.5, '${\\rm Varianza: }$', fontsize=fontsize, ha='left', va='baseline')
plt.text(1.14, ymax_ax-1.5, '${0:.5f}$'.format(var1), fontsize=fontsize, ha='left', va='baseline')

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

plt.bar(center, hist2 / (M2 * width), align='center', width=0.9*width, ec='k', fc='b')
plt.plot(xpdf, pdf, 'r-', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$\\bar{x}$', fontsize=fontsize, ha='center', va='baseline')

xts = np.arange(0.85, 1.2, 0.05)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:.2f}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(2.5, 11, 2.5)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [yt, yt], 'k')
    plt.text(xmin-ytm, yt, '${:.1f}$'.format(yt), fontsize=fontsize, ha='right', va='center')

# ylabel
plt.text(xmin+2*ytm, ymax_ax, '$p(\\bar{x};\,A)$', fontsize=fontsize, ha='left', va='baseline')

# legends
plt.text(0.9, ymax_ax, '$M={:d}$'.format(M2), fontsize=fontsize, ha='left', va='baseline')

plt.text(1.1, ymax_ax, '${\\rm Media: }$', fontsize=fontsize, ha='left', va='baseline')
plt.text(1.14, ymax_ax, '${0:.5f}$'.format(mean2), fontsize=fontsize, ha='left', va='baseline')

plt.text(1.1, ymax_ax-1.5, '${\\rm Varianza: }$', fontsize=fontsize, ha='left', va='baseline')
plt.text(1.14, ymax_ax-1.5, '${0:.5f}$'.format(var2), fontsize=fontsize, ha='left', va='baseline')

plt.axis('off')

# save as pdf image
plt.savefig('problem_7_13.pdf', bbox_inches='tight')

plt.show()

