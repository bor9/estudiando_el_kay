import matplotlib.pyplot as plt
import numpy as np
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

# limite superor de la densidad uniforme
beta = 10
# numero de muestras
N = 100
# numero de experimentos
M = 10000

#####################
# END OF PARAMETERS #
#####################

estimaciones1 = np.zeros((M, ))
estimaciones2 = np.zeros((M, ))

np.random.seed(4)
for i in np.arange(M):
    x = np.random.uniform(low=0, high=beta, size=(N, ))
    estimaciones1[i] = (N+1) / (2 * N) * np.amax(x)
    estimaciones2[i] = np.mean(x)

# abscissa values
dbeta = 1
xmin = beta / 2 - dbeta
xmax = beta / 2 + dbeta

bin_edges = np.linspace(xmin, xmax, 60)
hist1, bins = np.histogram(estimaciones1, bin_edges, density=False)
hist2, bins = np.histogram(estimaciones2, bin_edges, density=False)
width = 0.9 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2

# escalas iguales en ambas graficas
fs = np.amax(hist1)/np.amax(hist2)
print(np.amax(hist2))

# axis parameters
dx = 0.1
xmin_ax = xmin - dx
xmax_ax = xmax + dx

dy = 1000
ymax_ax = np.amax(hist1) + dy
ymin_ax = -dy


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -1000
ytm = 0.03
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

plt.bar(center, hist1, align='center', width=width, ec='k', fc='b')

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$\hat{\\theta}_1$', fontsize=fontsize, ha='left', va='baseline')

xts = np.arange(4.25, 6.1, 0.25)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:.2f}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(1000, 5001, 1000)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [yt, yt], 'k')
    plt.text(xmin-ytm, yt, '${:d}$'.format(yt), fontsize=fontsize, ha='right', va='center')

# ylabel
plt.text(xmin+2*ytm, ymax_ax, '${\\rm Número\;de}$\n${\\rm ocurrencias}$', fontsize=fontsize, ha='left', va='center',
         ma='center')

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

plt.bar(center, fs * hist2, align='center', width=width, ec='k', fc='b')

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$\hat{\\theta}_2$', fontsize=fontsize, ha='left', va='baseline')

xts = np.arange(4.25, 6.1, 0.25)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:.2f}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(100, 501, 100)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [fs * yt, fs * yt], 'k')
    plt.text(xmin-ytm, fs * yt, '${:d}$'.format(yt), fontsize=fontsize, ha='right', va='center')

# ylabel
plt.text(xmin+2*ytm, ymax_ax, '${\\rm Número\;de}$\n${\\rm ocurrencias}$', fontsize=fontsize, ha='left', va='center',
         ma='center')

plt.axis('off')

# save as pdf image
plt.savefig('example_5_8.pdf', bbox_inches='tight')

plt.show()

