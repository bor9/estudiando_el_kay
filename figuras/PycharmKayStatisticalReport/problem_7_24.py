import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
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

f01 = 0.25
f02 = 0.05
N = 10
nf = 1024

#####################
# END OF PARAMETERS #
#####################

n = np.arange(N)
x1 = np.cos(2 * math.pi * f01 * n)
x2 = np.cos(2 * math.pi * f02 * n)

# MLE aproximado
w, X1 = signal.freqz(x1, a=1, worN=nf, whole=False, plot=None)
w, X2 = signal.freqz(x2, a=1, worN=nf, whole=False, plot=None)

J1_app = np.square(np.absolute(X1)) * 2 / N
J2_app = np.square(np.absolute(X2)) * 2 / N

f = np.arange(0, nf) * 0.5 / nf

# MLE exacto
J1 = np.zeros((nf, ))
J2 = np.zeros((nf, ))

for i in np.arange(1, nf):
    # se comienza en f[1] porque f[0] = 0 produce una matriz no invertible
    H = np.vstack((np.cos(2 * math.pi * f[i] * n), np.sin(2 * math.pi * f[i] * n))).T
    A = H @ np.linalg.inv(H.T @ H) @ H.T
    J1[i] = np.dot(x1, np.dot(A, x1))
    J2[i] = np.dot(x2, np.dot(A, x2))

# estimadores
f01_app_est = f[np.argmax(J1_app)]
f01_est = f[np.argmax(J1)]
f02_app_est = f[np.argmax(J2_app)]
f02_est = f[np.argmax(J2)]
print(f01_app_est)
print(f01_est)
print(f02_app_est)
print(f02_est)


# abscissa values
xmin = 0
xmax = 0.5
ymin = 0
ymax = 6
# axis parameters
xmin_ax = xmin
xmax_ax = xmax + 0.03
# para la grafica de g(x)
ymax_ax = ymax + 1
ymin_ax = ymin - 1


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.8
ytm = 0.007
# font size
fontsize = 12


# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)


fig = plt.figure(0, figsize=(9, 6), frameon=False)

# grafica de g(x)
ax = plt.subplot2grid((8, 1), (0, 0), rowspan=4, colspan=1)
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f[1:], J1[1:], color=col10, lw=2, label='$J(f)\;\mathrm{{exacta}}\;(\hat{{f_0}}={:.2f})$'.format(f01_est))
plt.plot(f, J1_app, color=col20, lw=2, label='$\mathrm{{Periodograma}}\;(\hat{{f_0}}={:.2f})$'.format(f01_app_est))
plt.plot(f[np.argmax(J1)]*np.ones((2,)), [0, np.amax(J1)], color=col10, ls='--')
plt.plot(f[np.argmax(J1_app)]*np.ones((2,)), [0, np.amax(J1_app)], color=col20, ls='--')

plt.text(ytm, ymax_ax, '$J(f)\;\mathrm{{con}}\;f_0={:.2f}$'.format(f01), fontsize=fontsize, ha='left', va='center')
plt.text(xmax_ax, xtm, '$f$', fontsize=fontsize, ha='center', va='baseline')

xts = np.arange(0.1, 0.55, 0.1)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:.1f}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')
plt.text(0, xtm, '$0$', fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(0, 7, 1)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [yt, yt], 'k')
    plt.text(xmin-ytm, yt, '${:d}$'.format(yt), fontsize=fontsize, ha='right', va='center')

leg = plt.legend(loc=1, fontsize=fontsize, frameon=False)

plt.axis('off')

ax = plt.subplot2grid((8, 1), (4, 0), rowspan=4, colspan=1)
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f[1:], J2[1:], color=col10, lw=2, label='$J(f)\;\mathrm{{exacta}}\;(\hat{{f_0}}={:.4f})$'.format(f02_est))
plt.plot(f, J2_app, color=col20, lw=2, label='$\mathrm{{Periodograma}}\;(\hat{{f_0}}={:.4f})$'.format(f02_app_est))
plt.plot(f[np.argmax(J2)]*np.ones((2,)), [0, np.amax(J2)], color=col10, ls='--')
plt.plot(f[np.argmax(J2_app)]*np.ones((2,)), [0, np.amax(J2_app)], color=col20, ls='--')


plt.text(ytm, ymax_ax, '$J(f)\;\mathrm{{con}}\;f_0={:.2f}$'.format(f02), fontsize=fontsize, ha='left', va='center')
plt.text(xmax_ax, xtm, '$f$', fontsize=fontsize, ha='center', va='baseline')

xts = np.arange(0.1, 0.55, 0.1)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:.1f}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')
plt.text(0, xtm, '$0$', fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(0, 7, 1)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [yt, yt], 'k')
    plt.text(xmin-ytm, yt, '${:d}$'.format(yt), fontsize=fontsize, ha='right', va='center')

leg = plt.legend(loc=1, fontsize=fontsize, frameon=False)

plt.axis('off')


# save as pdf image
plt.savefig('problem_7_24.pdf', bbox_inches='tight')
plt.show()
