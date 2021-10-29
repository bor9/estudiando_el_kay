import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rc
import matplotlib.colors as colors
from matplotlib import cm
from scipy import signal

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

a1 = -0.9
var_u = 1
# number of samples
N = 1000

#####################
# END OF PARAMETERS #
#####################

fmin = 0
fmax = 0.5
nf = 512

w, Pxx = signal.freqz(1, a=[1, a1], worN=nf, whole=False, plot=None)
Pxx = var_u * np.square(np.abs(Pxx))

f = np.linspace(0, fmax, nf, endpoint=False)

rxx0 = var_u / (1 - a1 ** 2)  # autocorrelacion en 0
crlb_Pxx = (4 * np.power(Pxx, 4) * np.square(a1 + np.cos(2 * math.pi * f))) / (N * var_u * rxx0) \
           + 2 * np.square(Pxx) / N

Pxx_dB = 10 * np.log10(Pxx)
crlb_P_dB = 10 * np.log10(crlb_Pxx)

# Simulación
M = 10000

Pxx_est_acumm = np.zeros(nf)
Pxx_est_sq_acumm = np.zeros(nf)
for i in np.arange(M):
    u = np.random.normal(loc=0, scale=math.sqrt(var_u), size=N)
    x = signal.lfilter([1], [1, a1], u)
    # autocorrelacion muestral
    rxx0_est = np.mean(x * x)
    rxx1_est = np.mean(x[1:] * x[:-1])
    # MLE de los parámetros del proceso
    a1_est = -rxx1_est / rxx0_est
    var_u_est = (rxx0_est ** 2 - rxx1_est ** 2) / rxx0_est
    # MLE de la PSD
    w, Pxx_est = signal.freqz(1, a=[1, a1_est], worN=nf, whole=False, plot=None)
    Pxx_est = var_u_est * np.square(np.abs(Pxx_est))
    # Pxx_est_dB = 10 * np.log10(Pxx_est)
    Pxx_est_sq_acumm += np.square(Pxx_est)
    Pxx_est_acumm += Pxx_est

Pxx_est_mean = Pxx_est_acumm / M
Pxx_est_var = Pxx_est_sq_acumm / M - np.square(Pxx_est_mean)

Pxx_est_var_dB = 10 * np.log10(Pxx_est_var)
Pxx_est_mean_dB = 10 * np.log10(Pxx_est_mean)

# axis parameters
dx = 0.05
fmin_ax = fmin - dx
fmax_ax = fmax + dx

ymax_ax = 50
ymin_ax = -60


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -6
ytm = 0.01
# font size
fontsize = 14
fontsize_t = 12

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 5), frameon=False)

# PLOT OF F(x | x < a)
ax = fig.add_subplot(111)

plt.xlim(fmin_ax, fmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(fmin_ax, 0), xycoords='data', xy=(fmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, Pxx_dB, color='k', linewidth=2)
plt.plot(f, crlb_P_dB, color=col10, linewidth=2)
plt.plot(f, Pxx_est_var_dB, color='r', linewidth=2)
plt.plot(f, Pxx_est_mean_dB, color='g', linewidth=2)

# xlabels, xticks and xtickslabels
plt.text(fmax_ax, xtm, '$f$', fontsize=fontsize, ha='right', va='baseline')

xticks = np.arange(fmin, fmax+0.1, 0.1)
for itick in xticks:
    plt.plot([itick, itick], [0, xtl], 'k')
    if math.isclose(math.fabs(itick), 0.1, rel_tol=1e-5):
        plt.text(itick, xtm, '${:.1f}$'.format(itick), fontsize=fontsize_t, ha='center', va='baseline')
    elif math.fabs(itick) < 0.01:
        pass
    else:
        plt.text(itick, -xtm+2, '${:.1f}$'.format(itick), fontsize=fontsize_t, ha='center', va='top')

# ylabels, yticks and ytickslabels
yticks = np.arange(ymin_ax+10, ymax_ax, 10)
for itick in yticks:
    plt.plot([0, ytl], [itick, itick], 'k')
    if itick < -15:
        plt.text(-ytm, itick, '${:d}$'.format(itick), fontsize=fontsize_t, ha='right', va='center')

plt.text(-ytm, 40, '$40$', fontsize=fontsize_t, ha='right', va='center')

plt.text(ytm, ymax_ax, '$\mathrm{Magnitude\,(dB)}$',
         fontsize=fontsize, ha='left', va='center')

leg = plt.legend(['$P_{xx}(f)$', '$\mathrm{CRLB}(\hat{P}_{xx}(f))$'], loc="upper left", fontsize=fontsize)
leg.get_frame().set_facecolor(0.97*np.ones((3,)))
leg.get_frame().set_edgecolor(0.97*np.ones((3,)))

plt.axis('off')

# save as pdf image
# plt.savefig('problem_3_20.pdf', bbox_inches='tight')

plt.show()

