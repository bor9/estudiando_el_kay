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
Ns = [40, 400]

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
crlb_Pxx1 = (4 * np.power(Pxx, 4) * np.square(a1 + np.cos(2 * math.pi * f))) / (Ns[0] * var_u * rxx0) \
           + 2 * np.square(Pxx) / Ns[0]
crlb_Pxx2 = (4 * np.power(Pxx, 4) * np.square(a1 + np.cos(2 * math.pi * f))) / (Ns[1] * var_u * rxx0) \
           + 2 * np.square(Pxx) / Ns[1]

Pxx_dB = 10 * np.log10(Pxx)
crlb_Pxx1_dB = 10 * np.log10(crlb_Pxx1)
crlb_Pxx2_dB = 10 * np.log10(crlb_Pxx2)

# Simulación
M = 10000
Pxx_est_acumm = np.zeros((nf, 2))
Pxx_est_sq_acumm = np.zeros((nf, 2))
for j in [0, 1]:
    for i in np.arange(M):
        u = np.random.normal(loc=0, scale=math.sqrt(var_u), size=Ns[j])
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
        Pxx_est_sq_acumm[:, j] += np.square(Pxx_est)
        Pxx_est_acumm[:, j] += Pxx_est

Pxx1_est_mean = Pxx_est_acumm[:, 0] / M
Pxx1_est_var = Pxx_est_sq_acumm[:, 0] / M - np.square(Pxx1_est_mean)

Pxx1_est_var_dB = 10 * np.log10(Pxx1_est_var)
Pxx1_est_mean_dB = 10 * np.log10(Pxx1_est_mean)

Pxx2_est_mean = Pxx_est_acumm[:, 1] / M
Pxx2_est_var = Pxx_est_sq_acumm[:, 1] / M - np.square(Pxx2_est_mean)

Pxx2_est_var_dB = 10 * np.log10(Pxx2_est_var)
Pxx2_est_mean_dB = 10 * np.log10(Pxx2_est_mean)


# axis parameters
dx = 0.05
fmin_ax = fmin
fmax_ax = fmax + dx

ymax_ax = 55
ymin_ax = -37


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -5
ytm = 0.01
# font size
fontsize = 12

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 5), frameon=False)

# Grafica con Ns[0]
ax = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)

plt.xlim(fmin_ax, fmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(fmin_ax, 0), xycoords='data', xy=(fmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, Pxx_dB, color='k', linewidth=1, )
plt.plot(f, Pxx1_est_mean_dB, 'r', linewidth=1)
plt.plot(f, Pxx1_est_var_dB, color=col10, linewidth=1)
plt.plot(f, crlb_Pxx1_dB, color=col20, linewidth=1)

# xlabels, xticks and xtickslabels
plt.text(fmax_ax, xtm, '$f$', fontsize=fontsize, ha='right', va='baseline')

xticks = np.arange(0.2, fmax+0.1, 0.1)
for itick in xticks:
    plt.plot([itick, itick], [0, xtl], 'k')
    plt.text(itick, -xtm+2, '${:.1f}$'.format(itick), fontsize=fontsize, ha='center', va='top')
plt.plot([0.1, 0.1], [0, xtl], 'k')

# ylabels, yticks and ytickslabels
yticks = np.arange(-30, 55, 10)
for itick in yticks:
    plt.plot([0, ytl], [itick, itick], 'k')
    plt.text(-ytm, itick, '${:d}$'.format(itick), fontsize=fontsize, ha='right', va='center')

plt.text(0.02, ymax_ax, '$\mathrm{Magnitude\,(dB)}$', fontsize=fontsize, ha='left', va='center')

plt.text((fmax_ax + fmin_ax) / 2, ymin_ax, '$N={:d}$'.format(Ns[0]), fontsize=fontsize, ha='center', va='baseline')

plt.axis('off')

# Grafica con Ns[1]
ax = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)

plt.xlim(fmin_ax, fmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(fmin_ax, 0), xycoords='data', xy=(fmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, Pxx_dB, color='k', linewidth=1, )
plt.plot(f, Pxx2_est_mean_dB, 'r', linewidth=1)
plt.plot(f, Pxx2_est_var_dB, color=col10, linewidth=1)
plt.plot(f, crlb_Pxx2_dB, color=col20, linewidth=1)

# xlabels, xticks and xtickslabels
plt.text(fmax_ax, xtm, '$f$', fontsize=fontsize, ha='right', va='baseline')

xticks = np.arange(fmin, fmax+0.1, 0.1)
for itick in xticks:
    plt.plot([itick, itick], [0, xtl], 'k')
    if math.isclose(math.fabs(itick), 0.1, rel_tol=1e-5):
        plt.text(itick, xtm, '${:.1f}$'.format(itick), fontsize=fontsize, ha='center', va='baseline')
    elif math.fabs(itick) < 0.01:
        pass
    else:
        plt.text(itick, -xtm+2, '${:.1f}$'.format(itick), fontsize=fontsize, ha='center', va='top')

# ylabels, yticks and ytickslabels
yticks = np.arange(-30, 55, 10)
for itick in yticks:
    plt.plot([0, ytl], [itick, itick], 'k')
    plt.text(-ytm, itick, '${:d}$'.format(itick), fontsize=fontsize, ha='right', va='center')

plt.text(0.02, ymax_ax, '$\mathrm{Magnitude\,(dB)}$', fontsize=fontsize, ha='left', va='center')

plt.text((fmax_ax + fmin_ax) / 2, ymin_ax, '$N={:d}$'.format(Ns[1]), fontsize=fontsize, ha='center', va='baseline')

leg = plt.legend(['$P_{xx}(f)$', '$\hat{P}_{xx}(f)$', '$\mathrm{CRLB}(P_{xx}(f))$',
                  '$\mathrm{var}(\hat{P}_{xx}(f))$'], loc="lower left", bbox_to_anchor=(0.52, 0.7), fontsize=fontsize,
                 frameon=False)

plt.axis('off')

# save as pdf image
plt.savefig('problem_7_26.pdf', bbox_inches='tight')

plt.show()

