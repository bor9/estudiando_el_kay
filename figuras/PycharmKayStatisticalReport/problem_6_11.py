import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rc
from scipy.linalg import toeplitz
from numpy.linalg import inv
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

# numero de muestras = orden del filtro FIR
N = 50
# parámetros de la autocorrelación
a = 1.81  # r_ww[0]
b1 = 0.9  # r_ww[2] - ruido de banda suprimida
b2 = -0.9  # r_ww[2] - ruido pasabanda

#####################
# END OF PARAMETERS #
#####################


# coeficientes del filtro con ruido de ACF rww1[k]
rww1 = np.zeros((N,))
rww1[0] = a
rww1[2] = b1
# matriz de covarianza
C1 = toeplitz(rww1)
w1_var = 1 / np.sum(inv(C1))
# coeficientes del filtro
h1 = np.sum(inv(C1), axis=1) * w1_var
# respuesta en frecuencia del filtro
w, H1 = signal.freqz(h1, a=1, worN=512, whole=False, plot=None)
f = w / (2 * math.pi) # en veriones mas nuevas de scipy esto puede hacerse en el comando freqz con fs=1
# PSD del ruido
Pww1 = a + 2 * b1 * np.cos(4 * math.pi * f)

# abscissa values
xmin = 0
xmax = 0.5

# axis parameters
dx = 0.04
xmin_ax = xmin - dx
xmax_ax = xmax + dx

dy = 0.6
ymax_ax = np.amax(Pww1) + dy
ymin_ax = -dy


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.7
ytm = 0.01
# font size
fontsize = 14

fig = plt.figure(0, figsize=(10, 6), frameon=False)

# PLOT OF P_ww
ax = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, Pww1, color='k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, -0.55, '$f$', fontsize=fontsize, ha='center', va='baseline')

plt.plot([1/2, 1/2], [0, xtl], 'k')
plt.text(1/2, xtm, '$\\dfrac{1}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([1/4, 1/4], [0, xtl], 'k')
plt.text(1/4, xtm, '$\\dfrac{1}{4}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, -0.55, '$0$', fontsize=fontsize, ha='right', va='baseline')


Pww10 = a + 2 * b1
plt.text(-ytm, Pww10, '${:.2f}$'.format(Pww10), fontsize=fontsize, ha='right', va='center')

plt.text(1.5*ytm, ymax_ax, '$P_{{ww}_1}(f)$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# Plot of filter response

# factor de escala para que las gráficas tengan la misma escala en y

abs_H1 = np.abs(H1)
sf = Pww10 / np.amax(abs_H1)
abs_H1 = sf * abs_H1

ax = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, abs_H1, color='k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, -0.55, '$f$', fontsize=fontsize, ha='center', va='baseline')

plt.plot([1/2, 1/2], [0, xtl], 'k')
plt.text(1/2, xtm, '$\\dfrac{1}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([1/4, 1/4], [0, xtl], 'k')
plt.text(1/4, xtm, '$\\dfrac{1}{4}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, -0.55, '$0$', fontsize=fontsize, ha='right', va='baseline')


plt.text(1.5*ytm, ymax_ax, '$|H_1(e^{j2\pi f})|$', fontsize=fontsize, ha='left', va='center')

plt.plot([0, ytl], [sf, sf], 'k')
plt.text(-ytm, sf, '$1$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

#################################
# The same but with noise w2[n] #
#################################

# coeficientes del filtro con ruido de ACF rww1[k]
rww2 = np.zeros((N,))
rww2[0] = a
rww2[2] = b2
# matriz de covarianza
C2 = toeplitz(rww2)
w2_var = 1 / np.sum(inv(C2))
# coeficientes del filtro
h2 = np.sum(inv(C2), axis=1) * w2_var
# respuesta en frecuencia del filtro
w, H2 = signal.freqz(h2, a=1, worN=512, whole=False, plot=None)
f = w / (2 * math.pi) # en veriones mas nuevas de scipy esto puede hacerse en el comando freqz con fs=1
# PSD del ruido
Pww2 = a + 2 * b2 * np.cos(4 * math.pi * f)

ax = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, Pww2, color='k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, -0.55, '$f$', fontsize=fontsize, ha='center', va='baseline')

plt.plot([1/2, 1/2], [0, xtl], 'k')
plt.text(1/2, xtm, '$\\dfrac{1}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([1/4, 1/4], [0, xtl], 'k')
plt.text(1/4, xtm, '$\\dfrac{1}{4}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, -0.55, '$0$', fontsize=fontsize, ha='right', va='baseline')


Pww10 = a + 2 * b1
plt.text(-ytm, Pww10, '${:.2f}$'.format(Pww10), fontsize=fontsize, ha='right', va='center')
plt.plot([0, ytl], [Pww10, Pww10], 'k')

plt.text(1.5*ytm, ymax_ax, '$P_{{ww}_2}(f)$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# Plot of filter response

# factor de escala para que las gráficas tengan la misma escala en y

abs_H2 = np.abs(H2)
sf = Pww10 / np.amax(abs_H2)
abs_H2 = sf * abs_H2

ax = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, abs_H2, color='k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, -0.55, '$f$', fontsize=fontsize, ha='center', va='baseline')

plt.plot([1/2, 1/2], [0, xtl], 'k')
plt.text(1/2, xtm, '$\\dfrac{1}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([1/4, 1/4], [0, xtl], 'k')
plt.text(1/4, xtm, '$\\dfrac{1}{4}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, -0.55, '$0$', fontsize=fontsize, ha='right', va='baseline')


plt.text(1.5*ytm, ymax_ax, '$|H_2(e^{j2\pi f})|$', fontsize=fontsize, ha='left', va='center')

plt.plot([0, ytl], [sf, sf], 'k')
plt.text(-ytm, sf, '$1$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')


# save as pdf image
plt.savefig('problem_6_11.pdf', bbox_inches='tight')


#############################
# Prueba: filtrado de ruido #
#############################

# generación de ruido con ACF r_ww[k] a partir de ruido blanco v[n]
var_v = 1
# r_ww[k] = r0 * delta[k] + r2 * delta[k-2]
r0 = a  # autocorrelacion en 0
r2 = b1  # autocorrelacion en 2
roots = np.roots([1, -var_v * r0, (var_v ** 2) * (r2 ** 2)])
d1 = math.sqrt(np.amin(roots))
d2 = -d1
c = math.sqrt(var_v * r0 - d1 ** 2)

#np.random.seed(2)
#np.random.seed(3)
np.random.seed(4)
ord = 2  # filter order
Nr = 10000  # number of samples
x = np.random.normal(loc=0.0, scale=1.0, size=Nr + ord)
y1 = np.zeros((Nr, ))
y2 = np.zeros((Nr, ))
ns = np.arange(Nr)
for n in np.arange(Nr + ord):
    y1[n - ord] = c * x[n] + d1 * x[n - ord]
    y2[n - ord] = c * x[n] + d2 * x[n - ord]
# sample autocorrelation
kmax = 8
ks = np.arange(kmax)
sample_rww1 = np.zeros((kmax, ))
sample_rww2 = np.zeros((kmax, ))
for k in ks:
    sample_rww1[k] = np.mean(y1[0: Nr - 1 - k] * y1[k: Nr - 1])
    sample_rww2[k] = np.mean(y2[0: Nr - 1 - k] * y2[k: Nr - 1])

# data and filtered data
A = 4
x1 = A + y1
x2 = A + y2

s1 = signal.lfilter(h1, 1, x1)
s2 = signal.lfilter(h2, 1, x2)

nmax = 140

# fig = plt.figure(1, figsize=(10, 6), frameon=False)
# ax = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=4)
# plt.plot(ns[:nmax], y1[:nmax], color='k', linewidth=2)
#
# ax = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=4)
# plt.plot(ks, sample_rww2, color='k', linewidth=2)

# abscissa values
nmax = 140
xmin = 0
xmax = nmax

# axis parameters
dx = 5
xmin_ax = xmin - dx
xmax_ax = xmax + 2 * dx

dy = 0.6
ymax_ax = 8 + dy
ymin_ax = -dy

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# font size
fontsize = 14

fig = plt.figure(2, figsize=(10, 6), frameon=False)

ax = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=4)
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)
xtm, foo = convert_display_to_data_coordinates(ax.transData, length=25)
foo, ytm = convert_display_to_data_coordinates(ax.transData, length=10)
xtm = -xtm
ytm = -ytm

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(ns[:nmax], x1[:nmax], color='r', linewidth=2, label='$x[n]$')
plt.plot(ns[:nmax], s1[:nmax], color='k', linewidth=2, label='$x[n]\;\;{\\rm filtrado}$')
# nivel de DC
plt.plot([0, nmax-1], [A, A], 'k--', linewidth=1,dashes=[6, 3])

plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')

nn = np.arange(20, nmax+10, 20)
for n in nn:
    plt.text(n, xtm, '${:d}$'.format(n), fontsize=fontsize, ha='center', va='baseline')
    plt.plot([n, n], [0, xtl], 'k')
xx = np.arange(2, 8, 2)
for x in xx:
    plt.text(ytm, x, '${:d}$'.format(x), fontsize=fontsize, ha='right', va='center')
    plt.plot([0, ytl], [x, x], 'k')

leg = plt.legend(loc=(0.5, 0.85), fontsize=fontsize, frameon=False, ncol=2)
plt.axis('off')


ax = plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=4)
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(ns[:nmax], x2[:nmax], color='r', linewidth=2, label='$x[n]$')
plt.plot(ns[:nmax], s2[:nmax], color='k', linewidth=2, label='$x[n]\;\;{\\rm filtrado}$')
# nivel de DC
plt.plot([0, nmax-1], [A, A], 'k--', linewidth=1,dashes=[6, 3])

plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')

nn = np.arange(20, nmax+10, 20)
for n in nn:
    plt.text(n, xtm, '${:d}$'.format(n), fontsize=fontsize, ha='center', va='baseline')
    plt.plot([n, n], [0, xtl], 'k')
xx = np.arange(2, 8, 2)
for x in xx:
    plt.text(ytm, x, '${:d}$'.format(x), fontsize=fontsize, ha='right', va='center')
    plt.plot([0, ytl], [x, x], 'k')
plt.axis('off')

############################
# Autocorrelación muestral #
############################
# abscissa values

xmin = 0
xmax = kmax

# axis parameters
dx = 1
xmin_ax = xmin - dx
xmax_ax = xmax + dx

dy = 0.2
ymax_ax = 2.2 + dy
ymin_ax = -1.2 - dy

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# font size
fontsize = 14

ax = plt.subplot2grid((4, 5), (0, 4), rowspan=2, colspan=1)
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)
xtm, foo = convert_display_to_data_coordinates(ax.transData, length=25)
foo, ytm = convert_display_to_data_coordinates(ax.transData, length=10)
xtm = -xtm
ytm = -ytm

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(ks, sample_rww1, linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=5)
plt.setp(bl, visible=False)

plt.text(xmax_ax, xtm, '$k$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, ymax_ax, '$\hat{r}_{ww_1}[k]$', fontsize=fontsize, ha='left', va='center')

plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
xx = [-1, 1, 2]
for x in xx:
    plt.text(ytm, x, '${:d}$'.format(x), fontsize=fontsize, ha='right', va='center')
    plt.plot([0, ytl], [x, x], 'k')

n = 5
plt.text(n, xtm, '${:d}$'.format(n), fontsize=fontsize, ha='center', va='baseline')
plt.plot([n, n], [0, xtl], 'k')
plt.axis('off')

ax = plt.subplot2grid((4, 5), (2, 4), rowspan=2, colspan=1)
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)
xtm, foo = convert_display_to_data_coordinates(ax.transData, length=25)
foo, ytm = convert_display_to_data_coordinates(ax.transData, length=10)
xtm = -xtm
ytm = -ytm

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(ks, sample_rww2, linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=5)
plt.setp(bl, visible=False)

plt.text(xmax_ax, xtm, '$k$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, ymax_ax, '$\hat{r}_{ww_2}[k]$', fontsize=fontsize, ha='left', va='center')

plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
xx = [-1, 1, 2]
for x in xx:
    plt.text(ytm, x, '${:d}$'.format(x), fontsize=fontsize, ha='right', va='center')
    plt.plot([0, ytl], [x, x], 'k')

n = 5
plt.text(n, xtm, '${:d}$'.format(n), fontsize=fontsize, ha='center', va='baseline')
plt.plot([n, n], [0, xtl], 'k')
plt.axis('off')

# save as pdf image
plt.savefig('problem_6_11_sequences.pdf', bbox_inches='tight')



plt.show()

print(w1_var)
print(w2_var)