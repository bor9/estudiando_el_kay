import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rc
from scipy.linalg import toeplitz
from numpy.linalg import inv

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

# number of samples
N = 50
# autocorrelation parameters
r0 = 1.81
r2 = 0.9


#####################
# END OF PARAMETERS #
#####################
# abscissa values
xmin = 0
xmax = 0.5

num_samples = 2000
f = np.linspace(xmin, xmax, num_samples)
Pww = r0 + 2 * r2 * np.cos(4 * math.pi * f)

# autocorrelation vector
r = np.zeros((N,))
r[0] = r0
r[2] = r2
# covariance matrix
C = toeplitz(r)
C_inv = inv(C)
Ns = np.arange(N)
ns = np.arange(num_samples)
var_A = np.zeros((num_samples,))
for n in ns:
    s = np.cos(2 * math.pi * f[n] * Ns)
    var_A[n] = 1 / s.dot(C_inv.dot(s))


# axis parameters
dx = 0.04
xmin_ax = xmin - dx
xmax_ax = xmax + dx

dy = 0.6
ymax_ax = np.amax(Pww) + dy
ymin_ax = -dy


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.7
ytm = 0.01
# font size
fontsize = 14

fig = plt.figure(0, figsize=(7, 6), frameon=False)

# PLOT OF P_ww
ax = fig.add_subplot(211)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, Pww, color='k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, -0.55, '$f$', fontsize=fontsize, ha='center', va='baseline')

plt.plot([1/2, 1/2], [0, xtl], 'k')
plt.text(1/2, xtm, '$\\dfrac{1}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([1/4, 1/4], [0, xtl], 'k')
plt.text(1/4, xtm, '$\\dfrac{1}{4}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, -0.55, '$0$', fontsize=fontsize, ha='center', va='baseline')


Pww0 = r0 + 2 * r2
plt.text(-ytm, Pww0, '${:.2f}$'.format(Pww0), fontsize=fontsize, ha='right', va='center')

plt.text(ytm, ymax_ax, '$P_{ww}(f)$', fontsize=fontsize, ha='left', va='center')

plt.axis('off')

# PLOT OF variance

# factor de escala para que las gráficas tengan la misma escala en y

sf = Pww0 / np.amax(var_A)
var_A = sf * var_A

ax = fig.add_subplot(212)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f, var_A, color='k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, -0.55, '$f_1$', fontsize=fontsize, ha='center', va='baseline')

plt.plot([1/2, 1/2], [0, xtl], 'k')
plt.text(1/2, xtm, '$\\dfrac{1}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([1/4, 1/4], [0, xtl], 'k')
plt.text(1/4, xtm, '$\\dfrac{1}{4}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-ytm, -0.55, '$0$', fontsize=fontsize, ha='center', va='baseline')


plt.text(ytm, ymax_ax, '${\\rm var}(\hat{A})$', fontsize=fontsize, ha='left', va='center')

ti = np.arange(0.05, 0.16, 0.05)
tis = sf * ti
for i in np.arange(len(ti)):
    plt.plot([0, ytl], [tis[i], tis[i]], 'k')
    plt.text(-ytm, tis[i], '${:.2f}$'.format(ti[i]), fontsize=fontsize, ha='right', va='center')



plt.axis('off')




# save as pdf image
plt.savefig('problem_6_10.pdf', bbox_inches='tight')

plt.show()

