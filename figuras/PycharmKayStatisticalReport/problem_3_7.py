import matplotlib.pyplot as plt
import numpy as np
import math
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

# number of samples
N = 20

#####################
# END OF PARAMETERS #
#####################


# abscissa values
xmin = -0.19
xmax = 0.69

f0 = np.linspace(xmin, xmax, 1000)
sinc = np.abs(np.sin(2*math.pi*f0*N)/(N*np.sin(2*math.pi*f0)))


# axis parameters
dx = 0.05
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ymax_ax = 1.3
ymin_ax = -0.3


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.2
ytm = 0.01
# font size
fontsize = 14

fig = plt.figure(0, figsize=(10, 3), frameon=False)

# PLOT OF F(x | x < a)
ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f0, sinc, color='k', linewidth=2)

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$f_0$', fontsize=fontsize, ha='right', va='baseline')

plt.plot([1/2, 1/2], [0, xtl], 'k')
plt.text(1/2, xtm, '$\\dfrac{1}{2}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([1/(2*N), 1/(2*N)], [0, xtl], 'k')
plt.text(1/(2*N), xtm, '$\\dfrac{1}{2N}$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([1/4, 1/4], [0, xtl], 'k')
plt.text(1/4, xtm, '$\\dfrac{1}{4}$', fontsize=fontsize, ha='center', va='baseline')


plt.text(-ytm, 1.01, '$1$', fontsize=fontsize, ha='right', va='baseline')
plt.plot([0, ytl], [1, 1], 'k')

plt.text(ytm, ymax_ax, '$\left|\\dfrac{\sin(2\pi f_0 N)}{N\,\sin(2\pi f_0)}\\right|$',
         fontsize=fontsize, ha='left', va='center')


plt.text(xmax_ax, ymax_ax, '$N={}$'.format(N), fontsize=fontsize, ha='right', va='center')

plt.axis('off')


# save as pdf image
plt.savefig('problem_3_7.pdf', bbox_inches='tight')

plt.show()

