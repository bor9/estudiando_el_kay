import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import math

from matplotlib import cm
from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

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

# parámetros

# probabilidad
P = 0.9
c = math.sqrt(2 * math.log(1 / (1 - 0.9)))

sigma = 10
sigma11 = 1
sigma21 = 1

sigma12 = 1
sigma22 = 2

sigma13 = 1
sigma23 = 10

#####################
# END OF PARAMETERS #
#####################


# parametros de las elipses calculados de forma teorica
# elipse 1
a1 = np.sqrt(c/(1/sigma11+1/sigma))
b1 = np.sqrt(c/(1/sigma21+1/sigma))
# elipse 2
a2 = np.sqrt(c/(1/sigma12+1/sigma))
b2 = np.sqrt(c/(1/sigma22+1/sigma))
# elipse 3
a3 = np.sqrt(c/(1/sigma13+1/sigma))
b3 = np.sqrt(c/(1/sigma23+1/sigma))

t = np.linspace(0, 2 * math.pi, 300)

# ecuaciones paramétricas de las elipses

# elipse 1
x1 = a1 * np.cos(t)
y1 = b1 * np.sin(t)

# elipse 2
x2 = a2 * np.cos(t)
y2 = b2 * np.sin(t)

# elipse 3
x3 = a3 * np.cos(t)
y3 = b3 * np.sin(t)

# axis parameters
xmax_ax = 2.2 * c
xmin_ax = -xmax_ax
ymax_ax = 2.2 * c
ymin_ax = -ymax_ax

# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.7
ytm = -0.4
# font size
fontsize = 14
# markersize
ms = 9
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0.1)
col20 = scalarMap.to_rgba(1)


f = plt.figure(0, figsize=(10, 3), frameon=False)
ax = plt.subplot2grid((1, 15), (0, 0), rowspan=1, colspan=5)

plt.axis('equal')
plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

f.canvas.draw()
ymin_ax, ymax_ax = ax.get_ylim()
xmin_ax, xmax_ax = ax.get_xlim()

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)


# elipse 1
plt.plot(x1, y1, 'k-', lw=2.5)

# focos
plt.plot(0, 0, 'k.', markersize=ms)

# otros puntos
plt.plot(a1, 0, 'k.', markersize=ms)

# labels
plt.text(xmax_ax-0.1, -0.8, '$\epsilon_1$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax-0.1, '$\epsilon_2$', fontsize=fontsize, ha='right', va='top')
plt.text(a1+0.2, xtm, '$a$', fontsize=fontsize, ha='left', va='baseline')


##########################
ax = plt.subplot2grid((1, 15), (0, 5), rowspan=1, colspan=5)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)

# elipse 2
plt.plot(x2, y2, 'k-', lw=2.5)

# focos
plt.plot(0, np.sqrt(b2**2-a2**2), 'k.', markersize=ms)
plt.plot(0, -np.sqrt(b2**2-a2**2), 'k.', markersize=ms)


# otros puntos
plt.plot(a2, 0, 'k.', markersize=ms)
plt.plot(0, b2, 'k.', markersize=ms)


# labels
plt.text(xmax_ax-0.1, -0.8, '$\epsilon_1$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax-0.1, '$\epsilon_2$', fontsize=fontsize, ha='right', va='top')
plt.text(a2+0.2, xtm, '$a$', fontsize=fontsize, ha='left', va='baseline')
plt.text(0.2, b2 + 0.2, '$b$', fontsize=fontsize, ha='left', va='baseline')

ax.set_yticklabels([])

#################
ax = plt.subplot2grid((1, 15), (0, 10), rowspan=1, colspan=5)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002), zorder=-1)

# elipse 3
plt.plot(x3, y3, 'k-', lw=2.5)

# focos
plt.plot(0, np.sqrt(b3**2-a3**2), 'k.', markersize=ms)
plt.plot(0, -np.sqrt(b3**2-a3**2), 'k.', markersize=ms)


# otros puntos
plt.plot(a3, 0, 'k.', markersize=ms)
plt.plot(0, b3, 'k.', markersize=ms)


# labels
plt.text(xmax_ax-0.1, -0.8, '$\epsilon_1$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax-0.1, '$\epsilon_2$', fontsize=fontsize, ha='right', va='top')
plt.text(a3+0.2, xtm, '$a$', fontsize=fontsize, ha='left', va='baseline')
plt.text(0.2, b3 + 0.2, '$b$', fontsize=fontsize, ha='left', va='baseline')


ax.set_yticklabels([])

# plt.axis('off')



# save as pdf image
plt.savefig('problem_11_15.pdf', bbox_inches='tight')

plt.show()

