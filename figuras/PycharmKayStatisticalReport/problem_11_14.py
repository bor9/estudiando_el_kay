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

print(c)
print(c**2)

# parametros de las elipses calculados de forma teorica
# elipse 1
a1 = c
b1 = c
# elipse 2
a2 = c
b2 = math.sqrt(2) * c
# elipse 3
a3 = math.sqrt(3) * c
b3 = c
theta3 = math.pi / 4

#####################
# END OF PARAMETERS #
#####################

t = np.linspace(0, 2 * math.pi, 300)

# ecuaciones paramétricas de las elipses

# elipse 1
x1 = a1 * np.cos(t)
y1 = b1 * np.sin(t)

# elipse 2
x2 = a2 * np.cos(t)
y2 = b2 * np.sin(t)

# elipse 3
x3 = a3 * np.cos(t) * np.cos(theta3) - b3 * np.sin(t) * np.sin(theta3)
y3 = a3 * np.cos(t) * np.sin(theta3) + b3 * np.sin(t) * np.cos(theta3)

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
plt.plot(c, 0, 'k.', markersize=ms)

# labels
plt.text(xmax_ax-0.2, -0.8, '$\epsilon_{\hat{a}}$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax-0.3, '$\epsilon_{\hat{b}}$', fontsize=fontsize, ha='right', va='top')
plt.text(c+0.2, xtm, '$c$', fontsize=fontsize, ha='left', va='baseline')


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
plt.plot(0, c, 'k.', markersize=ms)
plt.plot(0, -c, 'k.', markersize=ms)

# otros puntos
plt.plot(c, 0, 'k.', markersize=ms)
plt.plot(0, math.sqrt(2) * c, 'k.', markersize=ms)


# labels
plt.text(xmax_ax-0.2, -0.8, '$\epsilon_{\hat{a}}$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax-0.3, '$\epsilon_{\hat{b}}$', fontsize=fontsize, ha='right', va='top')
plt.text(ytm, c, '$c$', fontsize=fontsize, ha='right', va='center')
plt.text(ytm, -c, '$-c$', fontsize=fontsize, ha='right', va='center')
plt.text(c+0.2, xtm, '$c$', fontsize=fontsize, ha='left', va='baseline')
plt.text(0.2, math.sqrt(2) * c + 0.2, '$\sqrt{2}c$', fontsize=fontsize, ha='left', va='baseline')

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

# eje
plt.plot([ymin_ax, ymax_ax], [ymin_ax, ymax_ax], 'k--', lw=1)
plt.plot([ymin_ax, ymax_ax], [ymax_ax, ymin_ax], 'k--', lw=1)

# focos
plt.plot(math.sqrt(2) * c * np.cos(theta3), math.sqrt(2) * c * np.sin(theta3), 'k.', markersize=ms, zorder=3)
plt.plot(-math.sqrt(2) * c * np.cos(theta3), -math.sqrt(2) * c * np.sin(theta3), 'k.', markersize=ms)

# semi-ejes
plt.plot([0, np.sqrt(3) * c * np.cos(theta3)], [0, np.sqrt(3) * c * np.sin(theta3)], 'r-', lw=2, zorder=2)
plt.plot([0, -c * np.cos(theta3)], [0, c * np.sin(theta3)], 'b-', lw=2, zorder=2)

# angle
t1 = np.linspace(0, theta3, 30)
lt = 0.8
xt = lt * np.cos(t1)
yt = lt * np.sin(t1)
plt.plot(xt, yt, 'k', lw=1)

# labels
plt.text(xmax_ax-0.2, -0.8, '$\epsilon_{\hat{a}}$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax-0.3, '$\epsilon_{\hat{b}}$', fontsize=fontsize, ha='right', va='top')
plt.text(lt+0.1, 0.35, '$\pi/4$', fontsize=fontsize, ha='left', va='baseline')
plt.text(0.9, 2, '$\sqrt{3}c$', color='r', fontsize=fontsize, ha='center', va='center')
plt.text(-c*np.cos(theta3)/2+0.2, c * np.sin(theta3)/2+0.3, '$c$', color='b',
         fontsize=fontsize, ha='center', va='center')

plt.annotate(r'$(c,\,c)$', xytext=(0.5, 3.6), xycoords='data', xy=(c, c),
             textcoords='data', color='k', fontsize=fontsize, va="baseline", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.3", color='k', relpos=(0.6, 0),
                             patchA=None, patchB=None, shrinkA=0, shrinkB=3))


ax.set_yticklabels([])

# plt.axis('off')



# save as pdf image
plt.savefig('problem_11_14.pdf', bbox_inches='tight')

plt.show()

