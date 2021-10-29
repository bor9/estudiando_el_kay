import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import math

from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')


#####################################
# PARAMETERS - This can be modified #
#####################################

# hyperbola parameters
# focus: (c, 0) and (-c, 0)
c = 4
# vertice: (a, 0)
a = -2.5

# plot axis max values
xmin = -7
xmax = 9

ymin = -8
ymax = 8

#####################
# END OF PARAMETERS #
#####################

# hyperbola
xh = np.linspace(xmin, a, 300)
b = math.sqrt(c**2 - a**2)
yh= np.sqrt((np.square(xh)/a**2-1)*b**2)

# circunsference
xc = np.linspace(c-2*a, c+2*a, 500)
yc = np.sqrt(4*a**2-np.square(xc-c))

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)


fig = plt.figure(0, figsize=(4, 4), frameon=False)
ax = fig.add_subplot(111)
ax.axis('equal')

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)


# focus
plt.plot(-c, 0, 'k.', markersize=10)
plt.plot(c, 0, 'k.', markersize=10)

# hyperbola
plt.plot(xh, yh, color=col10, linewidth=2)
plt.plot(xh, -yh, color=col10, linewidth=2)
# circunsference
plt.plot(xc, yc, 'k', linewidth=1)
plt.plot(xc, -yc, 'k', linewidth=1)

# lines between focus and points in hyperbola
x_s1 = -2.8
y_s1 = math.sqrt((x_s1**2/a**2-1)*b**2)
plt.plot([-c, x_s1], [0, y_s1], color=col20, linewidth=1)
plt.plot([c, x_s1], [0, y_s1], color=col20, linewidth=1)
x_s2 = -6
y_s2 = math.sqrt((x_s2**2/a**2-1)*b**2)
plt.plot([-c, x_s2], [0, y_s2], color=col20, linewidth=1)
plt.plot([c, x_s2], [0, y_s2], color=col20, linewidth=1)

# radial lines: x_i1 is the x coordinate of the intersection between the distance and the circunsference.
# y = mx + n
m = y_s1/(x_s1-c)
n = y_s1 - m * x_s1
coeff = [1+m**2, 2*(m*n-c), n**2+c**2-4*a**2]
x_i1 = np.amin(np.roots(coeff))
y_i1 = math.sqrt(4*a**2-(x_i1-c)**2)
plt.plot([x_i1, c], [y_i1, 0], 'k', linewidth=1)
m = y_s2/(x_s2-c)
n = y_s2 - m * x_s2
coeff = [1+m**2, 2*(m*n-c), n**2+c**2-4*a**2]
x_i2 = np.amin(np.roots(coeff))
y_i2 = math.sqrt(4*a**2-(x_i2-c)**2)
plt.plot([x_i2, c], [y_i2, 0], 'k', linewidth=1)

# points in hyperbola
plt.plot(x_s1, y_s1, 'k.', markersize=6)
plt.plot(x_s2, y_s2, 'k.', markersize=6)

# labels
fontsize1 = 10
fontsize2 = 12
plt.text(-c, -1.2, '$\\rm{Antena\;1}$', fontsize=fontsize1, ha='right', va='baseline')
plt.text(c, -1.2, '$\\rm{Antena\;2}$', fontsize=fontsize1, ha='left', va='baseline')
plt.text(x_s1+0.2, y_s1+0.2, '$s$', fontsize=fontsize2, ha='left', va='baseline')
plt.text(x_s2-0.2, y_s2-0.8, '$s\'$', fontsize=fontsize2, ha='right', va='baseline')
plt.text(x_s1+0.35, 0.2, '$R_1$', fontsize=fontsize2, ha='right', va='center', color=col20)
plt.text(x_s2, y_s2/2, '$R_1\'$', fontsize=fontsize2, ha='center', va='center', color=col20)
plt.text(-0.2, -0.4, '$R_2-R_1$', fontsize=fontsize2, ha='left', va='center')


# R2' lines
m = y_s2/(x_s2-c)
d = 0.2
x_11 = c + 0.4
x_12 = x_11 + d
x_13 = x_12 + d
x_21 = x_s2 + 0.4
x_22 = x_21 + d
x_23 = x_22 + d
y_11 = -x_11/m + c/m
y_12 = -x_12/m + c/m
y_13 = -x_13/m + c/m
y_21 = -x_21/m + y_s2 + x_s2/m
y_22 = -x_22/m + y_s2 + x_s2/m
y_23 = -x_23/m + y_s2 + x_s2/m

plt.plot([x_11, x_13], [y_11, y_13], 'k', linewidth=1)
plt.plot([x_21, x_23], [y_21, y_23], 'k', linewidth=1)
plt.plot([x_12, x_22], [y_12, y_22], 'k', linewidth=1)
plt.text((x_22+x_12)/2, (y_22+y_12)/2+0.5, '$R_2\'$', fontsize=fontsize2, ha='left', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('example_6_3a.pdf', bbox_inches='tight')

plt.show()

