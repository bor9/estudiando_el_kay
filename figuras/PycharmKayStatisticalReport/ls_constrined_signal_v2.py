import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# colors
lgray = "#dddddd"  # ligth gray

# range of x and y axis
xmin_ax = -1
xmax_ax = 2
ymin_ax = -0.75
ymax_ax = 1

# font size
fontsize = 16
# arrows head length and head width
hl = 10
hw = 6
hl_ax = 8
hw_ax = 4

#########################

fig = plt.figure(0, figsize=(9, 3), frameon=False)
ax = plt.subplot2grid((1, 8), (0, 0), rowspan=1, colspan=4)


plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)


# x axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# z axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# y axis
y_e = -0.6
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmin_ax, y_e), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# pendiente del eje y
p = (0 - y_e) / (0 - xmin_ax)

s_x = 1.0
s_y = 0.75
# s vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# s projection
sp_x = s_x
sp_y = -0.15
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(sp_x, sp_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# sp to s line
plt.plot([s_x, s_x], [sp_y, s_y], 'k--', lw=1)


# region sombreada
px1 = 0.9
py1 = -0.5
px2 = 1.85
py2 = p * (px2 - px1) + py1
px4 = -1
py4 = py1
py3 = py2
px3 = (py3-py4) / p + px4
vertices = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]])
ax.add_patch(Polygon(vertices, facecolor=lgray, edgecolor='none'))

# lineas de proyeccion de sp sobre los ejes
spx_x = sp_x - sp_y / p
plt.plot([sp_x, spx_x], [sp_y, 0], 'k--', lw=1)
spy_x = sp_y / p
plt.plot([sp_x, spy_x], [sp_y, sp_y], 'k--', lw=1)

# labels
plt.text(sp_x+0.08, sp_y-0.05, r'$\hat{\mathbf{s}}$', fontsize=fontsize, ha='left', va='center')
plt.text(s_x, s_y, r'$\mathbf{x}$', fontsize=fontsize, ha='left', va='bottom')

plt.text(spx_x+0.18, 0.07, r'$x[1]$', fontsize=12, ha='center', va='bottom')
plt.text(spy_x-0.14, sp_y+0.05, r'$x[0]$', fontsize=12, ha='right', va='center')


plt.axis('off')

#########################

fig = plt.figure(0, figsize=(9, 3), frameon=False)
ax = plt.subplot2grid((1, 8), (0, 4), rowspan=1, colspan=4)


plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)


# x axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# z axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# y axis
y_e = -0.6
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmin_ax, y_e), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# pendiente del eje y
p = (0 - y_e) / (0 - xmin_ax)

# s vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# s projection
sp_x = s_x
sp_y = -0.15
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(sp_x, sp_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# sp to s line
plt.plot([s_x, s_x], [sp_y, s_y], 'k--', lw=1)

# region sombreada
ax.add_patch(Polygon(vertices, facecolor=lgray, edgecolor='none'))

# pendiente del subespacio
ps = -py1/px1
# subespacio de restriccion
plt.plot([px1, 0], [py1, 0], 'r--', lw=2)
xx = -0.09
plt.plot([xx, 0], [-ps*xx, 0], 'r--', lw=2)
spp_y = -0.27
# proyeccion
spp_x = -spp_y / ps
plt.plot([sp_x, spp_x], [sp_y, spp_y], 'k--', lw=1)
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(spp_x, spp_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# lineas de proyeccion de spp sobre los ejes
sppx_x = spp_x - spp_y / p
plt.plot([spp_x, sppx_x], [spp_y, 0], 'k--', lw=1)
sppy_x = spp_y / p
plt.plot([spp_x, sppy_x], [spp_y, spp_y], 'k--', lw=1)

# labels
plt.text(sp_x+0.08, sp_y-0.05, r'$\hat{\mathbf{s}}$', fontsize=fontsize, ha='left', va='center')
plt.text(s_x, s_y, r'$\mathbf{x}$', fontsize=fontsize, ha='left', va='bottom')
plt.text(spp_x, spp_y-0.05, r'$\hat{\mathbf{s}}_c$', fontsize=fontsize, ha='right', va='top')

plt.axis('off')

# save as pdf image
plt.savefig('ls_constrained_signal_v2.pdf', bbox_inches='tight')
plt.show()


