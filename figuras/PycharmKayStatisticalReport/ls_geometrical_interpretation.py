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

fig = plt.figure(0, figsize=(9, 3), frameon=False)
# ax = fig.add_subplot(121)
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

# x1 vector
x1_x = 1
x1_y = -0.2
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1_x, x1_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# x2 vector
x2_x = -0.2
x2_y = -0.4
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x2_x, x2_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

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

# labels
plt.text(x1_x+0.04, x1_y-0.1, r'$\mathbf{h}_1$', fontsize=fontsize, ha='left', va='bottom')
plt.text(x2_x-0.04, x2_y-0.05, r'$\mathbf{h}_2$', fontsize=fontsize, ha='right', va='center')
plt.text(px1, py1, r'$S^2$', fontsize=fontsize, ha='left', va='top')

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
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmin_ax, -0.6), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))


s_x = 0.9
s_y = 0.75
# s vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# s projection
sp_x = s_x
sp_y = -0.4
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(sp_x, sp_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# sp to s arrow
plt.annotate("", xytext=(sp_x, sp_y), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

ax.add_patch(Polygon(vertices, facecolor=lgray, edgecolor='none'))


# labels
plt.text(sp_x-0.18, sp_y-0.03, r'$\hat{\mathbf{s}}$', fontsize=fontsize, ha='right', va='center')
plt.text(s_x, s_y, r'$\mathbf{x}$', fontsize=fontsize, ha='left', va='bottom')
plt.text(s_x+0.07, 0.25, r'$\boldsymbol{\varepsilon}=\mathbf{x}-\hat{\mathbf{s}}$', fontsize=fontsize, ha='left',
         va='center')

plt.text(px1, py1, r'$S^2$', fontsize=fontsize, ha='left', va='top')


plt.axis('off')

# save as pdf image
plt.savefig('ls_geometrical_interpretation.pdf', bbox_inches='tight')
plt.show()


