import matplotlib.pyplot as plt

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


fig = plt.figure(0, figsize=(4, 3), frameon=False)
ax = plt.subplot2grid((1, 8), (0, 0), rowspan=1, colspan=8)

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
m = y_e / xmin_ax


s_x = 0.9
s_y = 0.75
# s vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# x1 vector
x1_x = 0.6
x1_y = 0
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1_x, x1_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# hat theta
hs_x = s_x
hs_y = -0.45
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(hs_x, hs_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# proyecciones
x0 = hs_x
y0 = hs_y
y1 = 0
x1 = x0 - y0 / m
plt.plot([x0, x1], [y0, y1], 'k--', lw=1)
plt.plot([x0, x0-x1], [y0, y0], 'k--', lw=1)


plt.annotate("", xytext=(hs_x, hs_y), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, color='r', shrink=0.002))


# x2 vector
x2_x = -0.35
x2_y = m * x2_x
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x2_x, x2_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# proyecciones en los ejes
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1, y1), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x0-x1, y0), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# labels
plt.text(x1_x, x1_y+0.08, r'$e_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(x1, y1+0.08, r'$(x_3,\,e_2)e_2$', fontsize=fontsize, ha='center', va='baseline')
plt.text(x2_x-0.05, x2_y+0.1, r'$e_1$', fontsize=fontsize, ha='right', va='center')
plt.text(x0-x1-0.1, y0+0.05, r'$(x_3,\,e_1)e_1$', fontsize=fontsize, ha='right', va='center')
plt.text(s_x, s_y, r'$x_3$', fontsize=fontsize, ha='right', va='bottom')
plt.text(s_x+0.1, s_y, r'$z_3$', fontsize=fontsize, ha='left', va='top', color='r')
plt.text(hs_x+0.08, hs_y-0.15, r'$(x_3,\,e_1)e_1+(x_3,\,e_2)e_2$', fontsize=fontsize, ha='center', va='center')

plt.axis('off')

# save as pdf image
plt.savefig('problem_12_10.pdf', bbox_inches='tight')
plt.show()


