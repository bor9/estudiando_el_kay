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
fontsize = 13
# arrows head length and head width
hl = 8
hw = 5
hl_ax = 6
hw_ax = 3

fig = plt.figure(0, figsize=(9, 2.5), frameon=False)
ax = plt.subplot2grid((1, 12), (0, 0), rowspan=1, colspan=4)

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
m = (0 - y_e) / (0 - xmin_ax)

A_x = 0.4
A_y = 0.8
# A vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(A_x, A_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# x[1] vector
x1_x = 1.3
x1_y = -0.3
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1_x, x1_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# x[0] vector
x0_x = -0.8
x0_y = m * x0_x
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x0_x, x0_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# hat A
hA_x = A_x
hA_y = -0.3
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(hA_x, hA_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# proyecciones
x0 = hA_x
y0 = hA_y
y1 = 0
x1 = x0 - y0 / m
plt.plot([x0, x1], [y0, y1], 'k--', lw=1)
plt.plot([x0, x0-x1], [y0, y0], 'k--', lw=1)

# hat A[0]
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x0-x1, y0), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# error
plt.plot([hA_x, A_x], [hA_y, A_y], 'k--', lw=1)


# labels
plt.text(x1_x+0.1, x1_y-0.1, r'$x[1]$', fontsize=fontsize, ha='left', va='bottom')
plt.text(x0_x + 0.15, x0_y - 0.05, r'$x[0]$', fontsize=fontsize, ha='left', va='center')
plt.text(A_x + 0.05, A_y, r'$A$', fontsize=fontsize, ha='left', va='bottom')
plt.text(hA_x + 0.02, hA_y - 0.15, r'$\hat{A}[1]$', fontsize=fontsize, ha='left', va='center')
plt.text(x0-x1 - 0.08, y0, r'$\hat{A}[0]$', fontsize=fontsize, ha='right', va='bottom')

plt.axis('off')

###############################################################

ax = plt.subplot2grid((1, 12), (0, 4), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# x axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# z axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmin_ax, y_e), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))

# A vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(A_x, A_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# hat A
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(hA_x, hA_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# proyecciones
plt.plot([x0, x1], [y0, y1], 'k--', lw=1)
plt.plot([x0, x0-x1], [y0, y0], 'k--', lw=1)

# hat A[0]
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x0-x1, y0), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# Delta hat A[0]
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1, y1), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# error
plt.plot([hA_x, A_x], [hA_y, A_y], 'k--', lw=1)


# labels
plt.text(A_x + 0.05, A_y, r'$A$', fontsize=fontsize, ha='left', va='bottom')
plt.text(hA_x + 0.02, hA_y - 0.15, r'$\hat{A}[1]$', fontsize=fontsize, ha='left', va='center')
plt.text(x0-x1 - 0.08, y0, r'$\hat{A}[0]$', fontsize=fontsize, ha='right', va='bottom')
plt.text(x1, y1+0.1, r'$\Delta\hat{A}[1]$', fontsize=fontsize, ha='center', va='baseline')


plt.axis('off')

###############################################################

ax = plt.subplot2grid((1, 12), (0, 8), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# x axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# z axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))
# y axis
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(xmin_ax, y_e), textcoords='data',
             arrowprops=dict(width=0.01, headwidth=hw_ax, headlength=hl_ax, facecolor='black', shrink=0.002))


# Delta hat A[0]
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1, y1), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))


# x[0] vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x0_x, x0_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# x[1] vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1_x, x1_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

# proyeccion de x[1]
y2 = 0
x2 = (y2 - (x1_y - m * x1_x)) / m

plt.annotate("", xytext=(x2, y2), xycoords='data', xy=(x1_x, x1_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))

plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x2, y2), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))



# labels
plt.text(x1, y1+0.1, r'$\Delta\hat{A}[1]$', fontsize=fontsize, ha='right', va='baseline')
plt.text(x0_x + 0.15, x0_y - 0.05, r'$x[0]$', fontsize=fontsize, ha='left', va='center')

plt.text(x1_x-0.15, x1_y-0.2, r'$x[1]$', fontsize=fontsize, ha='right', va='bottom')
plt.text(x1_x+0.2, x1_y-0.1, r'$\hat{x}[1|0]$', fontsize=fontsize, ha='left', va='bottom')

plt.text(x2, y2+0.1, r'$x[1]-\hat{x}[1|0]$', fontsize=fontsize, ha='center', va='baseline')




plt.axis('off')

# save as pdf image
plt.savefig('lmmse_geometrical_interpretation_sequential.pdf', bbox_inches='tight')
plt.show()


