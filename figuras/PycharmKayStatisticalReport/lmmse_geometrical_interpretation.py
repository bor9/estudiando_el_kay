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
# ax = fig.add_subplot(121)
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
p = (0 - y_e) / (0 - xmin_ax)

s_x = 0.9
s_y = 0.75
# s vector
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(s_x, s_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))


# x1 vector
x1_x = 1.4
x1_y = -0.2
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x1_x, x1_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))
# x2 vector
x2_x = 0.25
x2_y = -0.5
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(x2_x, x2_y), textcoords='data',
             arrowprops=dict(width=1, headwidth=hw, headlength=hl, facecolor='black', shrink=0.002))


# labels
plt.text(x1_x+0.1, x1_y-0.1, r'$x[1]$', fontsize=fontsize, ha='left', va='bottom')
plt.text(x2_x+0.03, x2_y-0.07, r'$x[0]$', fontsize=fontsize, ha='center', va='top')
plt.text(s_x+0.12, s_y, r'$\theta$', fontsize=fontsize, ha='center', va='bottom')
plt.text(xmin_ax, ymax_ax, r'$N=2$', fontsize=fontsize, ha='left', va='top')


plt.axis('off')

# save as pdf image
plt.savefig('lmmse_geometrical_interpretation.pdf', bbox_inches='tight')
plt.show()


