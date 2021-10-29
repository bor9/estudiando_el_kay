import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rc('mathtext', fontset='cm')


def fun_g(x):
    return np.exp(-np.square(x) / 2) + 0.1 * np.exp(-np.square(x-10) / 2)


def fun_g_derivative(x):
    return -x * np.exp(-np.square(x) / 2) - 0.1 * (x - 10) * np.exp(-np.square(x - 10) / 2)


def fun_g_second_derivative(x):
    return (np.square(x) - 1) * np.exp(-np.square(x) / 2) \
           + 0.1 * (np.square(x - 10) - 1) * np.exp(-np.square(x - 10) / 2)


def newton_raphson(x0, nsteps):
    x_vals = np.zeros((nsteps+1,))
    i = 0
    x_vals[i] = x0
    for i in np.arange(1, nsteps+1):
        x_vals[i] = x_vals[i-1] - fun_g_derivative(x_vals[i-1]) / fun_g_second_derivative(x_vals[i-1])
    return x_vals


# abscissa values
xmin = -3
xmax = 13

x = np.linspace(xmin, xmax, 300)
# normal distribution and density values in x
g = fun_g(x)
g_derivative = fun_g_derivative(x)

nsteps = 20
x01 = 0.7
xn1 = newton_raphson(x01, nsteps)
x02 = 1.5
xn2 = newton_raphson(x02, nsteps)
x03 = 9.5
xn3 = newton_raphson(x03, nsteps)

print(xn2)

# axis parameters
dx = 0.5
xmin_ax = xmin - dx
xmax_ax = xmax + dx

# para la grafica de g(x)
ymax_ax1 = 1.2
ymin_ax1 = -0.2

# para la grafica de g'(x)
ymax_ax2 = 0.7
ymin_ax2 = -0.7

# x ticks labels margin
xtm = -0.13
ytm = 0.3
# font size
fontsize = 14
# colors from coolwarm
col1 = 'r'
col2 = 'deepskyblue'
col3 = 'green'

fig = plt.figure(0, figsize=(9, 6), frameon=False)

# grafica de g(x)
ax = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax1, ymax_ax1)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax1), xycoords='data', xy=(0, ymax_ax1), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, g, 'k', linewidth=2)
plt.plot(xn1, fun_g(xn1), '.', color=col1, markersize=9, label='$x_0={:.1f}$'.format(x01))
plt.plot(xn2, fun_g(xn2), '.', color=col2, markersize=9, label='$x_0={:.1f}$'.format(x02))
plt.plot(xn3, fun_g(xn3), '.', color=col3, markersize=9, label='$x_0={:.1f}$'.format(x03))

plt.plot(np.array([xn1[0:4], xn1[0:4]]), np.array([np.zeros((4,)), fun_g(xn1[0:4])]), '--', color=col1, lw=1)

xm = -0.18
plt.annotate(r'$x_0$', xytext=(xn1[0]+0.5, xm), xycoords='data', xy=(xn1[0], 0),
             textcoords='data', color=col1, fontsize=fontsize, va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.3", color=col1, relpos=(0.2, 1),
                             patchA=None, patchB=None, shrinkA=1, shrinkB=1))
plt.annotate(r'$x_2$', xytext=(xn1[2]-0.4, xm), xycoords='data', xy=(xn1[2], 0),
             textcoords='data', color=col1, fontsize=fontsize, va="baseline", ha="left",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.3", color=col1, relpos=(0.5, 1),
                             patchA=None, patchB=None, shrinkA=1, shrinkB=1))
plt.annotate(r'$x_1$', xytext=(xn1[1]-0.3, xm), xycoords='data', xy=(xn1[1], 0),
             textcoords='data', color=col1, fontsize=fontsize, va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.3", color=col1, relpos=(0.8, 1),
                             patchA=None, patchB=None, shrinkA=1, shrinkB=1))
plt.annotate(r'$x_3$', xytext=(xn1[3]-0.1, xm), xycoords='data', xy=(xn1[3], 0),
             textcoords='data', color=col1, fontsize=fontsize, va="baseline", ha="right",
             arrowprops=dict(arrowstyle="-|>, head_width=0.15, head_length=0.3", color=col1, relpos=(0.6, 1),
                             patchA=None, patchB=None, shrinkA=1, shrinkB=1))

plt.plot([xn2[0], xn2[0]], [0, fun_g(xn2[0])], '--', color=col2, lw=1)
plt.text(xn2[0]+0.1, xtm, '$x_0$', color=col2, fontsize=fontsize, ha='center', va='baseline')

plt.plot([xn3[0], xn3[0]], [0, fun_g(xn3[0])], '--', color=col3, lw=1)
plt.text(xn3[0], xtm, '$x_0$', color=col3, fontsize=fontsize, ha='center', va='baseline')

plt.text(ytm, ymax_ax1, '$g(x)$', fontsize=fontsize, ha='left', va='center')
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='center', va='baseline')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=10)

ax.legend(fontsize=fontsize, framealpha=0)

# grafica de g'(x)
ax = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax2, ymax_ax2)


# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax2), xycoords='data', xy=(0, ymax_ax2), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, g_derivative, 'k', linewidth=2)
plt.plot(xn1, fun_g_derivative(xn1), '.', color=col1, markersize=9)
plt.plot(xn2, fun_g_derivative(xn2), '.', color=col2, markersize=9)
plt.plot(xn3, fun_g_derivative(xn3), '.', color=col3, markersize=9)

plt.plot(xn1[0:2], [fun_g_derivative(xn1[0]), 0], color=col1, lw=1)
plt.plot(xn1[1:3], [fun_g_derivative(xn1[1]), 0], color=col1, lw=1)
plt.plot(xn1[2:4], [fun_g_derivative(xn1[2]), 0], color=col1, lw=1)

plt.plot(np.array([xn1[0:4], xn1[0:4]]), np.array([np.zeros((4,)), fun_g_derivative(xn1[0:4])]), '--', color=col1, lw=1)

plt.text(xn1[0]+0.1, -xtm, '$x_0$', color=col1, fontsize=fontsize, ha='center', va='top')

plt.plot([xn2[0], xn2[0]], [0, fun_g_derivative(xn2[0])], '--', color=col2, lw=1)
plt.text(xn2[0], -xtm, '$x_0$', color=col2, fontsize=fontsize, ha='center', va='top')

plt.plot([xn3[0], xn3[0]], [0, fun_g_derivative(xn3[0])], '--', color=col3, lw=1)
plt.text(xn3[0], xtm, '$x_0$', color=col3, fontsize=fontsize, ha='center', va='baseline')

plt.text(ytm, ymax_ax2, '$g\'(x)$', fontsize=fontsize, ha='left', va='center')
plt.text(xmax_ax, xtm, '$x$', fontsize=fontsize, ha='center', va='baseline')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=10)

# save as pdf image
plt.savefig('problem_7_18.pdf', bbox_inches='tight')

plt.show()

