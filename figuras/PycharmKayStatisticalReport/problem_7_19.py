import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import math

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
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


def fun_g(f0):
    return 2 * math.pi * np.sum(ns * x * np.sin(2 * math.pi * f0 * ns))


def fun_g_derivative(f0):
    return ((2 * math.pi) ** 2) * np.sum(np.square(ns) * x * np.cos(2 * math.pi * f0 * ns))


def newton_raphson(x0, nsteps):
    x_act = x0
    for _ in np.arange(nsteps):
        x_act = x_act - fun_g(x_act) / fun_g_derivative(x_act)
    return x_act


#####################################
# PARAMETERS - This can be modified #
#####################################

N = 10
f0 = 0.1
var_w = 0.01

#####################
# END OF PARAMETERS #
#####################

nf0 = 500
f0s = np.linspace(0, 0.5, nf0)
ns = np.arange(N)

# noise
np.random.seed(9)
w = np.random.normal(loc=0, scale=math.sqrt(var_w), size=(N, ))
# observed data
x = w + np.cos(2 * math.pi * f0 * ns)
sn = np.cos(2 * math.pi * np.outer(f0s, ns))
J = np.dot(sn, x)

# cálculo por busqueda de grilla y newton-raphson en M realizaciones
M = 500
# valor inicial del metodo de NR
f0i = [0.05, 0.12, 0.15, 0.2]
nsteps = 10
f0_grid = np.zeros((M, ))
f0_newt = np.zeros((M, len(f0i)))
for m in np.arange(M):
    w = np.random.normal(loc=0, scale=math.sqrt(var_w), size=(N,))
    x = w + np.cos(2 * math.pi * f0 * ns)
    # estimacion por busqueda de grilla
    J = np.dot(sn, x)
    f0_grid[m] = f0s[np.argmax(J)]
    # estimacion por metodo de NR
    for j in np.arange(len(f0i)):
        f0_newt[m, j] = newton_raphson(f0i[j], nsteps)

print("Newton-Raphson")
print("initial guess: " + str(f0i))
print("mean: " + str(np.mean(f0_newt, axis=0)))
print("variance" + str(np.var(f0_newt, axis=0)))

print("Grid search")
print("mean: " + str(np.mean(f0_grid)))
print("variance: " + str(np.var(f0_grid)))


# para plot con varias realizaciones
nrel = 10
Js = np.zeros((nf0, nrel))
for i in np.arange(nrel):
    w = np.random.normal(loc=0, scale=math.sqrt(var_w), size=(N, ))
    Js[:, i] = np.dot(np.cos(2 * math.pi * np.outer(f0s, ns)), w + np.cos(2 * math.pi * f0 * ns))

# abscissa values
xmin = 0
xmax = 0.5
ymin = -1.5
ymax = 6
# axis parameters
xmin_ax = xmin
xmax_ax = xmax + 0.04
# para la grafica de g(x)
ymax_ax = ymax + 0.5
ymin_ax = ymin


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.8
ytm = 0.007
# font size
fontsize = 12

# GRAFICAS
fig = plt.figure(0, figsize=(9, 3), frameon=False)
ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(f0s, Js)

plt.text(ytm, ymax_ax, '$J(f_0)$', fontsize=fontsize, ha='left', va='center')
plt.text(xmax_ax, xtm, '$f_0$', fontsize=fontsize, ha='center', va='baseline')

xts = np.arange(0.1, 0.55, 0.1)
for xt in xts:
    plt.plot([xt, xt], [0, xtl], 'k')
    plt.text(xt, xtm, '${:.1f}$'.format(xt), fontsize=fontsize, ha='center', va='baseline')

yts = np.arange(-1, 6, 1)
for yt in yts:
    plt.plot([xmin, xmin + ytl], [yt, yt], 'k')
    plt.text(xmin-ytm, yt, '${:d}$'.format(yt), fontsize=fontsize, ha='right', va='center')

plt.axis('off')

fontsize = 11
# save as pdf image
plt.savefig('problem_7_19_J.pdf', bbox_inches='tight')

# estimacion en cada realizacion
Ms = np.arange(M)

fig = plt.figure(1, figsize=(9, 6), frameon=False)

ax = plt.subplot2grid((10, 1), (0, 0), rowspan=2, colspan=1)
plt.xlim(0, M-1)
plt.plot(Ms, f0_grid, 'k', lw=1)
ax.set_xticklabels([])
plt.text(0.5, 1.05, '$\\rm{B\\acute{u}squeda\;de\;grilla:\; 500\;valores\;en\;}[0,\,1/2]$', fontsize=fontsize,
         ha='center', va='center', transform=ax.transAxes,
         bbox=dict(boxstyle="square", fc=(1, 1, 1), ec=(0, 0, 0)))

ax = plt.subplot2grid((10, 1), (2, 0), rowspan=2, colspan=1)
plt.xlim(0, M-1)
plt.plot(Ms, f0_newt[:, 0], 'k', lw=1)
ax.set_xticklabels([])
plt.text(0.5, 1.05, '$\mathrm{{Newton-Raphson\;con\;valor\;inicial\;}}f_0={:.2f}$'.format(f0i[0]), fontsize=fontsize,
         ha='center', va='center', transform=ax.transAxes,
         bbox=dict(boxstyle="square", fc=(1, 1, 1), ec=(0, 0, 0)))

ax = plt.subplot2grid((10, 1), (4, 0), rowspan=2, colspan=1)
plt.xlim(0, M-1)
plt.plot(Ms, f0_newt[:, 1], 'k', lw=1)
ax.set_xticklabels([])
plt.text(0.5, 1.05, '$\mathrm{{Newton-Raphson\;con\;valor\;inicial\;}}f_0={:.2f}$'.format(f0i[1]), fontsize=fontsize,
         ha='center', va='center', transform=ax.transAxes,
         bbox=dict(boxstyle="square", fc=(1, 1, 1), ec=(0, 0, 0)))

ax = plt.subplot2grid((10, 1), (6, 0), rowspan=2, colspan=1)
plt.xlim(0, M-1)
plt.plot(Ms, f0_newt[:, 2], 'k', lw=1)
ax.set_xticklabels([])
plt.text(0.5, 1.05, '$\mathrm{{Newton-Raphson\;con\;valor\;inicial\;}}f_0={:.2f}$'.format(f0i[2]), fontsize=fontsize,
         ha='center', va='center', transform=ax.transAxes,
         bbox=dict(boxstyle="square", fc=(1, 1, 1), ec=(0, 0, 0)))

ax = plt.subplot2grid((10, 1), (8, 0), rowspan=2, colspan=1)
plt.xlim(0, M-1)
plt.plot(Ms, f0_newt[:, 3], 'k', lw=1)
plt.text(0.5, 1.05, '$\mathrm{{Newton-Raphson\;con\;valor\;inicial\;}}f_0={:.2f}$'.format(f0i[3]), fontsize=fontsize,
         ha='center', va='center', transform=ax.transAxes,
         bbox=dict(boxstyle="square", fc=(1, 1, 1), ec=(0, 0, 0)))


# save as pdf image
plt.savefig('problem_7_19_estimations.pdf', bbox_inches='tight')

plt.show()
