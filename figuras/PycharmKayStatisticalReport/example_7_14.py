import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rc
from scipy import signal
from matplotlib import cm

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
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


#####################################
# PARAMETERS - This can be modified #
#####################################

# numero de muestras de los datos
N = 150
# parámetros del proceso MA a estimar
# ceros del filtro generador - par de complejos conjugados
mod_z = 0.9
arg_z = math.pi / 3
# numero de muestras en el dominio de la frecuencia
nf = 512

#####################
# END OF PARAMETERS #
#####################

# coeficientes del filtro generador
b1 = -2 * mod_z * math.cos(arg_z)
b2 = mod_z ** 2

# respuesta al impulso del filtro generador
h = [1, b1, b2]

# WGN
#np.random.seed(6)
np.random.seed(17)
w = np.random.normal(loc=0.0, scale=1.0, size=N)
# proceso MA(2)
x = signal.lfilter(h, 1, w)

# I(f)
If = np.square(np.absolute(np.fft.fft(x, n=2*nf))) / N
If = If[:nf]

# grilla de los complejos z1 y z2
nmod = 100
narg = 200
mod = np.linspace(0, 1, nmod)
arg = np.linspace(0, math.pi, narg)

# vector de frecuencias
f = np.arange(nf)/(2 * nf)
# inicialización de la matriz con la función de verosimilitud
likelihood = np.zeros((nmod, narg))
for i in np.arange(nmod):
    for j in np.arange(narg):
        omega, H1 = signal.freqz(b=[1, -2 * mod[i] * math.cos(arg[j]), mod[i] ** 2], a=1, worN=nf, whole=False,
                                 plot=None)
        likelihood[i, j] = np.trapz(If / np.square(np.absolute(H1)), f)

# indice del mínimo de la matriz en la función de verosimilitud
i, j = np.unravel_index(likelihood.argmin(), likelihood.shape)
# coeficientes estimados
b1_est = - 2 * mod[i] * math.cos(arg[j])
b2_est = mod[i] ** 2
# impresión de los resultados
print("i: {}".format(i))
print("j: {}".format(j))
print("|z|: {}".format(mod[i]))
print("Arg(z): {}".format(arg[j]))
print("b1_est: {}".format(b1_est))
print("b2_est: {}".format(b2_est))
print("b1: {}".format(b1))
print("b2: {}".format(b2))

# Gráfica de la superficie

zmin = -0.3
zmax = 2
likelihood_mod = likelihood
#likelihood_mod[likelihood > zmax] = np.nan

# niveles de la curva de nivel
levels = np.logspace(np.log10(likelihood[i, j]), np.log10(zmax), 20)

fig = plt.figure(0, figsize=(10, 6), frameon=False)
ax = fig.add_subplot(111, projection='3d')

# creación de la grilla en coordenadas polares
R, P = np.meshgrid(mod, arg)
# se convierte la grilla a coordenadas cartesianas
X, Y = R*np.cos(P), R*np.sin(P)
xx, yy = mod[i] * np.cos(arg[j]), mod[i] * np.sin(arg[j])


# grafica de la superficie como una curva de nivel
plt.contour(X, Y, np.transpose(likelihood_mod), cmap=cm.coolwarm,  levels=levels)
plt.contour(X, Y, np.transpose(likelihood_mod), cmap=cm.coolwarm, offset=zmin, levels=levels, zorder=0)
plt.plot([xx], [yy], [likelihood_mod[i, j]], 'k.', markersize=7)

# grafica del circulo unidad
plt.plot(np.cos(arg), np.sin(arg), zmin * np.ones(arg.shape), 'k')
plt.plot([xx], [yy], [zmin], 'k.', markersize=7)
plt.plot([mod_z*math.cos(arg_z)], [mod_z*math.sin(arg_z)], [zmin], 'r.', markersize=7)
plt.plot([xx, xx], [yy, yy], [zmin, likelihood_mod[i, j]], 'k--', linewidth=1)
plt.plot([0, xx], [0, yy], [zmin, zmin], 'k', zorder=10, linewidth=1)


ax.set_zlim(zmin, zmax)
ax.view_init(elev=33, azim=-103)

fontsize = 12
xticks = np.arange(-1, 1.5, 0.5)
plt.xticks(xticks, [])
for t in xticks:
    ax.text(t, -0.28, zmin, '${:.1f}$'.format(t), fontsize=fontsize, ha='right', va='baseline')
yticks = np.arange(0, 1.5, 0.5)
plt.yticks(yticks, [])
for t in yticks:
    ax.text(-1.22, t, zmin, '${:.1f}$'.format(t), fontsize=fontsize, ha='right', va='top')
zticks = np.arange(0, 2.5, 0.5)
ax.set_zticks(zticks)
ax.set_zticklabels([])
for t in zticks:
    ax.text(-1.22, 1.1, t-0.05, '${:.1f}$'.format(t), fontsize=fontsize, ha='right', va='center')


ax.set_xlabel(r'$\mathrm{Re}(z_1)$', fontsize=fontsize)
ax.set_ylabel(r'$\mathrm{Im}(z_1)$', fontsize=fontsize)

plt.savefig('example_7_14.pdf', bbox_inches='tight')

plt.show()