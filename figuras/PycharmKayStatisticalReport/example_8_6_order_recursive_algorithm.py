import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv, norm
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib import rc

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


##########################################
# PARAMETROS - Esto puede ser modificado #
##########################################

# coeficientes del polinomio - a estimar
A = 1
B = 0.03
# numero de muestras
N = 100
# varianza del ruido
var_w = 0.1

#####################
# FIN DE PARAMETROS #
#####################
# abscissa values
xmin = 0
xmax = 100
ymin = 0
ymax = 5

n = np.arange(N)
s = A + B * n
x = s + np.random.normal(0, math.sqrt(var_w), N)

# calculo de los estimadores
A1_est = np.mean(x)
A2_est = 2*(2*N-1)/(N*(N+1)) * np.sum(x) - 6/(N*(N+1)) * np.sum(n*x)
B2_est = - 6/(N*(N+1)) * np.sum(x) + 12/(N*(N**2-1)) * np.sum(n*x)

s1_est = np.ones((N,)) * A1_est
s2_est = A2_est + B2_est * n

#################################
# order recursive least squares #
#################################
kmax = 5  # orden maximo
x = x.reshape(-1, 1)
Jmins = np.zeros((kmax,))
thetas = np.zeros((kmax, kmax))
s_est = np.zeros((N, kmax))

# vector con el número de muestra: n = 0, ..., N-1
n = np.arange(N).reshape(-1, 1)
k = 1  # el número de paso es k + 1
# matriz de observación H_1
H = np.power(n, k - 1)
# matriz D_1
D = inv(H.T@H)
# estimador theta_1
theta_est = D @ (H.T @ x)
Jmin = norm(x - H @ theta_est, ord=2) ** 2
Jmins[k - 1] = Jmin
thetas[k - 1, k - 1] = theta_est
s_est[:, k-1] = H @ theta_est[:, 0]
for k in np.arange(2, kmax + 1):
    # cálculo de P_perp en el paso k - 1
    P_perp = np.identity(N) - H @ D @ H.T
    # columna a agregrar en H en el paso k: vector con los elementos del vector n elevados a la k - 1
    h = np.power(n, k - 1)
    # denominador de los elementos del vector theta y D en el paso k
    den = h.T @ P_perp @ h
    ## cálculo del estimador en el paso k
    # elemento theta_12 del estimador: dimensiones (k - 1) x 1
    theta_est11 = theta_est - (D @ H.T @ h @ h.T @ P_perp @ x) / den
    # elemento theta_22 del estimador: dimensiones 1 x 1
    theta_est12 = (h.T @ P_perp @ x) / den
    # concatenación en filas para formar el estimador theta_est en el paso k
    theta_est = np.concatenate((theta_est11, theta_est12), axis=0)
    ## construccion de la matrix D en el paso k
    D_11 = D + (D @ H.T @ h @ h.T @ H @ D) / den
    D_21 = - (D @ H.T @ h) / den
    # concatenación para formar D
    D = np.concatenate((np.concatenate((D_11, D_21), axis=1), np.concatenate((D_21.T, 1 / den), axis=1)), axis=0)
    # construccion de H en el paso k: [H h]
    H = np.concatenate((H, h), axis=1)
    ## calculo del error mínimo en el paso k
    Jmin = Jmin - (h.T @ P_perp @ x) ** 2 / den
    # se guardan los resultados
    Jmins[k-1] = Jmin
    thetas[0:k, k-1] = theta_est[:, 0]
    s_est[:, k-1] = H @ theta_est[:, 0]

print(Jmins)


# axis parameters
dx = 3
xmin_ax = xmin - dx
xmax_ax = xmax + 6

dy = 0.4
ymax_ax = ymax
ymin_ax = ymin - dy


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.5
ytm = -2
# font size
fontsize = 12

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(9, 3), frameon=False)

# PLOT OF P_ww
ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(n, x, 'k.', markersize=5, label='$x[n]$')
plt.plot(n, s1_est, color=col10, linewidth=2, label='$\hat{s}_1[n]=\hat{A}_1$')
plt.plot(n, s2_est, color=col20, linewidth=2, label='$\hat{s}_2[n]=\hat{A}_2+\hat{B}_2n$')

# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$n$', fontsize=fontsize, ha='center', va='baseline')
plt.text(ytm, xtm, '$0$', fontsize=fontsize, ha='right', va='baseline')
for i in np.arange(10, xmax+1, 10):
    plt.plot([i, i], [0, xtl], 'k')
    plt.text(i, xtm, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(1, ymax, 1):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ytm, i, '${}$'.format(i), fontsize=fontsize, ha='right', va='center')

# legend
leg = plt.legend(loc=(0.08, 0.65), fontsize=12, frameon=False)

plt.axis('off')

# save as pdf image
# plt.savefig('example_8_6_data.pdf', bbox_inches='tight')

###########################################
###########################################

# x and y ticks labels margin
xtm = -7
ytm = -0.2

# axis parameters
dx = 0.1
xmin_ax = 1
xmax_ax = kmax + 0.5

dy = 0.4
ymax_ax = 100
ymin_ax = 0 - dy

ks = np.arange(1, kmax+1)

fig = plt.figure(1, figsize=(4, 4), frameon=False)

# PLOT OF P_ww
ax = fig.add_subplot(111)

plt.xlim(xmin_ax-0.1, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(1, ymin_ax), xycoords='data', xy=(1, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(ks, Jmins, linestyle='-', marker='s', color='k', markersize=5)


# xlabels and xtickslabels
plt.text(xmax_ax, xtm, '$k$', fontsize=fontsize, ha='center', va='baseline')
plt.text(1-ytm, ymax_ax-2, '$J_{\\rm min}$', fontsize=fontsize, ha='left', va='center')
for i in np.arange(1, kmax + 1, 1):
    plt.plot([i, i], [0, xtl], 'k')
    plt.text(i, xtm, '${}$'.format(i), fontsize=fontsize, ha='center', va='baseline')

for i in np.arange(0, ymax_ax, 20):
    plt.plot([1, 1+ytl], [i, i], 'k')
    plt.text(1+ytm, i, '${}$'.format(i), fontsize=fontsize, ha='right', va='center')

# legend

plt.axis('off')

# save as pdf image
# plt.savefig('example_8_6_jmin.pdf', bbox_inches='tight')

plt.show()

