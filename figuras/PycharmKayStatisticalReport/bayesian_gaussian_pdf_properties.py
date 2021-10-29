import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rc('mathtext', fontset='cm')

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

#####################################
# PARAMETERS - This can be modified #
#####################################

# media
mu_x = 0
mu_y = 0

# covarianza
rho = 0.5
var_x = 1
var_y = 1

x0 = 0.8

#####################
# END OF PARAMETERS #
#####################

# vetor de media
mu = [mu_x, mu_y]
# matriz de covarianza
C = [[var_x, rho], [rho, var_y]]

xmax = 3
ymax = xmax
x, y = np.mgrid[-xmax:xmax:.01, -ymax:ymax:.01]
pos = np.dstack((x, y))
rv = multivariate_normal(mu, C)

x0_idx = np. where(np.abs(x[:, 0]-x0) < 1e-5)
x0_idx =x0_idx[0][0]

pdf = rv.pdf(pos)
zmax = np.amax(pdf)
zmin = -0.18
nlevels = 16
levels = np.linspace(0.005, zmax, nlevels)

fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.contourf(x, y, pdf)
plt.axis('equal')

x_c = x[x0_idx, :]
y_c = y[x0_idx, :]
pos_c = np.dstack((x_c, y_c))
p_cond = rv.pdf(pos_c)

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(y_c, p_cond)


fontsize = 12

dx = 0.5
xmin_ax = -xmax-1
xmax_ax = xmax+dx
ymin_ax = -ymax-dx
ymax_ax = ymax+dx

fig = plt.figure(2, figsize=(10, 6), frameon=False)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(xmin_ax, xmax_ax)
ax.set_zlim(ymin_ax, ymax_ax)
ax.set_zlim(zmin, zmax)
ax.view_init(elev=31, azim=-61)
# axis arrows
arw = Arrow3D([xmin_ax, xmax_ax], [0, 0], [zmin, zmin], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=100)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [ymin_ax, ymax_ax], [zmin, zmin], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=100)
ax.add_artist(arw)

arw = Arrow3D([xmin_ax, xmin_ax], [ymin_ax, ymax_ax], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=100)
ax.add_artist(arw)
arw = Arrow3D([xmin_ax, xmin_ax], [0, 0], [-0.03, zmax], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=100)
ax.add_artist(arw)


plt.contour(x, y, pdf, cmap=cm.coolwarm, levels=levels, zorder=-1)
plt.contour(x, y, pdf, cmap=cm.coolwarm, offset=zmin, levels=levels, zorder=-1)

plt.plot(x_c, y_c, p_cond, 'k', zorder=100, lw=2)
plt.plot([-xmax, xmax], [-rho*xmax, rho*xmax], [zmin, zmin], 'k', zorder=100)
plt.plot([x0, x0], [-ymax, ymax], [zmin, zmin], 'k', zorder=100)

plt.plot([x0, x0], [rho*x0, rho*x0], [zmin, rv.pdf([x0, rho*x0])], 'k--', zorder=100, lw=1)
plt.plot(xmin_ax*np.ones(x_c.shape), y_c, p_cond, 'k', zorder=100, lw=2)
plt.plot([xmin_ax, x0], [rho*x0, rho*x0], [rv.pdf([x0, rho*x0]), rv.pdf([x0, rho*x0])], 'k--', zorder=100, lw=1)
plt.plot([xmin_ax, x0], [rho*x0, rho*x0], [rv.pdf([x0, rho*x0]), rv.pdf([x0, rho*x0])], 'k--', zorder=100, lw=1)
plt.plot([xmin_ax, xmin_ax], [rho*x0, rho*x0], [0, rv.pdf([x0, rho*x0])], 'k--', zorder=100, lw=1)

# axis labels
ax.text(xmax_ax, -0.55, zmin, '$x$', fontsize=fontsize, ha='center', va='baseline')
ax.text(-0.3, ymax_ax, zmin, '$y$', fontsize=fontsize, ha='right', va='center')
ax.text(xmin_ax, 0.15, zmax, '$p(y|x_0)$', fontsize=fontsize, ha='left', va='center')
ax.text(xmin_ax, ymax_ax, -0.025, '$y$', fontsize=fontsize, ha='right', va='center')
# lines label
ax.text(x0, ymin_ax+0.3, zmin, '$x_0$', fontsize=fontsize, ha='center', va='top')
ax.text(xmax, rho*xmax_ax, zmin, '$y=\\rho x$', fontsize=fontsize, ha='left', va='top')
#ax.text(xmin_ax, rho*x0, 0.01, '$\hat{y}=\\rho x_0$', fontsize=fontsize, ha='left', va='baseline', zdir=(1, 1, 0.037))
ax.text(xmin_ax, rho*x0, -0.025, '$\hat{y}=\\rho x_0$', fontsize=fontsize, ha='left', va='baseline')

# Distance view. Default is 10.
ax.dist = 8

plt.axis('off')
# save as pdf image
plt.savefig('bayesian_gaussian_pdf_properties.pdf', bbox_inches='tight')

plt.show()