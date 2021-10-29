import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# Parámetros
x1 = [1, 1, 2]
x2 = [1, 0, 1]
x3 = [1, 1, 1]
e1 = [1/math.sqrt(6), 1/math.sqrt(6), 2/math.sqrt(6)]
e2 = [1/math.sqrt(2), -1/math.sqrt(2), 0]
e3 = [1/math.sqrt(3), 1/math.sqrt(3), -1/math.sqrt(3)]

xmin = -1
xmax = 2
ymin = -1
ymax = 2
zmin = -1
zmax = 2

fontsize1 = 10
fontsize2 = 12

d_ax = 0.5
xmin_ax = xmin - d_ax
xmax_ax = xmax + d_ax
ymin_ax = ymin - d_ax
ymax_ax = ymax + d_ax
zmin_ax = zmin - d_ax
zmax_ax = zmax + d_ax


# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col1 = scalarMap.to_rgba(0)
col2 = scalarMap.to_rgba(1)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

da = 0.05
arw = Arrow3D([-da, xmax_ax], [0, 0], [0, 0], arrowstyle="-|>, head_width=0.3, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=0)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [-da, ymax_ax], [0, 0], arrowstyle="-|>, head_width=0.3, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=0)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [0, 0], [-da, zmax_ax], arrowstyle="-|>, head_width=0.3, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=00)
ax.add_artist(arw)
plt.plot([xmin_ax, xmax_ax], [0, 0], [0, 0], 'k-', lw=0.5, alpha=0)
plt.plot([0, 0], [ymin_ax, ymax_ax], [0, 0], 'k-', lw=0.5, alpha=0)
plt.plot([0, 0], [0, 0], [zmin_ax, zmax_ax], 'k-', lw=0.5, alpha=0)

arw = Arrow3D([0, x1[0]], [0, x1[1]], [0, x1[2]], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1.5, mutation_scale=5, color=col1, zorder=00)
ax.add_artist(arw)
arw = Arrow3D([0, x2[0]], [0, x2[1]], [0, x2[2]], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1.5, mutation_scale=5, color=col1, zorder=00)
ax.add_artist(arw)
arw = Arrow3D([0, x3[0]], [0, x3[1]], [0, x3[2]], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1.5, mutation_scale=5, color=col1, zorder=00)
ax.add_artist(arw)

arw = Arrow3D([0, e1[0]], [0, e1[1]], [0, e1[2]], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1.5, mutation_scale=5, color=col2, zorder=00)
ax.add_artist(arw)
arw = Arrow3D([0, e2[0]], [0, e2[1]], [0, e2[2]], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1.5, mutation_scale=5, color=col2, zorder=00)
ax.add_artist(arw)
arw = Arrow3D([0, e3[0]], [0, e3[1]], [0, e3[2]], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1.5, mutation_scale=5, color=col2, zorder=00)
ax.add_artist(arw)



set_axes_equal(ax)
plt.show()