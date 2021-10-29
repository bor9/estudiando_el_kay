import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors
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


xmax = 6
ymax = 2.5
zmin = 0
zmax = 1

fontsize1 = 10
fontsize2 = 12

dx = 0.5
xmin_ax = -1.5
xmax_ax = xmax+dx
ymin_ax = -dx
ymax_ax = ymax+dx

# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col11 = scalarMap.to_rgba(0)
col12 = scalarMap.to_rgba(0.15)
col21 = scalarMap.to_rgba(1)
col22 = scalarMap.to_rgba(0.9)


fig = plt.figure(0, figsize=(10, 6), frameon=False)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(xmin_ax, xmax_ax)
ax.set_ylim(ymin_ax, ymax_ax)
ax.set_zlim(zmin, zmax)
ax.view_init(elev=44, azim=-50)
# axis arrows
arw = Arrow3D([-0.5, xmax_ax], [0, 0], [zmin, zmin], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=0)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [-0.25, ymax_ax], [zmin, zmin], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=0)
ax.add_artist(arw)
arw = Arrow3D([0, 0], [0, 0], [-0.03, zmax], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=00)
ax.add_artist(arw)


arw = Arrow3D([xmin_ax, xmin_ax], [-0.25, ymax_ax], [0, 0], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=100)
ax.add_artist(arw)
arw = Arrow3D([xmin_ax, xmin_ax], [0, 0], [-0.03, zmax], arrowstyle="-|>, head_width=0.5, head_length=1",
              lw=1, mutation_scale=5, facecolor='black', zorder=100)
ax.add_artist(arw)

# ax.add_collection3d(Poly3DCollection(verts))

# Density surface
z1 = 1/3
R1 = [[2, 0, z1], [3, 0, z1], [3, 1, z1], [2, 1, z1]]
z2 = 1/6
R2 = [[0, 1, z2], [2, 1, z2], [2, 2, z2], [0, 2, z2]]
R3 = [[3, 1, z2], [5, 1, z2], [5, 2, z2], [3, 2, z2]]

coll = Poly3DCollection([R1], facecolors=[col11], edgecolors=['k'], zorder=10)
fig.gca().add_collection(coll)
coll = Poly3DCollection([R2, R3], facecolors=[col21, col21], edgecolors=['k', 'k'], zorder=9)
fig.gca().add_collection(coll)

z3 = 0
R4 = [[2, 0, z3], [3, 0, z3], [3, 1, z3], [2, 1, z3]]
R5 = [[0, 1, z3], [2, 1, z3], [2, 2, z3], [0, 2, z3]]
R6 = [[3, 1, z3], [5, 1, z3], [5, 2, z3], [3, 2, z3]]

coll = Poly3DCollection([R4, R5, R6], facecolors=[col12, col22, col22], edgecolors=['k', 'k', 'k'], zorder=1)
fig.gca().add_collection(coll)

plt.plot([2, 2], [0, 0], [0, z1], 'k--', lw=1)
plt.plot([3, 3], [0, 0], [0, z1], 'k--', lw=1)
plt.plot([2, 2], [1, 1], [0, z1], 'k--', lw=1, zorder=1)
plt.plot([3, 3], [1, 1], [0, z1], 'k--', lw=1)

plt.plot([5, 5], [1, 1], [0, z2], 'k--', lw=1, zorder=0)
plt.plot([5, 5], [2, 2], [0, z2], 'k--', lw=1, zorder=0)
plt.plot([3, 3], [2, 2], [0, z2], 'k--', lw=1, zorder=0)

plt.plot([2, 2], [2, 2], [0, z2], 'k--', lw=1, zorder=0)
plt.plot([0, 0], [1, 1], [0, z2], 'k--', lw=1, zorder=0)
plt.plot([0, 0], [2, 2], [0, z2], 'k--', lw=1, zorder=0)

plt.plot([0, 2], [0, 0], [1/3, 1/3], 'k--', lw=1, zorder=0)
plt.plot([0, 0], [0, 1], [1/6, 1/6], 'k--', lw=1, zorder=0)
ax.text(0, -0.1, 1/3, '$\\frac{1}{3}$', fontsize=fontsize2, ha='right', va='center')
ax.text(0, -0.1, 1/6, '$\\frac{1}{6}$', fontsize=fontsize2, ha='right', va='center')


# PDF condicional
plt.plot([xmin_ax, xmin_ax], [0, 1], [1/3, 1/3], 'k', lw=2)
plt.plot([xmin_ax, xmin_ax], [1, 2], [2/3, 2/3], 'k', lw=2)
plt.plot([xmin_ax, 0], [1, 1], [zmin, zmin], 'k--', lw=1)
plt.plot([xmin_ax, 0], [2, 2], [zmin, zmin], 'k--', lw=1, zorder=0)
plt.plot([xmin_ax, xmin_ax], [1, 1], [zmin, 2/3], 'k--', lw=1)
plt.plot([xmin_ax, xmin_ax], [2, 2], [zmin, 2/3], 'k--', lw=1)

xtl = 0.07
for i in np.arange(1, 6):
    plt.plot([i, i], [0, xtl], [zmin, zmin], 'k', lw=1)
    ax.text(i + 0.2, -0.25, zmin, '${}$'.format(i), fontsize=fontsize1, ha='center', va='baseline')

# axis labels
ax.text(xmax_ax, -0.25, zmin, '$\\theta_1$', fontsize=fontsize2, ha='center', va='baseline')
ax.text(-0.35, ymax_ax, zmin, '$\\theta_2$', fontsize=fontsize2, ha='right', va='center')
ax.text(0, -0.15, zmax+0.05, '$p(\\theta_1,\,\\theta_2|\mathbf{x})$', fontsize=fontsize2, ha='center', va='center')

ax.text(xmin_ax + 0.4, ymax_ax, zmin, '$\\theta_2$', fontsize=fontsize2, ha='right', va='center')
ax.text(xmin_ax, -0.35, zmax, '$p(\\theta_2|\mathbf{x})$', fontsize=fontsize2, ha='center', va='center')
plt.plot([xmin_ax, xmin_ax], [0, 1], [2/3, 2/3], 'k--', lw=1)

ax.text(xmin_ax, -0.1, 1/3, '$\\frac{1}{3}$', fontsize=fontsize2, ha='right', va='center')
ax.text(xmin_ax, -0.1, 2/3, '$\\frac{2}{3}$', fontsize=fontsize2, ha='right', va='center')

ax.text(xmin_ax, 0+0.12, zmin+0.03, '$0$', fontsize=fontsize1, ha='center', va='baseline')
ax.text(xmin_ax, 1+0.12, zmin+0.03, '$1$', fontsize=fontsize1, ha='center', va='baseline')
ax.text(xmin_ax, 2+0.12, zmin+0.03, '$2$', fontsize=fontsize1, ha='center', va='baseline')


# Distance view. Default is 10.
ax.dist = 9

plt.axis('off')

# save as pdf image
plt.savefig('general_bayesian_map_vector.pdf', bbox_inches='tight')

plt.show()