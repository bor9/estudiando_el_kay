import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.colors as colors

from matplotlib import rc
from matplotlib import cm

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')

#####################################
# PARAMETERS - This can be modified #
#####################################

R_dev = 100  # range deviation
R_dev_prob = 0.99  # range deviation probability
c = 3 * 10e8  # electromagnetic propagation velocity

#####################
# END OF PARAMETERS #
#####################

z_0 = norm.ppf((1 + R_dev_prob) / 2)
st_dev_tau_0 = 2 * R_dev / (z_0 * c)
st_dev_R = c * st_dev_tau_0 / 2


# axis parameters
xmin = -R_dev - R_dev / 2
xmax = R_dev + R_dev / 2
ymax_value = norm.pdf(0, loc=0, scale=st_dev_R)
ymin = -0.15 * ymax_value
ymax = 1.2 * ymax_value

dx = R_dev / 10
xmin_ax = xmin - dx
xmax_ax = xmax + dx
ymin_ax = ymin
ymax_ax = ymax

# abscissa values
x = np.linspace(xmin, xmax, 500)

# normal distribution and density values in x
pdf = norm.pdf(x, loc=0, scale=st_dev_R)
x_R = np.linspace(-R_dev, R_dev, 400)
x_R_prob = norm.pdf(x_R, loc=0, scale=st_dev_R)

###############
#    PLOTS    #
###############
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col11 = scalarMap.to_rgba(0.2)
col12 = scalarMap.to_rgba(0.4)
col20 = scalarMap.to_rgba(1)
col21 = scalarMap.to_rgba(0.85)
col22 = scalarMap.to_rgba(0.7)

fontsize = 14
# vertical tick margin
vtm = -0.0015
# horizontal tick margin
htm = 10


fig = plt.figure(0, figsize=(5, 3), frameon=False)

ax = fig.add_subplot(111)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=5, headlength=8, facecolor='black', shrink=0.002))
plt.plot(x, pdf, color=col20)

# area
ax.fill_between(x_R, 0, x_R_prob, color=col12)
# area limits
plt.plot(-R_dev * np.ones(2,), [0, x_R_prob[0]], color=col10)
plt.plot(R_dev * np.ones(2,), [0, x_R_prob[0]], color=col10)
plt.plot(x_R, x_R_prob, color=col10)

plt.text(xmax_ax, vtm, '$\hat{R}-R$', fontsize=fontsize, ha='center', va='baseline')
plt.text(R_dev, vtm, '$100$', fontsize=fontsize, ha='center', va='baseline')
plt.text(-R_dev, vtm, '$-100$', fontsize=fontsize, ha='center', va='baseline')
plt.text(4, vtm, '$0$', fontsize=fontsize, ha='left', va='baseline')
plt.text(8, ymax_ax, '$p(\hat{R}-R)$', fontsize=fontsize, ha='left', va='center')
plt.text(0, 0.004, 'Área\n0.99', fontsize=fontsize, ha='center', va='center', color=col10,
         backgroundcolor=col12)


plt.axis('off')

# save as pdf image
plt.savefig('problem_1_1.pdf', bbox_inches='tight')
plt.show()
