import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import matplotlib.colors as colors

from matplotlib import cm
from matplotlib import rc

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


#####################################
# PARAMETERS - This can be modified #
#####################################

# normal pdf variances
var1 = 0.5
var2 = 2
var_std = 1
# normal pdf mean
theta = 6
epsilon = 1.5

# maximum deviation from the mean where to plot each gaussian
max_mean_dev = 3.1 * var2

#####################
# END OF PARAMETERS #
#####################

# abscissa values
xmin = theta - max_mean_dev
xmax = theta + max_mean_dev

x = np.linspace(xmin, xmax, 300)
# normal distribution and density values in x
pdf_var1 = norm.pdf(x, theta, math.sqrt(var1))
pdf_var2 = norm.pdf(x, theta, math.sqrt(var2))
pdf_std = norm.pdf(x, theta, math.sqrt(var_std))


# axis parameters
dx = xmax / 20
xmin_ax = xmin - dx
xmax_ax = xmax + dx

ym = np.amax(pdf_var1)
ymax_ax = ym + ym / 10
ymin_ax = -ym / 10

# areas to fill limits
pdf1_xinf = np.linspace(xmin, theta-epsilon, 50)
pdf1_inf = norm.pdf(pdf1_xinf, theta, math.sqrt(var1))
pdf1_xsup = np.linspace(theta+epsilon, xmax, 50)
pdf1_sup = norm.pdf(pdf1_xsup, theta, math.sqrt(var1))
pdf2_xinf = np.linspace(xmin, theta-epsilon, 50)
pdf2_inf = norm.pdf(pdf2_xinf, theta, math.sqrt(var2))
pdf2_xsup = np.linspace(theta+epsilon, xmax, 50)
pdf2_sup = norm.pdf(pdf2_xsup, theta, math.sqrt(var2))

epsilon1 = epsilon / math.sqrt(var1)
epsilon2 = epsilon / math.sqrt(var2)
pdfstd1_xinf = np.linspace(xmin, theta-epsilon1, 50)
pdfstd1_inf = norm.pdf(pdfstd1_xinf, theta, math.sqrt(var_std))
pdfstd1_xsup = np.linspace(theta+epsilon1, xmax, 50)
pdfstd1_sup = norm.pdf(pdfstd1_xsup, theta, math.sqrt(var_std))
pdfstd2_xinf = np.linspace(xmin, theta-epsilon2, 50)
pdfstd2_inf = norm.pdf(pdfstd2_xinf, theta, math.sqrt(var_std))
pdfstd2_xsup = np.linspace(theta+epsilon2, xmax, 50)
pdfstd2_sup = norm.pdf(pdfstd2_xsup, theta, math.sqrt(var_std))


# length of the ticks for all subplot (6 pixels)
display_length = 6  # in pixels
# x ticks labels margin
xtm = -0.09
ytm = 0.4
# font size
fontsize = 14
# colors from coolwarm
cNorm = colors.Normalize(vmin=0, vmax=1)
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.coolwarm)
col10 = scalarMap.to_rgba(0)
col20 = scalarMap.to_rgba(1)

fig = plt.figure(0, figsize=(10, 6), frameon=False)

# PLOT OF F(x | x < a)
ax = plt.subplot2grid((2, 8), (0, 0), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, pdf_var1, color='k', linewidth=2)

# filled areas
ax.fill_between(pdf1_xinf, 0, pdf1_inf, color=col10)
ax.fill_between(pdf1_xsup, 0, pdf1_sup, color=col10)

# xlabels and xtickslabels
plt.plot([theta, theta], [0, xtl], 'k')
plt.plot([theta-epsilon, theta-epsilon], [0, xtl], 'k')
plt.plot([theta+epsilon, theta+epsilon], [0, xtl], 'k')
plt.text(theta, xtm, '$\\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta-epsilon, xtm, '$\\theta-\epsilon$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta+epsilon, xtm, '$\\theta+\epsilon$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, '$\hat{\\theta}$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, '$p(\hat{\\theta})=\mathcal{N}(\\theta,\,\sigma^2_{\hat{\\theta}})$',
         fontsize=fontsize, ha='left', va='center')


plt.text(xmax_ax+0.4, ymax_ax, '$\sigma^2_{\hat{\\theta}}<\sigma^2_{\check{\\theta}}$',
         fontsize=fontsize, ha='center', va='center')

plt.axis('off')


##
ax = plt.subplot2grid((2, 8), (0, 4), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(theta, ymin_ax), xycoords='data', xy=(theta, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, pdf_std, color='k', linewidth=2)

# filled areas
ax.fill_between(pdfstd1_xinf, 0, pdfstd1_inf, color=col10)
ax.fill_between(pdfstd1_xsup, 0, pdfstd1_sup, color=col10)

xtm2 = -0.11
# xlabels and xtickslabels
plt.plot([theta-epsilon1, theta-epsilon1], [0, xtl], 'k')
plt.plot([theta+epsilon1, theta+epsilon1], [0, xtl], 'k')
# plt.text(theta-epsilon1, xtm, '$$-\epsilon/\sqrt{\\textrm{var}(\hat{\\theta})}$$',
#          fontsize=fontsize, ha='center', va='baseline')
# plt.text(theta-epsilon1, xtm, '$$-\\frac{\epsilon}{\sigma_{\hat{\\theta}}}$$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta-epsilon1, xtm2, '$-\epsilon/\sigma_{\hat{\\theta}}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta+epsilon1, xtm2, '$\epsilon/\sigma_{\hat{\\theta}}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm2, '$(\hat{\\theta}-\\theta$)/\sigma_{\hat{\\theta}}',
         fontsize=fontsize, ha='center', va='baseline')
plt.text(theta + ytm, ymax_ax, '$p((\hat{\\theta}-\\theta$)/\sigma_{\hat{\\theta}})=\mathcal{N}(0,\,1)$',
         fontsize=fontsize, ha='left', va='center')
plt.axis('off')


#########################
#########################

ax = plt.subplot2grid((2, 8), (1, 0), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, pdf_var2, color='k', linewidth=2)

# filled areas
ax.fill_between(pdf2_xinf, 0, pdf2_inf, color=col10)
ax.fill_between(pdf2_xsup, 0, pdf2_sup, color=col10)

# xlabels and xtickslabels
plt.plot([theta, theta], [0, xtl], 'k')
plt.plot([theta-epsilon, theta-epsilon], [0, xtl], 'k')
plt.plot([theta+epsilon, theta+epsilon], [0, xtl], 'k')
plt.text(theta, xtm, '$\\theta$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta-epsilon, xtm, '$\\theta-\epsilon$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta+epsilon, xtm, '$\\theta+\epsilon$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm, '$\check{\\theta}$', fontsize=fontsize, ha='right', va='baseline')
plt.text(ytm, ymax_ax, '$p(\check{\\theta})=\mathcal{N}(\\theta,\,\sigma^2_{\check{\\theta}})$',
         fontsize=fontsize, ha='left', va='center')
plt.axis('off')

##
ax = plt.subplot2grid((2, 8), (1, 4), rowspan=1, colspan=4)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# axis arrows
plt.annotate("", xytext=(xmin_ax, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(theta, ymin_ax), xycoords='data', xy=(theta, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, pdf_std, color='k', linewidth=2)

# filled areas
ax.fill_between(pdfstd2_xinf, 0, pdfstd2_inf, color=col10)
ax.fill_between(pdfstd2_xsup, 0, pdfstd2_sup, color=col10)

xtm2 = -0.11
# xlabels and xtickslabels
plt.plot([theta-epsilon2, theta-epsilon2], [0, xtl], 'k')
plt.plot([theta+epsilon2, theta+epsilon2], [0, xtl], 'k')
plt.text(theta-epsilon2, xtm2, '$-\epsilon/\sigma_{\check{\\theta}}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(theta+epsilon2, xtm2, '$\epsilon/\sigma_{\check{\\theta}}$', fontsize=fontsize, ha='center', va='baseline')
plt.text(xmax_ax, xtm2, '$(\check{\\theta}-\\theta$)/\sigma_{\check{\\theta}}',
         fontsize=fontsize, ha='center', va='baseline')
plt.text(theta + ytm, ymax_ax, '$p((\check{\\theta}-\\theta$)/\sigma_{\check{\\theta}})=\mathcal{N}(0,\,1)$',
         fontsize=fontsize, ha='left', va='center')
plt.axis('off')


# save as pdf image
plt.savefig('problem_2_7.pdf', bbox_inches='tight')

plt.show()

