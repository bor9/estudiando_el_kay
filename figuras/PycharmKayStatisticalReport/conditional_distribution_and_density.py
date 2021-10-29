import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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
    x_coord_len = data_coords[1, 0] - data_coords[0, 0]
    # transform from display coordinates to data coordinates in y axis
    data_coords = inv.transform([(0, 0), (0, length)])
    # get the length of the segment in data units
    y_coord_len = data_coords[1, 1] - data_coords[0, 1]
    return x_coord_len, y_coord_len


#####################################
# PARAMETERS - This can be modified #
#####################################

# normal pdf standard deviation
sigma = 1
# normal pdf mean
mu = 2.5
# F(x | x < a1)
a1 = mu - sigma/2
# F(x | b2 < x < a2)
b2 = mu - sigma/2
a2 = mu + 3*sigma/2

# maximum deviation from the mean where to plot each gaussian
max_mean_dev = 4 * sigma

#####################
# END OF PARAMETERS #
#####################

# abscissa values
x = np.linspace(mu-max_mean_dev, mu+max_mean_dev, 400)
# normal distribution and density values in x
norm_cdf = norm.cdf(x, mu, sigma)
norm_pdf = norm.pdf(x, mu, sigma)

# conditional distribution F(x | x < a1)
# index of the number in x closest to a1
idx_a1 = np.argmax(x > a1)
cond_cdf1 = np.ones(x.shape)
cond_cdf1[0:idx_a1] = norm_cdf[0:idx_a1]/norm_cdf[idx_a1]
cond_pdf1 = np.zeros(x.shape)
cond_pdf1[0:idx_a1] = norm_pdf[0:idx_a1]/norm_cdf[idx_a1]

# conditional distribution F(x | b2 < x < a2)
# index of the number in x closest to a2 y b2
idx_a2 = np.argmax(x > a2)
idx_b2 = np.argmax(x > b2)
cond_cdf2 = np.ones(x.shape)
cond_cdf2[0:idx_b2] = 0
cond_cdf2[idx_b2:idx_a2] = (norm_cdf[idx_b2:idx_a2] - norm_cdf[idx_b2])/(norm_cdf[idx_a2]-norm_cdf[idx_b2])
cond_pdf2 = np.zeros(x.shape)
cond_pdf2[idx_b2:idx_a2] = norm_pdf[idx_b2:idx_a2]/(norm_cdf[idx_a2]-norm_cdf[idx_b2])

print(np.sum(cond_pdf1)*(x[1]-x[0]))
print(np.sum(cond_pdf2)*(x[1]-x[0]))


# value of the pdf in 0 - maximum value of the normal pdf
pdf_max = cond_pdf1[idx_a1-1]

# axis parameters
dx = 0.5
xmin = mu - max_mean_dev - dx
xmax = mu + max_mean_dev + dx

ymax1 = 1.2
ymin1 = -0.1

ymax2 = pdf_max * 1.2
ymin2 = -pdf_max * 0.1

# vertical tick margin
vtm = -0.12
vtm2 = vtm * (ymax2-ymin2) / (ymax1-ymin1)
# horizontal tick margin
htm = -0.2
# font size
fontsize = 14
bggrey = 0.97
# dashes length/space
dashed = (4, 4)

# length of the ticks for all subplot (7 pixels)
display_length = 7  # in pixels

fig = plt.figure(0, figsize=(10, 6), frameon=False)

# PLOT OF F(x | x < a)
ax = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)

plt.xlim(xmin, xmax)
plt.ylim(ymin1, ymax1)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax1), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.plot(x, norm_cdf, 'k', linewidth=2)
plt.plot(x, cond_cdf1, 'r', linewidth=2)

# legend
leg = plt.legend(['$F(x)$', '$F(x|\mathbf{x}\leq a)$'], loc=(0.58, 0.15), fontsize=12)
leg.get_frame().set_facecolor(bggrey*np.ones((3,)))
leg.get_frame().set_edgecolor(bggrey*np.ones((3,)))
# xlabels and xtickslabels
plt.plot([a1, a1], [0, 1], 'k--', linewidth=0.8, dashes=dashed)
plt.plot([0, a1], [1, 1], 'k--', linewidth=0.8, dashes=dashed)
F_a1 = norm.cdf(a1, mu, sigma)
plt.plot([0, a1], [F_a1, F_a1], 'k--', linewidth=0.8, dashes=dashed)
plt.text(xmax, vtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(a1, vtm, '$a$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0, vtm, '$0$', fontsize=fontsize, ha='center', va='baseline')
# ylabels and ytickslabels
plt.text(htm, 1, '$1$', fontsize=fontsize, ha='right', va='center')
plt.text(htm, F_a1, '$F(a)$', fontsize=fontsize, ha='right', va='center')


plt.axis('off')

# PLOT OF f(x | x < a)
ax = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)

plt.xlim(xmin, xmax)
plt.ylim(ymin2, ymax2)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax2), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, norm_pdf, 'k', linewidth=2)
plt.plot(x, cond_pdf1, 'r', linewidth=2)

# legend
leg = plt.legend(['$f(x)$', '$f(x|\mathbf{x}\leq a)$'], loc=(0.58, 0.7), fontsize=12)
leg.get_frame().set_facecolor(bggrey*np.ones((3,)))
leg.get_frame().set_edgecolor(bggrey*np.ones((3,)))
# xticks
plt.plot(a1*np.ones((2,)), [0, vtl], 'k', linewidth=0.8)
# xlabels and xtickslabels
f_a1 = norm.pdf(a1, mu, sigma)
plt.plot([0, a1], [f_a1, f_a1], 'k--', linewidth=0.8, dashes=dashed)
plt.plot([0, a1], [f_a1/F_a1, f_a1/F_a1], 'k--', linewidth=0.8, dashes=dashed)
plt.text(xmax, vtm2, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(a1, vtm2, '$a$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0, vtm2, '$0$', fontsize=fontsize, ha='center', va='baseline')
plt.text(htm, f_a1, '$f(a)$', fontsize=fontsize, ha='right', va='center')
plt.text(htm, f_a1/F_a1, r'$\frac{f(a)}{F(a)}$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

# PLOT OF F(x | b < x < a)
ax = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)

plt.xlim(xmin, xmax)
plt.ylim(ymin1, ymax1)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax1), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.plot(x, norm_cdf, 'k', linewidth=2)
plt.plot(x, cond_cdf2, 'r', linewidth=2)

# legend
leg = plt.legend(['$F(x)$', '$F(x|b<\mathbf{x}\leq a)$'], loc=(0.55, 0.15), fontsize=12)
leg.get_frame().set_facecolor(bggrey*np.ones((3,)))
leg.get_frame().set_edgecolor(bggrey*np.ones((3,)))
# xlabels and xtickslabels
F_b2 = norm.cdf(b2, mu, sigma)
F_a2 = norm.cdf(a2, mu, sigma)
plt.plot([b2, b2], [0, F_b2], 'k--', linewidth=0.8, dashes=dashed)
plt.plot([a2, a2], [0, 1], 'k--', linewidth=0.8, dashes=dashed)
plt.plot([0, a2], [1, 1], 'k--', linewidth=0.8, dashes=dashed)
plt.plot([0, b2], [F_b2, F_b2], 'k--', linewidth=0.8, dashes=dashed)
plt.plot([0, a2], [F_a2, F_a2], 'k--', linewidth=0.8, dashes=dashed)
plt.text(xmax, vtm, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(b2, vtm, '$b$', fontsize=fontsize, ha='center', va='baseline')
plt.text(a2, vtm, '$a$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0, vtm, '$0$', fontsize=fontsize, ha='center', va='baseline')
# ylabels and ytickslabels
dy = 0.025
plt.text(htm, 1+dy, '$1$', fontsize=fontsize, ha='right', va='center')
plt.text(htm, F_a2-dy, '$F(a)$', fontsize=fontsize, ha='right', va='center')
plt.text(htm, F_b2, '$F(b)$', fontsize=fontsize, ha='right', va='center')


plt.axis('off')

# PLOT OF f(x | b < x < a)
ax = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)

plt.xlim(xmin, xmax)
plt.ylim(ymin2, ymax2)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax2), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(x, norm_pdf, 'k', linewidth=2)
plt.plot(x, cond_pdf2, 'r', linewidth=2)

# legend
leg = plt.legend(['$f(x)$', '$f(x|b<\mathbf{x}\leq a)$'], loc=(0.55, 0.7), fontsize=12)
leg.get_frame().set_facecolor(bggrey*np.ones((3,)))
leg.get_frame().set_edgecolor(bggrey*np.ones((3,)))
# xticks
plt.plot(b2*np.ones((2,)), [0, vtl], 'k', linewidth=0.8)
plt.plot(a2*np.ones((2,)), [0, vtl], 'k', linewidth=0.8)
# xlabels and xtickslabels
f_b2 = norm.pdf(b2, mu, sigma)
f_a2 = norm.pdf(a2, mu, sigma)
plt.text(xmax, vtm2, '$x$', fontsize=fontsize, ha='right', va='baseline')
plt.text(b2, vtm2, '$b$', fontsize=fontsize, ha='center', va='baseline')
plt.text(a2, vtm2, '$a$', fontsize=fontsize, ha='center', va='baseline')
plt.text(0, vtm2, '$0$', fontsize=fontsize, ha='center', va='baseline')
plt.plot([0, a2], [f_a2/(F_a2-F_b2), f_a2/(F_a2-F_b2)], 'k--', linewidth=0.8, dashes=dashed)
plt.text(htm, f_a2/(F_a2-F_b2), r'$\frac{f(a)}{F(a)-F(b)}$', fontsize=fontsize, ha='right', va='center')
plt.plot([0, b2], [f_b2/(F_a2-F_b2), f_b2/(F_a2-F_b2)], 'k--', linewidth=0.8, dashes=dashed)
plt.text(htm, f_b2/(F_a2-F_b2), r'$\frac{f(b)}{F(a)-F(b)}$', fontsize=fontsize, ha='right', va='center')

plt.axis('off')

# save as eps image
plt.savefig('conditional_distribition_and_density.pdf', bbox_inches='tight')
plt.show()


