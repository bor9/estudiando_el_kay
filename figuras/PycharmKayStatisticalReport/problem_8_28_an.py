import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')

# construccion de a[m]=s*m+t(recta)
a0 = 1
ap = 0.2
p = 5

t = a0
s = (ap - t) / p

m = np.arange(p)
a = s * m + t

### Respuesta la impulso del filtro a identificar

nmax = p + 3
nmin = -nmax
n0 = nmax

n = np.arange(nmin, nmax + 1)
a1 = np.zeros(n.shape)
a1[n0: n0 + p] = a
a2 = np.zeros(n.shape)
a2[n0 - p + 1: n0 + 1] = a[::-1]
a3 = np.zeros(n.shape)
a3[n0: n0 + p] = a[::-1]
a4 = np.zeros(n.shape)
j = 3
a4[n0 + j: n0 + p + j] = a[::-1]


ymin = 0
ymax = np.amax(a1)

delta_n = 1
nmin_ax = nmin - delta_n
nmax_ax = nmax + 2 * delta_n
delta_y = 0.6
ymax_ax = ymax + delta_y
ymin_ax = ymin - 0.1



baseline = -0.4
fontsize1 = 12
fontsize2 = 14
y_sep = 0.6


fig = plt.figure(1, figsize=(10, 1.2), frameon=False)
ax = plt.subplot2grid((1, 16), (0, 0), rowspan=1, colspan=4)
plt.xlim(nmin_ax, nmax_ax)
plt.ylim(ymin_ax, ymax_ax)
plt.annotate("", xytext=(nmin_ax, 0), xycoords='data', xy=(nmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(n, a1, linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=4)
plt.setp(bl, visible=False)
plt.text(nmax_ax, baseline, '$k$', fontsize=fontsize2, ha='center', va='baseline')
i = p-1
plt.text(i, baseline, '$p$', fontsize=fontsize1, ha='center', va='baseline')
i = 0
plt.text(i, baseline, '${}$'.format(i), fontsize=fontsize1, ha='center', va='baseline')
plt.text(y_sep, ymax_ax, '$a[k]$', fontsize=fontsize2, ha='left', va='center')
plt.axis('off')

ax = plt.subplot2grid((1, 16), (0, 4), rowspan=1, colspan=4)
plt.xlim(nmin_ax, nmax_ax)
plt.ylim(ymin_ax, ymax_ax)
plt.annotate("", xytext=(nmin_ax, 0), xycoords='data', xy=(nmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(n, a2, linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=4)
plt.setp(bl, visible=False)
plt.text(nmax_ax, baseline, '$k$', fontsize=fontsize2, ha='center', va='baseline')
i = -p+1
plt.text(i, baseline, '$-p$', fontsize=fontsize1, ha='center', va='baseline')
i = 0
plt.text(i, baseline, '${}$'.format(i), fontsize=fontsize1, ha='center', va='baseline')
plt.text(y_sep, ymax_ax, '$a[-k]$', fontsize=fontsize2, ha='left', va='center')
plt.axis('off')

ax = plt.subplot2grid((1, 16), (0, 8), rowspan=1, colspan=4)
plt.xlim(nmin_ax, nmax_ax)
plt.ylim(ymin_ax, ymax_ax)
plt.annotate("", xytext=(nmin_ax, 0), xycoords='data', xy=(nmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(n, a3, linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=4)
plt.setp(bl, visible=False)
plt.text(nmax_ax, baseline, '$k$', fontsize=fontsize2, ha='center', va='baseline')
i = p-1
plt.text(i, baseline, '$p$', fontsize=fontsize1, ha='center', va='baseline')
i = 0
plt.text(i, baseline, '${}$'.format(i), fontsize=fontsize1, ha='center', va='baseline')
plt.text(y_sep, ymax_ax, '$a[p-k]$', fontsize=fontsize2, ha='left', va='center')
plt.axis('off')

ax = plt.subplot2grid((1, 16), (0, 12), rowspan=1, colspan=4)
plt.xlim(nmin_ax, nmax_ax)
plt.ylim(ymin_ax, ymax_ax)
plt.annotate("", xytext=(nmin_ax, 0), xycoords='data', xy=(nmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(n, a4, linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=4)
plt.setp(bl, visible=False)
plt.text(nmax_ax, baseline, '$k$', fontsize=fontsize2, ha='center', va='baseline')
i = p+j-1
plt.text(i, baseline, '$p+i$', fontsize=fontsize1, ha='center', va='baseline')
i = j
plt.text(i, baseline, '$i$', fontsize=fontsize1, ha='center', va='baseline')
plt.text(y_sep, ymax_ax, '$a[p+i-k]$', fontsize=fontsize2, ha='left', va='center')
plt.axis('off')


plt.savefig('problem_8_28_an.pdf', bbox_inches='tight')
plt.show()