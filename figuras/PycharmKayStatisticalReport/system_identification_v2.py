import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy import signal
import math
from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)
rc('mathtext', fontset='cm')

##########################
#### ESTE ES EL USADO ####
##########################

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

def sample_crosscorrelation(u, x, p):
    n = len(x)
    r = np.zeros(p+1)
    for i in np.arange(p+1):
        r[i] = np.mean(u[:n-i] * x[i:])
    return r


#### Parametros del problema ####
# Orden del filtro FIR a identificar
p = 11
# Largo de la señal de entrada
N = 1000
# Número de experimentos para calcular el error cuadrático medio de la estimacion
nexp = 10000
# Potencia de la señal de entrada
pot_u1 = 1
# Filtro a estimar (FIR con h[n] triangular)
h = signal.triang(p)
# Potencia de las señales de salida del filtro
pot_x1 = pot_u1 * np.sum(np.square(h))
# Potencia del ruido aditivo
pot_w = pot_x1  # para snr = 0 dB
# pot_w = 1
# Coeficientes del filtro IIR de orden dos para generar el ruido coloreado.
b = [1]
a = [1, -0.8, 0.64]  # polos de magnitud 0.8 y fase pi/3
#a = [1, -0.9, 0.81] # polos de magnitud 0.9 y fase pi/3
# Potencia del ruido coloreado (proceso AR(2))
pot_u2 = (1 + a[2]) / ((1 - a[2]) * ((1 + a[2])**2 - a[1]**2))


# Error cuadratico medio de la estimacion de cada coeficiente
mse_h1 = np.zeros(p)
mse_h2 = np.zeros(p)

r_uu2_accum = np.zeros(p)

#np.random.seed(42)
np.random.seed(46)

for i in np.arange(nexp):

    ###### Entradas
    ### 1 - WGN.
    u1 = math.sqrt(pot_u1) * np.random.randn(N)

    ### 2 - Ruido coloreado, obtenido mediante el filtrado de ruido blanco con un filtro IIR de orden 2 (Proceso
    ### autorregresivo de orden 2, AR(2))
    # Ruido blanco filtrado
    u2 = signal.lfilter(b, a, np.random.randn(N))

    # normalización a potencia pot_u1
    u2 = math.sqrt(pot_u1 / pot_u2) * u2

    ###### Salida del sistema a identificar
    x1 = signal.lfilter(h, 1, u1)
    x2 = signal.lfilter(h, 1, u2)

    ###### Salidas observadas contaminadas con ruido blanco, SNR = 0 dB
    # Ruido aditivo
    w = math.sqrt(pot_w) * np.random.randn(N)
    # Salidas observadas
    x1_obs = x1 + w
    x2_obs = x2 + w

    ###### Cálculo de los estimadores MVU de los coeficientes del sistema desconocido

    ###### Autocorrelacion de la entrada y correlación cruzada entre la entrada y la salida en cada caso

    ### Entrada u1[n]
    ### Autocorrelacion
    r_uu1 = sample_crosscorrelation(u1, u1, p-1)
    # Correlacion cruzada entre u1[n] y x[n]
    r_u1x = sample_crosscorrelation(u1, x1_obs, p-1)
    # Matriz de autocorrelacion
    Ru1 = linalg.toeplitz(r_uu1)
    # Estimador de los coeficientes
    h1_est = linalg.solve(Ru1, r_u1x)

    ### Entrada u2[n]
    ### Autocorrelacion
    r_uu2 = sample_crosscorrelation(u2, u2, p-1)
    # Correlacion cruzada entre u1[n] y x[n]
    r_u2x = sample_crosscorrelation(u2, x2_obs, p-1)
    # Matriz de autocorrelacion
    Ru2 = linalg.toeplitz(r_uu2)
    # Estimador de los coeficientes
    h2_est = linalg.solve(Ru2, r_u2x)

    # Calculo del error cuadrático medio de la estimacion
    mse_h1 = mse_h1 + np.square(h1_est - h)
    mse_h2 = mse_h2 + np.square(h2_est - h)

    # Calculo de la autocorrelacion de u2 como promedio en todos los experimentos. No se usa, es solo para verificacion.
    r_uu2_accum = r_uu2_accum + r_uu2

mse_h1 = mse_h1 / nexp
mse_h2 = mse_h2 / nexp
r_uu2_accum = r_uu2_accum / nexp

###### Cálculo analítico de la psd y la autocorrelación del ruido coloreado
# numero de muestras de la fft
nfft = 1024
# Respuesta en frecuencia del filtro generador del ruido coloreado
w, U2 = signal.freqz(b, a, worN=nfft, whole=True)
# PSD del ruido coloreado
psd_u2 = np.square(np.absolute(U2))
# Autocorrelacion del ruido coloreado
r_uu2_analitic = np.real(np.fft.ifft(psd_u2))
pot_u2_alt = r_uu2_analitic[0]
r_uu2_analitic = r_uu2_analitic / pot_u2_alt
# Normalizacion a potencia unitaria
psd_u2 = psd_u2[:nfft//2] / pot_u2_alt
w = w[:nfft//2]


## Gráficas

# comparacion de la autocorrelación analitica y muestral (para verificación)
plt.figure(0)
plt.plot(np.arange(p), r_uu2_analitic[:p], 'sk')
plt.plot(np.arange(p), r_uu2_accum[:p], 'sr')
plt.title("Autocorrelación muestral y autocorrelación analítica")
plt.xlabel("Retardo (muestras)")
plt.ylabel("Amplitud")
plt.legend(["analítica", "muestral"])


### Respuesta la impulso del filtro a identificar
nmin = 0
nmax = p-1
ymin = 0
ymax = np.amax(h)

delta_n = 1
nmin_ax = nmin - delta_n
nmax_ax = nmax + 2 * delta_n
delta_y = 0.3
ymax_ax = ymax + delta_y
ymin_ax = ymin - delta_y

n = np.arange(nmin, nmax + 1)

baseline = -0.25
fontsize1 = 12
fontsize2 = 14
yt_sep = 0.12

fig = plt.figure(1, figsize=(6, 2), frameon=False)
plt.xlim(nmin_ax, nmax_ax)
plt.ylim(ymin_ax, ymax_ax)
plt.annotate("", xytext=(nmin_ax, 0), xycoords='data', xy=(nmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(n, h, linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=4.5)
plt.setp(bl, visible=False)
plt.text(nmax_ax, baseline, '$k$', fontsize=fontsize2, ha='right', va='baseline')
for i in np.arange(1, p):
    plt.text(i, baseline, '${}$'.format(i), fontsize=fontsize1, ha='center', va='baseline')
i = 0
plt.text(i-yt_sep, baseline, '${}$'.format(i), fontsize=fontsize1, ha='right', va='baseline')
plt.text(0.3, ymax_ax, '$h[k]$', fontsize=fontsize2, ha='left', va='center')

plt.plot([0, 0.15], [1, 1], 'k')
plt.text(-yt_sep, 1, '$1$', fontsize=fontsize1, ha='right', va='center')

plt.axis('off')
plt.savefig('system_identification_impulse_response.pdf', bbox_inches='tight')

# length of the ticks for all plots (6 pixels)
display_length = 6  # in pixels

### PSD y autocorrelación de la entrada u2[n] (ruido coloreado)

fig = plt.figure(2, figsize=(6, 5), frameon=False)

# PSD de u2[n]

xmin = 0
xmax = math.pi
ymin = 0
ymax = np.amax(psd_u2)

delta_x = 0.35
xmin_ax = xmin - 0.1
xmax_ax = xmax + delta_x
delta_y = 1.2
ymin_ax = ymin - delta_y
ymax_ax = ymax + delta_y

baseline = -1
ylm = -0.07

ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

plt.plot(w, psd_u2, color='k', linewidth=2)

plt.text(xmax_ax, baseline, '$\omega\,\mathrm{(rad)}$', fontsize=fontsize2, ha='center', va='baseline')
# xticks
plt.plot([math.pi/3, math.pi/3], [0, xtl], 'k')
plt.plot([math.pi/2, math.pi/2], [0, xtl], 'k')
plt.plot([math.pi, math.pi], [0, xtl], 'k')
plt.plot([math.pi/3, math.pi/3], [0, ymax], 'k--', lw=1)
# xticks labels
plt.text(math.pi/3, baseline, '$\dfrac{\pi}{3}$', fontsize=fontsize2, ha='center', va='baseline')
plt.text(math.pi/2, baseline, '$\dfrac{\pi}{2}$', fontsize=fontsize2, ha='center', va='baseline')
plt.text(math.pi, baseline, '$\pi$', fontsize=fontsize2, ha='center', va='baseline')
plt.text(-ylm/2, baseline, '$0$', fontsize=fontsize2, ha='left', va='baseline')
# yticks and labels
for i in np.arange(0, ymax, 4):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ylm, i, '${:.0f}$'.format(i), fontsize=fontsize2, ha='right', va='center')
# y axis label
plt.text(-ylm, ymax_ax, '$S_{u_2u_2}(\omega)}$', fontsize=fontsize2, ha='left', va='center')
plt.axis('off')

# Autocorrelación de u2[n]
# se usa la misma escala que la PSD para homogeneidad de las gráficas.
ymax = ymax / 2
ymax_ax = (ymax_ax - ymin_ax) / 2
ymin_ax = -ymax_ax
baseline = -1.1

ax = plt.subplot2grid((4, 1), (2, 0), rowspan=2, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))


# numero de muestras para las las gráficas
M = 31
n = np.linspace(0, xmax, M)
# factor de normalizacion de la autocorrelacion
fnorm = ymax / np.amax(np.absolute(r_uu2_analitic))

(markers, stemlines, bl) = plt.stem(n, r_uu2_analitic[:M] * fnorm, linefmt='k', markerfmt='sk',
                                    use_line_collection=True)
plt.setp(markers, markersize=4.5)
plt.setp(bl, visible=False)

plt.text(xmax_ax, baseline, '$k$', fontsize=fontsize2, ha='right', va='baseline')

# plt.plot([math.pi, math.pi], [0, xtl], 'k')
# xtick and xticks labels
i = [10, 20, 30]
for ii in i:
    plt.text(ii * xmax / (M - 1), baseline, '${:.0f}$'.format(ii), fontsize=fontsize2, ha='center', va='baseline')
    tt = ii * math.pi / 30
    plt.plot([tt, tt], [0, xtl], 'k')


# yticks and labels
for i in np.arange(-1, 2):
    plt.plot([0, ytl], [i * fnorm, i * fnorm], 'k')
    plt.text(ylm, i * fnorm, '${:.0f}$'.format(i), fontsize=fontsize2, ha='right', va='center')
# y axis label
plt.text(-ylm, ymax_ax, '$r_{u_2u_2}[k]$', fontsize=fontsize2, ha='left', va='center')
plt.axis('off')

plt.savefig('system_identification_u2_input_v2.pdf', bbox_inches='tight')

### Realización de las entradas u1[n] y u2[v]

fig = plt.figure(3, figsize=(9, 5), frameon=False)

# u1[n]
# número de muestras para las las gráficas
M = 101

xmin = 0
xmax = M-1
ymin = -3
ymax = 3

delta_x = 7
xmin_ax = xmin - 2
xmax_ax = xmax + delta_x
delta_y = 0
ymin_ax = ymin - delta_y
ymax_ax = ymax + delta_y

baseline = -0.7
ylm = -1.5

ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(np.arange(M), u1[:M], linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=3.5)
plt.setp(stemlines, linewidth=1)
plt.setp(bl, visible=False)

for i in np.arange(-2, 3):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ylm, i, '${:.0f}$'.format(i), fontsize=fontsize1, ha='right', va='center')

plt.text(xmax_ax, baseline, '$n$', fontsize=fontsize2, ha='right', va='baseline')
plt.text(-ylm, ymax_ax, '$u_1[n]$', fontsize=fontsize2, ha='left', va='center')
plt.axis('off')

ax = plt.subplot2grid((4, 1), (2, 0), rowspan=2, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(np.arange(M), u2[:M], linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=3.5)
plt.setp(stemlines, linewidth=1)
plt.setp(bl, visible=False)

for i in np.arange(-2, 3):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ylm, i, '${:.0f}$'.format(i), fontsize=fontsize1, ha='right', va='center')

plt.text(xmax_ax, baseline, '$n$', fontsize=fontsize2, ha='right', va='baseline')
plt.text(-ylm, ymax_ax, '$u_2[n]$', fontsize=fontsize2, ha='left', va='center')
plt.axis('off')

plt.savefig('system_identification_inputs_v2.pdf', bbox_inches='tight')

### Realización de las salidas x1[n] y x2[v]
fig = plt.figure(4, figsize=(9, 5), frameon=False)

ymin = -8
ymax = 8

delta_y = 0
ymin_ax = ymin - delta_y
ymax_ax = ymax + delta_y

baseline = -0.7 * 8 / 3
ylm = -1.5

ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(np.arange(M), x1[:M], linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=3.5)
plt.setp(stemlines, linewidth=1)
plt.setp(bl, visible=False)
(markers, stemlines, bl) = plt.stem(np.arange(M), x1_obs[:M], linefmt='r', markerfmt='sr', use_line_collection=True)
plt.setp(markers, markersize=3.5)
plt.setp(stemlines, linewidth=1)
plt.setp(bl, visible=False)


for i in np.arange(-6, 7, 2):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ylm, i, '${:.0f}$'.format(i), fontsize=fontsize1, ha='right', va='center')

plt.text(xmax_ax, baseline, '$n$', fontsize=fontsize2, ha='right', va='baseline')
plt.text(-ylm, ymax_ax, '$x_1[n],$', fontsize=fontsize2, ha='left', va='center')
plt.text(10, ymax_ax, '$x_1[n]+w[n]$', fontsize=fontsize2, ha='left', va='center', color='red')
plt.axis('off')

ax = plt.subplot2grid((4, 1), (2, 0), rowspan=2, colspan=1)

plt.xlim(xmin_ax, xmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(np.arange(M), x2[:M], linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=3.5)
plt.setp(stemlines, linewidth=1)
plt.setp(bl, visible=False)
(markers, stemlines, bl) = plt.stem(np.arange(M), x2_obs[:M], linefmt='r', markerfmt='sr', use_line_collection=True)
plt.setp(markers, markersize=3.5)
plt.setp(stemlines, linewidth=1)
plt.setp(bl, visible=False)

for i in np.arange(-6, 7, 2):
    plt.plot([0, ytl], [i, i], 'k')
    plt.text(ylm, i, '${:.0f}$'.format(i), fontsize=fontsize1, ha='right', va='center')

plt.text(xmax_ax, baseline, '$n$', fontsize=fontsize2, ha='right', va='baseline')
plt.text(-ylm, ymax_ax, '$x_2[n],$', fontsize=fontsize2, ha='left', va='center')
plt.text(10, ymax_ax, '$x_2[n]+w[n]$', fontsize=fontsize2, ha='left', va='center', color='red')
plt.axis('off')

plt.savefig('system_identification_outputs_v2.pdf', bbox_inches='tight')

### Estimación de la respuesta al impulso del filtro a identificar
nmin = 0
nmax = p-1
ymin = 0
ymax = np.amax(h)

delta_n = 1
nmin_ax = nmin - delta_n
nmax_ax = nmax + 2 * delta_n
delta_y = 0.3
ymax_ax = ymax + delta_y
ymin_ax = ymin - delta_y

n = np.arange(nmin, nmax + 1)

baseline = -0.25
yt_sep = 0.12

fig = plt.figure(5, figsize=(6, 4), frameon=False)

ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2, colspan=1)
plt.xlim(nmin_ax, nmax_ax)
plt.ylim(ymin_ax, ymax_ax)
plt.annotate("", xytext=(nmin_ax, 0), xycoords='data', xy=(nmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(m1, stemlines, bl) = plt.stem(n, h, linefmt='k', markerfmt='sk', use_line_collection=True, label='$\mathrm{sistema}$')
plt.setp(m1, markersize=4.5)
plt.setp(bl, visible=False)
(m2, stemlines, bl) = plt.stem(n, h1_est, linefmt='r', markerfmt='sr', use_line_collection=True,
                               label='$\mathrm{estimación}$')
plt.setp(m2, markersize=4.5)
plt.setp(bl, visible=False)
# legend
leg = plt.legend(loc=(0.75, 0.7), fontsize=12, frameon=False)

plt.text(nmax_ax, baseline, '$k$', fontsize=fontsize2, ha='right', va='baseline')
for i in np.arange(1, p):
    plt.text(i, baseline, '${}$'.format(i), fontsize=fontsize1, ha='center', va='baseline')
i = 0
plt.text(i-yt_sep, baseline, '${}$'.format(i), fontsize=fontsize1, ha='right', va='baseline')
plt.text(0.3, ymax_ax, '$h[k]$', fontsize=fontsize2, ha='left', va='center')

plt.plot([0, 0.15], [1, 1], 'k')
plt.text(-yt_sep, 1, '$1$', fontsize=fontsize1, ha='right', va='center')
plt.text((nmin_ax+nmax_ax)/2, ymax_ax, '$\mathrm{Entrada:\,WGN}$', fontsize=12, ha='center', va='center')

plt.axis('off')


ax = plt.subplot2grid((4, 1), (2, 0), rowspan=2, colspan=1)
plt.xlim(nmin_ax, nmax_ax)
plt.ylim(ymin_ax, ymax_ax)
plt.annotate("", xytext=(nmin_ax, 0), xycoords='data', xy=(nmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

(markers, stemlines, bl) = plt.stem(n, h, linefmt='k', markerfmt='sk', use_line_collection=True)
plt.setp(markers, markersize=4.5)
plt.setp(bl, visible=False)
(markers, stemlines, bl) = plt.stem(n, h2_est, linefmt='r', markerfmt='sr', use_line_collection=True)
plt.setp(markers, markersize=4.5)
plt.setp(bl, visible=False)


plt.text(nmax_ax, baseline, '$k$', fontsize=fontsize2, ha='right', va='baseline')
for i in np.arange(1, p):
    plt.text(i, baseline, '${}$'.format(i), fontsize=fontsize1, ha='center', va='baseline')
i = 0
plt.text(i-yt_sep, baseline, '${}$'.format(i), fontsize=fontsize1, ha='right', va='baseline')
plt.text(0.3, ymax_ax, '$h[k]$', fontsize=fontsize2, ha='left', va='center')

plt.plot([0, 0.15], [1, 1], 'k')
plt.text(-yt_sep, 1, '$1$', fontsize=fontsize1, ha='right', va='center')
plt.text((nmin_ax+nmax_ax)/2, ymax_ax, '$\mathrm{Entrada:\,ruido\,coloreado}$', fontsize=12, ha='center', va='center')
plt.axis('off')

plt.savefig('system_identification_impulse_response_mvu_v2.pdf', bbox_inches='tight')

### CRLB y MSE práctico.

crlb1 = pot_w / N * np.ones(p)
Ru2 = linalg.toeplitz(r_uu2)
Ru2 = linalg.toeplitz(r_uu2_analitic[0:p])
crlb2 = pot_w * np.diagonal(linalg.inv(Ru2))/N

ymin = 0
ymax = 0.022

delta_y = 0.002
ymax_ax = ymax + 0.009
ymin_ax = ymin - 0.002

baseline = -0.0025

fig = plt.figure(6, figsize=(6, 4), frameon=False)
ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
plt.xlim(nmin_ax, nmax_ax)
plt.ylim(ymin_ax, ymax_ax)

# horizontal and vertical ticks length
xtl, ytl = convert_display_to_data_coordinates(ax.transData, length=display_length)

plt.annotate("", xytext=(nmin_ax, 0), xycoords='data', xy=(nmax_ax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin_ax), xycoords='data', xy=(0, ymax_ax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))


plt.plot(n, crlb1, 'k', label='$\mathrm{CRLB\;WGN}$')
plt.plot(n, mse_h1, 'sk', markersize=5, label='$\mathrm{WGN}$')
plt.plot(n, crlb2, 'r', label='$\mathrm{CRLB\;ruido\;coloreado}$')
plt.plot(n, mse_h2, 'sr', markersize=5, label='$\mathrm{ruido\;coloreado}$')

plt.text(nmax_ax, baseline, '$k$', fontsize=fontsize2, ha='right', va='baseline')
for i in np.arange(1, p):
    plt.text(i, baseline, '${}$'.format(i), fontsize=fontsize1, ha='center', va='baseline')
    plt.plot([i, i], [0, xtl], 'k', linewidth=1)
i = 0
plt.text(i-yt_sep, baseline, '${}$'.format(i), fontsize=fontsize1, ha='right', va='baseline')
for i in np.arange(0.01, 0.025, 0.01):
    plt.text(-yt_sep, i, '${}$'.format(i), fontsize=fontsize1, ha='right', va='center')
    plt.plot([0, ytl], [i, i], 'k', linewidth=1)

plt.text(0.3, ymax_ax, '$\mathrm{MSE}[k]$', fontsize=fontsize1, ha='left', va='center')


leg = plt.legend(loc=(0.5, 0.7), fontsize=fontsize1, frameon=False)

plt.axis('off')
plt.savefig('system_identification_crlb_mse_v2.pdf', bbox_inches='tight')

# frecuencia de resonancia del filtro generador de ruido coloreado
zp = np.angle(np.roots(a))
N = 2 * math.pi / zp[0]
print("Período N = {}".format(N))
print("Frecuencia de resonancia = {}".format(zp[0]))
print("pi/3 = {}".format(math.pi / 3))

# plt.show()