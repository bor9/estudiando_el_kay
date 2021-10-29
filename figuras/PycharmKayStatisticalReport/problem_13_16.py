import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import numpy as np
from scipy import signal

from matplotlib import rc
from matplotlib import rcParams

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# colors
lgray = "#dddddd"  # ligth gray

### Parámetros ###
a = 0.8
var_u = 1
var_w = 1

# número de frecuencias entre 0 y 0.5
nf = 512


##################

# resolución de la ecuación de ricatti de estado estacionario
# para calcular M[\infty]
p = [-var_u * var_w, var_u + var_w * (1 - a ** 2), a ** 2]
print(poly.polyroots(p))
M_inf = np.max(poly.polyroots(p))
# cálculo de Mp[\infty]
Mp_inf = (a ** 2) * M_inf + var_u
# cálculo de K[\infty]
K_inf = Mp_inf / (var_w + Mp_inf)
print(M_inf)
print(Mp_inf)
print(K_inf)
print(a * (1 - K_inf))