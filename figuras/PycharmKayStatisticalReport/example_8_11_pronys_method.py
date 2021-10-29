import numpy as np
from scipy import linalg

# hd es la señal deseada dada, con hd.shape = (N, )
# el filtro de Prony a diseñar tiene función de tranferencia con
# q coeficientes en el numerador y p coeficientes en el denominador
# estimación de los coeficientes del denominador, a[k], k = 1,...,p
x = hd[q + 1:]
H = linalg.toeplitz(hd[q: N - 1], hd[q: q - p: -1])
a = np.linalg.solve(H.T @ H, -H.T @ x)
# estimación de los coeficientes del numerador, b[k], k = 0,...,q
h = hd[: q + 1]
H0 = linalg.toeplitz(np.concatenate(([0], hd[: q])), np.zeros((p, )))
b = h + H0 @ a