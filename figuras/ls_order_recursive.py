import numpy as np
from numpy.linalg import inv, norm

# vector con el número de muestra: n = 0, ..., N-1
n = np.arange(N).reshape(-1, 1)
# número de paso
k = 1
# matriz de observación H en el paso k = 1
H = np.power(n, k - 1)
# matriz D en el paso k = 1
D = inv(H.T@H)
# estimador en el paso k = 1
theta = D @ (H.T @ x)
# error LS mínimo en el paso k = 1
Jmin = norm(x - H @ theta, ord=2) ** 2
for k in np.arange(2, kmax + 1):
    # cálculo de P_perp en el paso k - 1
    P_perp = np.identity(N) - H @ D @ H.T
    # columna a agregrar en H en el paso k: 
    # vector con los elementos del vector n elevados a la k - 1
    h = np.power(n, k - 1)
    # denominador de los elementos del vector theta y la matriz D en el paso k
    den = h.T @ P_perp @ h
    ## cálculo del estimador en el paso k
    # elemento theta_12 del estimador: dimensiones (k - 1) x 1
    theta_11 = theta - (D @ H.T @ h @ h.T @ P_perp @ x) / den
    # elemento theta_22 del estimador: dimensiones 1 x 1
    theta_12 = (h.T @ P_perp @ x) / den
    # concatenación en filas para formar el estimador theta en el paso k
    theta = np.concatenate((theta_11, theta_12), axis=0)
    ## construccion de la matrix D en el paso k
    D_11 = D + (D @ H.T @ h @ h.T @ H @ D) / den
    D_21 = - (D @ H.T @ h) / den
    # concatenación para formar D
    D = np.concatenate((np.concatenate((D_11, D_21), axis=1), 
                        np.concatenate((D_21.T, 1 / den), axis=1)), axis=0)
    # construcción de H en el paso k: [H h]
    H = np.concatenate((H, h), axis=1)
    ## cálculo del error mínimo en el paso k
    Jmin = Jmin - (h.T @ P_perp @ x) ** 2 / den
