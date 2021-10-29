import numpy as np

# generación del proceso de Gauss-Markov h[n] = Ah[n-1] + u[n] de dimensión p=2.
# N es el número total de muestras a generar: h[0], ..., h[N-1]
# valor inicial h[-1]: variable aleatoria de distribución normal multivariada con
# media mu_hi y matriz de covarianza C_hi.
# media de h[-1]
mu_hi = [1, 1]
# covarianza de h[-1]
C_hi = [[0.1, 0], [0, 0.1]]
# u[n] es WGN con matriz de covarianza Q
# media de u[n]
mu_u = [0, 0]
# covarianza de u[n]
Q = [[0.0001, 0], [0, 0.0001]]
# matriz de transición de estados
A = np.array([[0.99, 0], [0, 0.999]])
h = np.zeros((2, N))  # matriz para almacenar los valores de h[n].
# generación de h[-1]
h_i = np.random.multivariate_normal(mu_hi, C_hi, 1)[0]
h_prev = h_i
for n in np.arange(N):
    h_prev = A @ h_prev + np.random.multivariate_normal(mu_u, Q, 1)[0]
    h[:, n] = h_prev
