import numpy as np

# x_r y x_i son la señal de interferencia y la señal de referencia
# respectivamente con x_r.shape = x_i.shape = (N, ).
# p es el orden del filtro adaptativo y lamb es el factor de olvido.
# theta_i es la estimador inicial (en n=-1), con theta_i.shape = (p, ),
# y sigma_i es la matriz de covarianza del estimador inicial con
# sigma_i.shape = (p, p).
theta = theta_i
sigma = sigma_i
# agregado de ceros a x_r, ya que h[n] = [x_r[n] x_r[n-1] ... x_r[n-p+1]]
x_r_pad = np.concatenate((np.zeros((p - 1,)), x_r))
for i in np.arange(N):
    h = x_r_pad[i + (p - 1): i - 1 if i - 1 >= 0 else None: -1]
    e = x_i[i] - h @ theta
    K = (sigma @ h) / (lamb ** i + h @ sigma @ h)
    theta = theta + K * e
    sigma = (np.eye(p) - K[:, None] @ h[None, :]) @ sigma