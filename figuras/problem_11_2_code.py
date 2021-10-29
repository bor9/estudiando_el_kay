from scipy.stats import norm
from scipy import optimize

def j(z):
    return epsilon * norm.cdf(z-x) + (1 - epsilon) * norm.cdf(z+x) - 1/2

z0 = 0
j_root = optimize.root(j, z0)
