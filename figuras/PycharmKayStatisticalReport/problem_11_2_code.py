from scipy.stats import norm
from scipy import optimize

def j(z):
    return epsilon2 * norm.cdf(z-x) + (1 - epsilon2) * norm.cdf(z+x) - 1/2

z0 = 0
j_root = optimize.root(j, z0)
