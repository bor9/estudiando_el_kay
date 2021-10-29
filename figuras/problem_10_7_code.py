from scipy.stats import norm

hat_A = x + (norm.pdf(-A0 - x) - norm.pdf(A0 - x)) / (norm.cdf(A0 - x) - norm.cdf(-A0 - x))
