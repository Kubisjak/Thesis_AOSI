import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

# Original simulation parameters
# interest rate model notation: dr(t)= alpha (gamma-r(t))dt + sigma dW_r(t)
S0 = 100  # Price of underlying today
K = 90  # Strike at expiry
sigmaS = 0.05  # expected vol.
R0 = 0.15  # initial value
sigmaR = 0.05  # volatility
gamma_original = 0.15  # long term mean
alpha_original = 2  # rate of mean reversion

# Current model parameters
# Interest rate model dr(t)= (alpha*r(t)+beta)dt + gamma dW_r(t)
T = 1
t = 0
r_t = R0  # = R0
gamma = sigmaR  # Volatility

# Transformation: alpha = - alpha_original, beta = alpha_original*gamma_original
alpha = - alpha_original
beta = alpha_original * gamma_original

# Asset Paths Parameters
S_t = S0
sigma = sigmaS

# Coorrelation parameter
rho = 0.5

D_1 = np.exp(((T - t) / T) * np.log(S_t))
J_1 = -1 * (sigma ** 2 / (4 * T)) * (t - T) ** 2
C = - r_t / alpha * (np.exp(alpha * (t - T)) - 1) + beta / alpha ** 2 * (np.exp(alpha * (t - T)) - 1) + beta / alpha * (
    T - t)
SR_1 = r_t / alpha ** 2 * (alpha * t - np.exp(alpha * (t - T)) * (alpha * T + 1) + 1)
SR_2 = beta / (2 * alpha ** 3) * (
    alpha ** 2 * T ** 2 - alpha ** 2 * t ** 2 + 2 * np.exp(alpha * (t - T)) * (alpha * T + 1) - 2 * alpha * t - 2)

D = np.log(D_1) + J_1 + C - (1 / T * (SR_1 + SR_2))

K_star = np.log(K / D_1) - J_1 - C + (SR_1 + SR_2) / T


def alpha_hat(u, T):
    global alpha
    x = (1 - np.exp(alpha * (u - T))) / alpha
    return x


def sigma_tilde(u, T):
    global sigma
    x = sigma * (T - u) / T
    return x


def gamma_tilde(u, T):
    global gamma, alpha
    x = gamma / alpha * (1 - np.exp(alpha * (u - T))) - gamma / (alpha ** 2 * T) * (
        alpha * u - ((alpha * T + 1) * (np.exp(alpha * (u - T)))) + 1)
    return x


def gamma_2tilde(u, T):
    global gamma, alpha
    x = gamma / alpha * (1 - np.exp(alpha * (u - T))) - gamma / (alpha ** 2 * T) * (
        alpha * u - ((alpha * T + 1) * (np.exp(alpha * (u - T)))) + 1) - gamma * (
        (1 - np.exp(alpha * (u - T))) / alpha)
    return x


# [0] means we take only first value of the output
sigma2_x = integrate.quad(lambda u: abs(gamma ** 2 * alpha_hat(u, T) ** 2), t, T)[0]
print(sigma2_x)

sigma2_x1 = integrate.quad(lambda u: abs(gamma_2tilde(u, T) ** 2), t, T)[0]
print(sigma2_x1)

sigma2_x2 = integrate.quad(lambda u: abs(gamma_tilde(u, T) ** 2), t, T)[0]
print(sigma2_x2)

sigma2_y = integrate.quad(lambda u: abs(sigma_tilde(u, T) ** 2), t, T)[0]
print(sigma2_y)

sigma_xy = integrate.quad(lambda u: sigma_tilde(u, T) * (-gamma) * alpha_hat(u, T), t, T)[0]
print(sigma_xy)

sigma_x1y = integrate.quad(lambda u: sigma_tilde(u, T) * gamma_2tilde(u, T), t, T)[0]
print(sigma_x1y)

sigma_x2y = integrate.quad(lambda u: sigma_tilde(u, T) * gamma_tilde(u, T), t, T)[0]
print(sigma_x2y)

sigma_xx2 = integrate.quad(lambda u: gamma_tilde(u, T) * (-gamma) * alpha_hat(u, T), t, T)[0]
print("Sigma xx2:", sigma_xx2)

sigma_x1x2 = integrate.quad(lambda u: gamma_tilde(u, T) * gamma_2tilde(u, T), t, T)[0]
print("Sigma xx2:", sigma_xx2)

sigma2_v = sigma2_x1 + sigma2_y + 2 * rho * sigma_x1y
sigma2_u = sigma2_x2 + sigma2_y + 2 * rho * sigma_x2y
sigma_u = np.sqrt(sigma2_x2 + sigma2_y + 2 * rho * sigma_x2y)

cov_uv = rho * (sigma_x1y + sigma_x2y) + sigma2_y + sigma_x1x2
cov_xu = rho * sigma_xy + sigma_xx2

x = (cov_uv - K_star) / sigma_u
I_1 = np.exp(D - C) + np.exp(sigma2_v/2) * norm.cdf(x)
print(np.exp(C + D))

print(x, I_1)

y = (cov_xu - K_star) / sigma_u
I_2 = -K * np.exp(-C) * np.exp(sigma2_x/2) * norm.cdf(y)
print(y, I_2)

P = I_1 + I_2

print("Price:", P)
