import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Parameters for Monte Carlo:
# Set up parameters for Asian Options:
S0 = 100  # Price of underlying today
K = 100  # Strike at expiry
sigmaS = 0.05  # expected vol.
nsteps = 500
nsims = 10000  # Number of simulated paths
# Set up parameters for UO model in stochastic interest rate:
R0 = 0.15  # initial value
sigmaR = 0.05  # volatility
gamma = 0.15  # long term mean
alpha = 2  # rate of mean reversion
rho = 0.5
t = 0
T = 1
dt = T / nsteps


def generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha, rho):
    AssetPaths = np.zeros((nsims, nsteps + 1))
    InterestPaths = np.zeros((nsims, nsteps + 1))

    AssetPaths[:, 0] = S0
    InterestPaths[:, 0] = R0

    W1 = np.random.randn(nsims, nsteps)
    W2 = rho * W1 + np.sqrt(1 - rho ** 2) * np.random.randn(nsims, nsteps)
    sigmaRdt = sigmaS * np.sqrt(dt)
    gammadt = gamma * (1 - np.exp(-alpha * dt))

    for i in range(0, nsims):
        for j in range(0, nsteps):
            InterestPaths[i, j + 1] = InterestPaths[i, j] * np.exp(-alpha * dt) \
                                      + gammadt + sigmaR * np.sqrt((1 - np.exp(-2 * alpha * dt)) / (2 * alpha)) * W1[
                i, j]

            AssetPaths[i, j + 1] = AssetPaths[i, j] * np.exp(
                (InterestPaths[i, j] - 0.5 * sigmaS ** 2) * dt + sigmaRdt * W2[i, j])
    return InterestPaths.T, AssetPaths.T


def geoAsianOpt(S0, sigma, K, r, T, steps):
    # This is a closed form solution for geometric Asian options
    #
    # S0 = Current price of underlying asset
    # sigma = Volatility
    # K = Strike price
    # r = Risk-free rate
    # T = Time to maturity\
    # steps = # of time steps
    Nt = T * steps

    adj_sigma = sigma * np.sqrt((2 * Nt + 1) / (6 * (Nt + 1)))

    rho = 0.5 * (r - (sigma ** 2) * 0.5 + adj_sigma ** 2)

    d1 = (np.log(S0 / K) + (rho + 0.5 * adj_sigma ** 2) * T) / (adj_sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (rho - 0.5 * adj_sigma ** 2) * T) / (adj_sigma * np.sqrt(T))

    price_call = np.exp(-r * T) * (S0 * np.exp(rho * T) * norm.cdf(d1) - K * norm.cdf(d2))
    price_put = np.exp(-r * T) * (K * norm.cdf(-d2) - S0 * np.exp(rho * T) * norm.cdf(-d1))

    return price_call, price_put


def european_option_simulation(S0, K, r, T, sigma, nsims):
    nuT = (r - 0.5 * sigma - 2) * T
    siT = sigma * np.sqrt(T)
    DiscPayoff = np.exp(-r * T) * max(0, (S0 * np.exp(nuT + siT * np.random.randn(nsims, 1) - K)))
    Price = np.mean(DiscPayoff)
    return Price


def european_real_price(S0, K, T, alpha, gamma, sigmaS, sigmaR):
    # Transformation of Vasicek model:
    # alpha(gamma-R(t))dt +sigma*dW(t) to
    # (a-bR(t))dt + sigma*dW(t)
    b = alpha
    a = gamma * alpha
    # Calculation According to VU: Theorem 2.2
    _lambda = (1 / b) * (1 - np.exp(-b * T))
    A = (a / b) * T + (R0 - (a / b)) * _lambda
    B = (sigmaR / b) * np.sqrt(T - _lambda - 0.5 * b * _lambda ** 2)
    nu = np.sqrt(sigmaS ** 2 + B ** 2 - 2 * rho * sigmaS * B)

    d1 = ((np.log(S0) / K + A - 0.5 * nu ** 2 * T) / (sigmaS * T) + nu * np.sqrt(T))
    d2 = ((np.log(S0) / K + A - 0.5 * nu ** 2 * T) / (sigmaS * T) - B * np.sqrt(T))

    I1 = norm.cdf(d1)
    I2 = norm.cdf(d2)

    EuropeanPriceReal = S0 * I1 - K * np.exp(-A + 0.5 * B ** 2 * T) * I2

    return EuropeanPriceReal


def MC_AO_Geometric_Average_Stochastic_Interest_Vasicek(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha, rho):
    R, SR = generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha, rho)

    # plt.plot(R)
    # plt.show()
    # plt.plot(SR)
    # plt.show()

    # European Option MC simulation
    Rdt = R * dt
    r = sum(Rdt, 0)
    discount = np.exp(-r * T)
    Emax = np.maximum(0, SR[nsteps, :] - K)

    EuropeanPayoff = Emax
    EuropeanPrices = np.exp(-r * T) * np.maximum(0, SR[nsteps, :] - K)
    EuropeanPriceMC = np.mean(EuropeanPrices)

    EuropeanPriceReal = european_real_price(S0, K, T, alpha, gamma, sigmaS, sigmaR)

    # Asset Paths
    AssetPathsMean = np.mean(SR, 0)
    Rdt = R * dt
    DiscountR = sum(Rdt, 0)

    aCallPayoff = np.maximum(AssetPathsMean - K, 0)  # MEAN OF ROWNS !!!!
    aCallPrice = aCallPayoff * np.exp(
        -DiscountR)  # Each row is discounted with mean of corresponding interest rate process

    aCallPrice_mean = np.mean(aCallPrice)
    aCallPrice_STDerror = np.std(aCallPrice) / np.sqrt(nsims)

    # Control variate coefficient:

    cov_European_call = np.cov(aCallPrice, EuropeanPrices)
    k_opt_call = -cov_European_call[0, 1] / cov_European_call[1, 1]

    # --------------------------------------------------------------------------
    # Calculate the control variate (with European Option) estimate:

    CallPrice = aCallPrice + k_opt_call * (EuropeanPrices - EuropeanPriceReal)
    CallPrice_mean = np.mean(CallPrice)
    CallPrice_STDerror = np.std(CallPrice) / np.sqrt(nsims)

    print("European Option MC price: ", EuropeanPriceMC)
    print("European Option real price: ", EuropeanPriceReal)
    print("Monte Carlo price with control variate: ", CallPrice_mean)
    return EuropeanPriceMC, EuropeanPriceReal, CallPrice_mean


def CF_AO_Geometric_Average_Stochastic_Interest_Vasicek(t, T, S0, R0, K, sigmaS, sigmaR, gamma_original, alpha_original,
                                                        rho):
    r_t = R0  # = R0
    gamma = sigmaR  # Volatility
    S_t = S0
    sigma = sigmaS

    # Transformation: alpha = - alpha_original, beta = alpha_original*gamma_original
    alpha = - alpha_original
    beta = alpha_original * gamma_original

    D_1 = np.exp(((T - t) / T) * np.log(S_t))
    J_1 = -1 * (sigma ** 2 / (4 * T)) * (t - T) ** 2
    C = - r_t / alpha * (np.exp(alpha * (t - T)) - 1) + beta / alpha ** 2 * (
    np.exp(alpha * (t - T)) - 1) + beta / alpha * (
        T - t)
    SR_1 = r_t / alpha ** 2 * (alpha * t - np.exp(alpha * (t - T)) * (alpha * T + 1) + 1)
    SR_2 = beta / (2 * alpha ** 3) * (
        alpha ** 2 * T ** 2 - alpha ** 2 * t ** 2 + 2 * np.exp(alpha * (t - T)) * (alpha * T + 1) - 2 * alpha * t - 2)

    D = np.log(D_1) + J_1 + C - (1 / T * (SR_1 + SR_2))

    K_star = np.log(K / D_1) - J_1 - C + (SR_1 + SR_2) / T

    def alpha_hat(u, T):
        x = (1 - np.exp(alpha * (u - T))) / alpha
        return x

    def sigma_tilde(u, T):
        x = sigma * (T - u) / T
        return x

    def gamma_tilde(u, T):
        x = gamma / alpha * (1 - np.exp(alpha * (u - T))) - gamma / (alpha ** 2 * T) * (
            alpha * u - ((alpha * T + 1) * (np.exp(alpha * (u - T)))) + 1)
        return x

    def gamma_2tilde(u, T):
        x = gamma / alpha * (1 - np.exp(alpha * (u - T))) - gamma / (alpha ** 2 * T) * (
            alpha * u - ((alpha * T + 1) * (np.exp(alpha * (u - T)))) + 1) - gamma * (
            (1 - np.exp(alpha * (u - T))) / alpha)
        return x

    # [0] means we take only first value of the output
    sigma2_x = integrate.quad(lambda u: abs(gamma ** 2 * alpha_hat(u, T) ** 2), t, T)[0]

    sigma2_x1 = integrate.quad(lambda u: abs(gamma_2tilde(u, T) ** 2), t, T)[0]

    sigma2_x2 = integrate.quad(lambda u: abs(gamma_tilde(u, T) ** 2), t, T)[0]

    sigma2_y = integrate.quad(lambda u: abs(sigma_tilde(u, T) ** 2), t, T)[0]

    sigma_xy = integrate.quad(lambda u: sigma_tilde(u, T) * (-gamma) * alpha_hat(u, T), t, T)[0]

    sigma_x1y = integrate.quad(lambda u: sigma_tilde(u, T) * gamma_2tilde(u, T), t, T)[0]

    sigma_x2y = integrate.quad(lambda u: sigma_tilde(u, T) * gamma_tilde(u, T), t, T)[0]

    sigma_xx2 = integrate.quad(lambda u: gamma_tilde(u, T) * (-gamma) * alpha_hat(u, T), t, T)[0]

    sigma_x1x2 = integrate.quad(lambda u: gamma_tilde(u, T) * gamma_2tilde(u, T), t, T)[0]

    sigma2_v = sigma2_x1 + sigma2_y + 2 * rho * sigma_x1y
    sigma2_u = sigma2_x2 + sigma2_y + 2 * rho * sigma_x2y
    sigma_u = np.sqrt(sigma2_x2 + sigma2_y + 2 * rho * sigma_x2y)

    cov_uv = rho * (sigma_x1y + sigma_x2y) + sigma2_y + sigma_x1x2
    cov_xu = rho * sigma_xy + sigma_xx2

    x = (cov_uv - K_star) / sigma_u
    I_1 = np.exp(D - C) + np.exp(sigma2_v / 2) * norm.cdf(x)

    y = (cov_xu - K_star) / sigma_u
    I_2 = -K * np.exp(-C) * np.exp(sigma2_x / 2) * norm.cdf(y)

    P = I_1 + I_2

    print("Price:", P)
    return P


print(CF_AO_Geometric_Average_Stochastic_Interest_Vasicek(t, T, S0, R0, K, sigmaS, sigmaR, gamma, alpha, rho))
print(
    MC_AO_Geometric_Average_Stochastic_Interest_Vasicek(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha, rho))
