# Imports --------------------------------------------------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk
import time
import numpy as np
import matplotlib
from scipy.stats import norm
import scipy.integrate as integrate

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # NavigationToolbar2TkAgg

# from matplotlib.figure import Figure
plt.rcParams["toolbar"] = "None"  # Do not display toolbar when calling plot


# ----------------------------------------------------------------------------------------------------------------------


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


class AsianOptionApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Asian Option Pricing Model")

        GUI = tk.Frame(self, borderwidth=1, padx=10, pady=10)
        GUI.pack()
        GUI.grid_columnconfigure(0, weight=1)
        GUI.grid_columnconfigure(1, weight=1)

        self.L = LeftFrame(GUI)
        self.R = RightFrame(GUI)
        self.O = OutputFrame(GUI)

        self.set_starting_parameters()

    def set_starting_parameters(self):
        self.L.nsteps.set(100)
        self.L.nsims.set(5000)
        self.L.t.set(0)
        self.L.T.set(1)
        self.L.S0.set(100)
        self.L.R0.set(0.15)
        self.L.K.set(100)
        self.L.sigmaS.set(0.05)
        self.L.sigmaR.set(0.05)
        self.L.alpha.set(1)
        self.L.gamma.set(1)
        self.L.rho.set(0.5)

    def update_parameters(self, nsteps, nsims, t, T, S0, R0, K, sigmaS, sigmaR, gamma, alpha, rho):
        self.L.nsteps.set(nsteps)
        self.L.nsims.set(nsims)
        self.L.t.set(t)
        self.L.T.set(T)
        self.L.S0.set(S0)
        self.L.R0.set(R0)
        self.L.K.set(K)
        self.L.sigmaS.set(sigmaS)
        self.L.sigmaR.set(sigmaR)
        self.L.alpha.set(alpha)
        self.L.gamma.set(gamma)
        self.L.rho.set(rho)

    def generate_paths_with_interest(self, S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR,
                                     gamma, alpha, rho):

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
                                          + gammadt + sigmaR * np.sqrt(
                    (1 - np.exp(-2 * alpha * dt)) / (2 * alpha)) * W1[
                                                          i, j]

                AssetPaths[i, j + 1] = AssetPaths[i, j] * np.exp(
                    (InterestPaths[i, j] - 0.5 * sigmaS ** 2) * dt + sigmaRdt * W2[
                        i, j])
        return InterestPaths.T, AssetPaths.T

    def geoAsianOpt(self, S0, sigma, K, r, T, steps):
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

        d1 = (np.log(S0 / K) + (rho + 0.5 * adj_sigma ** 2) * T) / (
            adj_sigma * np.sqrt(T))
        d2 = (np.log(S0 / K) + (rho - 0.5 * adj_sigma ** 2) * T) / (
            adj_sigma * np.sqrt(T))

        price_call = np.exp(-r * T) * (
            S0 * np.exp(rho * T) * norm.cdf(d1) - K * norm.cdf(d2))
        price_put = np.exp(-r * T) * (
            K * norm.cdf(-d2) - S0 * np.exp(rho * T) * norm.cdf(-d1))

        return price_call, price_put

    def european_option_simulation(self, S0, K, r, T, sigma, nsims):
        nuT = (r - 0.5 * sigma - 2) * T
        siT = sigma * np.sqrt(T)
        DiscPayoff = np.exp(-r * T) * max(0, (
            S0 * np.exp(nuT + siT * np.random.randn(nsims, 1) - K)))
        Price = np.mean(DiscPayoff)
        return Price

    def european_real_price(self, S0, K, T, alpha, gamma, sigmaS, sigmaR):
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

        d1 = (
            (np.log(S0) / K + A - 0.5 * nu ** 2 * T) / (sigmaS * T) + nu * np.sqrt(
                T))
        d2 = (
            (np.log(S0) / K + A - 0.5 * nu ** 2 * T) / (sigmaS * T) - B * np.sqrt(
                T))

        I1 = norm.cdf(d1)
        I2 = norm.cdf(d2)

        EuropeanPriceReal = S0 * I1 - K * np.exp(-A + 0.5 * B ** 2 * T) * I2

        return EuropeanPriceReal

    def MC_AO_Arithmetic_Average_Stochastic_Interest_Vasicek(self, S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma,
                                                             alpha, rho):
        R, SR = generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0,
                                             sigmaR,
                                             gamma, alpha, rho)

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

        EuropeanPriceReal = european_real_price(S0, K, T, alpha, gamma, sigmaS,
                                                sigmaR)

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

    def CF_AO_Geometric_Average_Stochastic_Interest_Vasicek(self, t, T, S0, R0, K, sigmaS, sigmaR, gamma_original,
                                                            alpha_original, rho):
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
        SR_1 = r_t / alpha ** 2 * (
            alpha * t - np.exp(alpha * (t - T)) * (alpha * T + 1) + 1)
        SR_2 = beta / (2 * alpha ** 3) * (
            alpha ** 2 * T ** 2 - alpha ** 2 * t ** 2 + 2 * np.exp(
                alpha * (t - T)) * (
                alpha * T + 1) - 2 * alpha * t - 2)

        D = np.log(D_1) + J_1 + C - (1 / T * (SR_1 + SR_2))

        K_star = np.log(K / D_1) - J_1 - C + (SR_1 + SR_2) / T

        def alpha_hat(u, T):
            x = (1 - np.exp(alpha * (u - T))) / alpha
            return x

        def sigma_tilde(u, T):
            x = sigma * (T - u) / T
            return x

        def gamma_tilde(u, T):
            x = (gamma / alpha) * (1 - np.exp(alpha * (u - T))) - (gamma / (
                alpha ** 2 * T)) * (
                                                                      alpha * u - (
                                                                          (
                                                                              alpha * T + 1) * (
                                                                              np.exp(
                                                                                  alpha * (
                                                                                      u - T)))) + 1)
            return x

        def gamma_2tilde(u, T):
            x = gamma / alpha * (1 - np.exp(alpha * (u - T))) - gamma / (
                alpha ** 2 * T) * (
                                                                    alpha * u - ((
                                                                                     alpha * T + 1) * (
                                                                                     np.exp(
                                                                                         alpha * (
                                                                                             u - T)))) + 1) - gamma * (
                (1 - np.exp(alpha * (u - T))) / alpha)
            return x

        # [0] means we take only first value of the output
        sigma2_x = \
            integrate.quad(lambda u: abs(gamma ** 2 * alpha_hat(u, T) ** 2), t, T)[
                0]

        sigma2_x1 = integrate.quad(lambda u: abs(gamma_2tilde(u, T) ** 2), t, T)[0]

        sigma2_x2 = integrate.quad(lambda u: abs(gamma_tilde(u, T) ** 2), t, T)[0]

        sigma2_y = integrate.quad(lambda u: abs(sigma_tilde(u, T) ** 2), t, T)[0]

        sigma_xy = \
            integrate.quad(lambda u: sigma_tilde(u, T) * (-gamma) * alpha_hat(u, T),
                           t, T)[0]

        sigma_x1y = \
            integrate.quad(lambda u: sigma_tilde(u, T) * gamma_2tilde(u, T), t, T)[
                0]

        sigma_x2y = \
            integrate.quad(lambda u: sigma_tilde(u, T) * gamma_tilde(u, T), t, T)[0]

        sigma_xx2 = \
            integrate.quad(lambda u: gamma_tilde(u, T) * (-gamma) * alpha_hat(u, T),
                           t, T)[0]

        sigma_x1x2 = \
            integrate.quad(lambda u: gamma_tilde(u, T) * gamma_2tilde(u, T), t, T)[
                0]

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
        return P


class LeftFrame:
    def __init__(self, master):
        global frame_rows

        self.frame = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=500)
        self.frame.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        self.frame.rowconfigure(1, weight=1)
        self.frame.rowconfigure(5, weight=1)

        # Parameter Frame Label
        self.frame_position_row = 0
        self.parameter_frame_label = tk.Label(self.frame, text="Parameters: ", justify=tk.LEFT)
        self.parameter_frame_label.grid(row=self.frame_position_row, column=0, columnspan=1, sticky=tk.W)
        self.frame_position_row += 1

        # Parameter frame
        self.parameter_frame = tk.Frame(self.frame, relief=tk.GROOVE, borderwidth=1, padx=10, pady=10)
        self.parameter_frame.grid(row=self.frame_position_row, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        self.parameter_frame_position_row = 0

        self.t = tk.DoubleVar(master)
        self.T = tk.DoubleVar(master)

        self.S0 = tk.DoubleVar(master)  # Create IntVariable called
        label_S0 = tk.Label(self.parameter_frame, text="S0: ")
        label_S0.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_S0 = tk.Entry(self.parameter_frame, textvariable=self.S0)
        entry_S0.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.parameter_frame_position_row += 1

        self.K = tk.DoubleVar(master)  # Create IntVariable
        label_K = tk.Label(self.parameter_frame, text="K: ")
        label_K.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_K = tk.Entry(self.parameter_frame, textvariable=self.K)
        entry_K.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.parameter_frame_position_row += 1

        self.sigmaS = tk.DoubleVar(master)  # Create IntVariable
        label_sigmaS = tk.Label(self.parameter_frame, text="SigmaS: ")
        label_sigmaS.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_sigmaS = tk.Entry(self.parameter_frame, textvariable=self.sigmaS)
        entry_sigmaS.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.parameter_frame_position_row += 1

        self.nsteps = tk.IntVar(master)  # Create IntVariable
        label_nsteps = tk.Label(self.parameter_frame, text="nsteps: ")
        label_nsteps.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_nsteps = tk.Entry(self.parameter_frame, textvariable=self.nsteps)
        entry_nsteps.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.parameter_frame_position_row += 1

        self.nsims = tk.IntVar(master)  # Create IntVariable
        label_nsims = tk.Label(self.parameter_frame, text="nsims: ")
        label_nsims.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_nsims = tk.Entry(self.parameter_frame, textvariable=self.nsims)
        entry_nsims.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)

        # Drop Down List
        self.frame_position_row += 1
        label_option_list_processes = tk.Label(self.frame, text="Stochastic interest process: ", justify=tk.LEFT)
        label_option_list_processes.grid(row=self.frame_position_row, column=0, columnspan=1, sticky=tk.W)

        self.frame_position_row += 1
        interest_stochastic_process_variable = tk.StringVar(master)
        interest_stochastic_process_variable.set("Ornstein-Uhlenbeck")  # default value

        self.frame_position_row += 1
        option_list_processes = tk.OptionMenu(self.frame, interest_stochastic_process_variable,
                                              "Ornstein-Uhlenbeck", "Hull-White")
        option_list_processes.configure(anchor="w")
        option_list_processes.grid(row=self.frame_position_row, sticky=tk.W + tk.E)

        # Stochastic interest process parameter frame - SIP
        self.frame_position_row += 1
        self.SIP_parameter_frame = tk.Frame(self.frame, relief=tk.GROOVE, borderwidth=1, padx=10, pady=10)
        self.SIP_parameter_frame.grid(row=self.frame_position_row, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        self.SIP_parameter_frame_position_row = 0
        self.R0 = tk.DoubleVar(master)  # Create IntVariable
        label_R0 = tk.Label(self.SIP_parameter_frame, text="R0: ")
        label_R0.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_R0 = tk.Entry(self.SIP_parameter_frame, textvariable=self.R0)
        entry_R0.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        self.sigmaR = tk.DoubleVar(master)  # Create IntVariable
        label_sigmaR = tk.Label(self.SIP_parameter_frame, text="SigmaR: ")
        label_sigmaR.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_sigmaR = tk.Entry(self.SIP_parameter_frame, textvariable=self.sigmaR)
        entry_sigmaR.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        self.gamma = tk.DoubleVar(master)  # Create IntVariable
        label_gamma = tk.Label(self.SIP_parameter_frame, text="Gamma: ")
        label_gamma.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_gamma = tk.Entry(self.SIP_parameter_frame, textvariable=self.gamma)
        entry_gamma.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        self.rho = tk.DoubleVar(master)  # Create IntVariable
        label_rho = tk.Label(self.SIP_parameter_frame, text="Rho: ")
        label_rho.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_rho = tk.Entry(self.SIP_parameter_frame, textvariable=self.rho)
        entry_rho.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        self.alpha = tk.DoubleVar(master)  # Create IntVariable
        label_alpha = tk.Label(self.SIP_parameter_frame, text="Alpha: ")
        label_alpha.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_alpha = tk.Entry(self.SIP_parameter_frame, textvariable=self.alpha)
        entry_alpha.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        # Total rows used
        # total_rows_left = self.parameter_frame_position_row + self.frame_left_position_row

        frame_rows = self.frame_position_row


class RightFrame:
    def __init__(self, master):
        global frame_rows
        self.frame = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=500)
        self.frame.grid(row=0, column=1, sticky=tk.W + tk.E + tk.N + tk.S)

        # Right Frame ----------------------------------------------------------------------------------------------
        # Animation canvas frame
        canvas_rowspan = frame_rows

        canvas_frame = tk.Frame(self.frame)
        canvas_frame.grid(row=0, rowspan=canvas_rowspan, column=0, columnspan=2)
        f = plt.figure(figsize=(2, 2), dpi=150)
        canvas = FigureCanvasTkAgg(f, master=canvas_frame)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
        # canvas.tkcanvas.pack(side=tk.TOP, fill=tk.BOTH)

        button_quit = ttk.Button(self.frame, text="Quit", command=lambda: self._quit(master))
        button_quit.grid(row=canvas_rowspan + 1, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        button_calculate = ttk.Button(self.frame, text="Calculate", command=lambda: self._quit(master))
        button_calculate.grid(row=canvas_rowspan + 1, column=1, sticky=tk.W + tk.E + tk.N + tk.S)

    def _quit(self, master):
        master.quit()
        master.destroy()


class OutputFrame:
    def __init__(self, master):
        self.frame_bottom_position_row = 0

        frame_bottom = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=100)
        frame_bottom.grid(row=1, column=0, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)

        label_bottom_frame = tk.Label(frame_bottom, text="Output: ", justify=tk.LEFT)
        label_bottom_frame.grid(row=self.frame_bottom_position_row, column=0, columnspan=1, sticky=tk.W)


app = AsianOptionApp()

app.mainloop()
