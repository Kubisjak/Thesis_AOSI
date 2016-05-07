# Imports --------------------------------------------------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk
import time
import numpy as np
import matplotlib
from scipy.stats import norm

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # NavigationToolbar2TkAgg

# from matplotlib.figure import Figure
plt.rcParams["toolbar"] = "None"  # Do not display toolbar when calling plot


# ----------------------------------------------------------------------------------------------------------------------


def generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha, rho):

    dt = T / nsteps

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


def geoAsianOpt(S0, sigma, K, r, T, steps, rho):
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


def european_real_price(S0, R0, K, T, alpha, gamma, sigmaS, sigmaR, rho):
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


def main(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha, rho):

    dt = T / nsteps
    R, SR = generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha, rho)

    # European Option MC simulation
    Rdt = R * dt
    r = sum(Rdt, 0)
    discount = np.exp(-r * T)
    Emax = np.maximum(0, SR[nsteps, :] - K)

    EuropeanPayoff = Emax
    EuropeanPrices = np.exp(-r * T) * np.maximum(0, SR[nsteps, :] - K)
    EuropeanPriceMC = np.mean(EuropeanPrices)

    EuropeanPriceReal = european_real_price(S0, R0, K, T, alpha, gamma, sigmaS, sigmaR, rho)

    # Asset Paths
    AssetPathsMean = np.mean(SR, 0)
    Rdt = R * dt
    DiscountR = sum(Rdt, 0)

    aCallPayoff = np.maximum(AssetPathsMean - K, 0)  # MEAN OF ROWNS !!!!
    aCallPrice = aCallPayoff * np.exp(-DiscountR)  # Each row is discounted with mean of corresponding interest rate process

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

    return SR, R, EuropeanPriceMC, EuropeanPriceReal, CallPrice_mean


def _quit():
    GUI.quit()
    GUI.destroy()


def init():
    S0_init = S0.get()  # Price of underlying today
    K_init = K.get()  # Strike at expiry
    sigmaS_init = sigmaS.get()  # expected vol.
    nsteps_init = nsteps.get()
    nsims_init = nsims.get()  # Number of simulated paths

    # Set up parameters for UO model in stochastic interest rate:
    R0_init = R0.get()  # initial value
    sigmaR_init = sigmaR.get()  # volatility
    gamma_init = gamma.get()  # long term mean
    alpha_init = alpha.get()  # rate of mean reversion
    rho_init = rho.get()

    # parameters = ["S0", "K", "sigmaS", "T", "nsteps", "nsims", "R0", "sigmaR", "gamma", "alpha", "rho"]
    # print(S0_init, K_init, sigmaS_init, nsteps_init, nsteps_init, R0_init, sigmaR_init, alpha_init, rho_init)

    T = 1

    SR, R, EuropeanPriceMC, EuropeanPriceReal, CallPrice_mean = \
        main(S0_init, nsims_init, nsteps_init, K_init, sigmaS_init, T, R0_init, sigmaR_init, gamma_init, alpha_init,
             rho_init)

    return SR, R, EuropeanPriceMC, EuropeanPriceReal, CallPrice_mean


def calculate():
    SR, R, EuropeanPriceMC, EuropeanPriceReal, CallPrice_mean = init()
    plt.clf()

    plt.plot(SR, alpha=0.4)
    plt.plot(np.mean(SR, 1), color="black", linewidth=1)
    plt.xlabel("Step", fontsize=8)
    plt.ylabel("Price", fontsize=8)
    plt.suptitle("Asset price with stochastic interest", fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)

    plt.tight_layout()
    plt.gcf().canvas.draw()

    print("European Option MC price: ", EuropeanPriceMC)
    print("European Option real price: ", EuropeanPriceReal)
    print("Asian Option Monte Carlo price with control variate: ", CallPrice_mean)

    return


GUI = tk.Tk()
GUI.title("Asian Option Pricing Model")

# Create Canvas --------------------------------------------------------------------------------------------------------
GUI.grid_columnconfigure(0, weight=1)
GUI.grid_columnconfigure(1, weight=1)
GUI.grid_columnconfigure(2, weight=4)
GUI.grid_columnconfigure(3, weight=4)

# Buttons
# Parameter frame
# Number of cars
button_position_row = 0
label_parameters = tk.Label(GUI, text="Parameters: ")  # Create label for parameter
label_parameters.grid(row=button_position_row, column=0, columnspan=2, sticky=tk.W + tk.E)  # Place label on grid
button_position_row += 1

S0 = tk.DoubleVar(GUI)  # Create IntVariable called
label_S0 = tk.Label(GUI, text="S0: ")  # Create label for parameter
label_S0.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_S0 = tk.Entry(GUI, textvariable=S0)  # Create entry widget
entry_S0.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

K = tk.DoubleVar(GUI)  # Create IntVariable
label_K = tk.Label(GUI, text="K: ")  # Create label for parameter
label_K.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_K = tk.Entry(GUI, textvariable=K)  # Create entry widget
entry_K.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

sigmaS = tk.DoubleVar(GUI)  # Create IntVariable
label_sigmaS = tk.Label(GUI, text="SigmaS: ")  # Create label for parameter
label_sigmaS.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_sigmaS = tk.Entry(GUI, textvariable=sigmaS)  # Create entry widget
entry_sigmaS.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

nsteps = tk.IntVar(GUI)  # Create IntVariable
label_nsteps = tk.Label(GUI, text="nsteps: ")  # Create label for parameter
label_nsteps.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_nsteps = tk.Entry(GUI, textvariable=nsteps)  # Create entry widget
entry_nsteps.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

nsims = tk.IntVar(GUI)  # Create IntVariable
label_nsims = tk.Label(GUI, text="nsims: ")  # Create label for parameter
label_nsims.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_nsims = tk.Entry(GUI, textvariable=nsims)  # Create entry widget
entry_nsims.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

R0 = tk.DoubleVar(GUI)  # Create IntVariable
label_R0 = tk.Label(GUI, text="R0: ")  # Create label for parameter
label_R0.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_R0 = tk.Entry(GUI, textvariable=R0)  # Create entry widget
entry_R0.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

sigmaR = tk.DoubleVar(GUI)  # Create IntVariable
label_sigmaR = tk.Label(GUI, text="SigmaR: ")  # Create label for parameter
label_sigmaR.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_sigmaR = tk.Entry(GUI, textvariable=sigmaR)  # Create entry widget
entry_sigmaR.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

gamma = tk.DoubleVar(GUI)  # Create IntVariable
label_gamma = tk.Label(GUI, text="Gamma: ")  # Create label for parameter
label_gamma.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_gamma = tk.Entry(GUI, textvariable=gamma)  # Create entry widget
entry_gamma.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

alpha = tk.DoubleVar(GUI)  # Create IntVariable
label_alpha = tk.Label(GUI, text="Alpha: ")  # Create label for parameter
label_alpha.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_alpha = tk.Entry(GUI, textvariable=alpha)  # Create entry widget
entry_alpha.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

rho = tk.DoubleVar(GUI)  # Create IntVariable
label_rho = tk.Label(GUI, text="Rho: ")  # Create label for parameter
label_rho.grid(row=button_position_row, column=0, sticky=tk.E)  # Place label on grid
entry_rho = tk.Entry(GUI, textvariable=rho)  # Create entry widget
entry_rho.grid(row=button_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)  # Place entry widget on grid
button_position_row += 1

button_quit = ttk.Button(GUI, text="Quit", command=_quit)
button_quit.grid(row=button_position_row, column=3, sticky=tk.W + tk.E + tk.N + tk.S)

button_calculate = ttk.Button(GUI, text="Calculate", command=lambda: calculate())
button_calculate.grid(row=button_position_row, column=2, sticky=tk.W + tk.E + tk.N + tk.S)

# Animation canvas frame
canvas_frame = tk.Frame(GUI, bd=1)
canvas_frame.grid(row=0, rowspan=button_position_row, column=2, columnspan=2)
f = plt.figure(figsize=(3, 3), dpi=150)
canvas = FigureCanvasTkAgg(f, master=canvas_frame)
canvas.show()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
canvas.tkcanvas.pack(side=tk.TOP, fill=tk.BOTH)

S0.set(100)  # Price of underlying today
K.set(90)  # Strike at expiry
sigmaS.set(0.05)  # expected vol.
nsteps.set(100)
nsims.set(10)  # Number of simulated paths
# Set up parameters for UO model in stochastic interest rate:
R0.set(0.15)  # initial value
sigmaR.set(0.05)  # volatility
gamma.set(0.15)  # long term mean
alpha.set(2)  # rate of mean reversion
rho.set(0.5)

GUI.mainloop()
