import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha):
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


def generate_asset_paths_no_interest(S0, r, sigmaS, T, nsteps, nsims):
    dt = T / nsteps
    # Generate potential paths
    assetpaths = S0 * np.ones([nsteps, nsims]) * np.cumprod(
        np.exp((r - sigmaS * sigmaS / 2) * dt + sigmaS * np.sqrt(dt) * np.random.randn(nsteps, nsims)), 0)
    return assetpaths

def geoAsianOpt(S0, sigma, K, r, T, steps):
    #This is a closed form solution for geometric Asian options
    #
    #S0 = Current price of underlying asset
    #sigma = Volatility
    #K = Strike price
    #r = Risk-free rate
    #T = Time to maturity\
    #steps = # of time steps
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
    DiscPayoff = np.exp(-r*T) * max(0, (S0*np.exp(nuT+siT*np.random.randn(nsims, 1)-K)))
    Price = np.mean(DiscPayoff)
    
    return Price
    
    
def SimulateOrnsteinUhlenbeck(S0, mu, sigma, _lambda, deltat, t):
    # NOT WORKING YET!!!
    periods = np.floor(t / deltat)
    S = np.zeros([periods, 1])
    S[0] = S0
    exp_minus_lambda_deltat = np.exp(-_lambda * deltat)


    # Calculate the random term.
    if (_lambda == 0):
        # Handle the case of lambda = 0 i.e. no mean reversion.
        dWt = np.sqrt(deltat) * np.randn(periods, 1)
    else:
        dWt = np.sqrt((1 - np.exp(-2 * _lambda * deltat)) / (2 * _lambda)) * np.random.randn(periods, 1)
        
        # And iterate through time calculating each price.
        for t in np.linspace(2,1,periods):
            S[t] = S[t - 1] * exp_minus_lambda_deltat + mu * (1 - exp_minus_lambda_deltat) + sigma * dWt[t]
            # OPTIM Note : % Precalculating all dWt's rather than one-per loop makes this function
            # approx 50% faster. Useful for Monte-Carlo simulations.
            # OPTIM Note : calculating exp(-lambda*deltat) makes it roughly 50% faster
            # again.
            # OPTIM Note : this is only about 25% slower than the rough calculation
            # without the exp correction.      
    return S

# Set up parameters for Asian Options:
S0 = 100  # Price of underlying today
K = 90  # Strike at expiry
sigmaS = 0.15  # expected vol.
T = 1
nsteps = 100
nsims = 1 # Number of simulated paths
r = 0.15

# Set up parameters for UO model in stochastic interest rate:
R0 = 0.15  # initial value
sigmaR = 0.05  # volatility
gamma = 0.15  # long term mean
alpha = 2  # rate of mean reversion

rho = 0.5
dt = T / nsteps


S = generate_asset_paths_no_interest(S0, r, sigmaS, T, nsteps, nsims)
R, S = generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha)
plt.plot(S)
plt.show()
plt.plot(R)
plt.show()
