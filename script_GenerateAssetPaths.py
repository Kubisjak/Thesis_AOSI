# -*- coding: utf-8 -*-
"""
@author: Kubisjak
Function will simulate Asset Paths with Stochastic Interest

def SimAssetPaths(S0,sigmaS,T,R0,gamma,alpha,sigmaR,nsteps,nsims,rho):
    return AssetPaths, AssetPathsWithInterest
"""

from __future__ import division  # Force float division
import math
import numpy as np
import matplotlib.pyplot as plt


def generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha):
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
    return InterestPaths, AssetPaths


def AssetPaths(S0, r, sigmaS, T, nsteps, nsims):
    # Function to generate sample paths for assets assuming geometric
    # Brownian motion.
    #
    # S = AssetPaths(S0,mu,sig,dt,steps,nsims)
    #
    # Inputs: S0 - stock price
    #       : sig - volatility
    #       : dt - size of time steps
    #       : steps - number of time steps to calculate
    #       : nsims - number of simulation paths to generate
    #
    # Output: S - a matrix where each column represents a simulated
    #             asset price path.

    dt = T / nsteps

    # Generate potential paths
    assetpaths = S0 * np.ones([nsims, 1]) * np.cumprod(
        np.exp((r - sigmaS * sigmaS / 2) * dt + sigmaS * np.sqrt(dt) * np.random.randn(nsteps, nsims)), 0)
    return assetpaths


def european_option_simulation(S0, K, r, T, sigma, nsims):

    nuT = (r - 0.5 * sigma - 2) * T
    siT = sigma * np.sqrt(T)

    DiscPayoff = np.exp(-r*T) * max(0, (S0*np.exp(nuT+siT*np.random.randn(nsims, 1)-K)))

    Price = np.mean(DiscPayoff)

    return Price

# Set up parameters for Asian Options:
S0 = 100  # Price of underlying today
K = 90  # Strike at expiry
sigmaS = 0.15  # expected vol.
T = 1
nsteps = 1000
nsims = 1  # Number of simulated paths
r = 0.15

# Set up parameters for UO model in stochastic interest rate:
R0 = 0.15  # initial value
sigmaR = 0.05  # volatility
gamma = 0.15  # long term mean
alpha = 2  # rate of mean reversion

rho = 0.5
dt = T / nsteps


S = AssetPaths(S0, r, sigmaS, T, nsteps, nsims)

plt.plot(S)
plt.show()

InterestPaths, AssetPaths = generate_paths_with_interest(S0, nsims, nsteps, K, sigmaS, T, R0, sigmaR, gamma, alpha)

plt.plot(AssetPaths.T)
plt.show()
