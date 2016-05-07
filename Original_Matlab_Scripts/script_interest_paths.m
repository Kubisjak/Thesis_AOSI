clear all;
clc;
%--------------------------------------------------------------------------
% Set up parameters for Asian Options:
S0 =100;       % Price of underlying today
K = 90;       % Strike at expiry
sigma = 0.10;  % expected vol.
T = 1; 
steps = 6000;
nruns = 100; % Number of simulated paths

% Set up parameters for UO model in stochastic interest rate:
I0 = 0.15; %initial value
Isigma = 0.01; % volatility
mu = 0.15; %long term mean
lambda = 1; % rate of mean reversion

dt=T/steps;

%--------------------------------------------------------------------------
% Generate potential future asset paths with stochastic interest rates
% according to Ornstein Uhlenbeck model.

% r_stoch is a column vector with size of (steps,1), i.e. each step the
% interest rate is changed 

% Generating Asset paths with stochastic interest each step
r_stoch = SimulateOrnsteinUhlenbeck( I0, mu, Isigma, lambda, dt, T );
r = mean(r_stoch)

S = StochasticInterestAssetPaths(S0,r_stoch,sigma,T,steps,nruns);