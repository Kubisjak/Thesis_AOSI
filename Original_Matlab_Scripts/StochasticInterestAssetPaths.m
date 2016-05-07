function S = StochasticInterestAssetPaths(S0,r,sig,T,steps,nsims)
% Function to generate sample paths for assets assuming geometric
% Brownian motion.
%
% S = AssetPaths(S0,mu,sig,dt,steps,nsims)
%
% Inputs: S0 - stock price
%       : sig - volatility
%       : dt - size of time steps
%       : steps - number of time steps to calculate
%       : nsims - number of simulation paths to generate
%
% Output: S - a matrix where each column represents a simulated
%             asset price path.

dt = T/steps;

r = repmat(r,1,nsims);

% Generate potential paths
S = S0*[ones(1,nsims); ...
            cumprod(exp((r - sig*sig/2)*dt ...
            +sig*sqrt(dt)*randn(steps,nsims)),1)];
end