clear variables
close all
clc


% =====================================
% Input Variables
Settings.NRepl = 2000; % number of replications 
Settings.dt = 1/250; % time-step of the simulation, e.g. 1/250 is daily if 250 business days per year.


Params.T = 5; % Maturity
Params.K = 1;  % Strike

% CPI
Params.CPI0 = 1;
Params.CPIdrift = 0.0249;
Params.gamma = 0.175;

% Short-Rate
Params.n0 = 0.02;
Params.kappa = 0.231;
Params.mu = 0.046;
Params.sigma = 0.00546;

% Correlation
Params.rho = 0.8;

% Market Prices of Risk
Params.lambda = 0;
Params.theta = 0;
% =======================================

% Simulate the asset paths
Settings.PlotSims = 'yes';
[CPIPaths, nPaths] = kamil_fncAssetPaths(Params, Settings);

% Calculate Asian option value
WMC = kamil_fncAsianOption(nPaths, CPIPaths, Params, Settings);




