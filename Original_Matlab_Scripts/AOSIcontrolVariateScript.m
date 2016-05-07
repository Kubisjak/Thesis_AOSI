clc;
clear all;

%--------------------------------------------------------------------------
% Set up parameters for Asian Options:
S0 =100;       % Price of underlying today
K = 90;       % Strike at expiry
sigmaS = 0.05;  % expected vol.
T = 1; 
nsteps = 500;
nsims = 10000; % Number of simulated paths

% Set up parameters for UO model in stochastic interest rate:
R0 = 0.15; %initial value
sigmaR = 0.05; % volatility
gamma = 0.15; %long term mean
alpha = 2; % rate of mean reversion

rho = 0.5;
dt=T/nsteps;

Param_K =(K);
Param_sigmaS =(sigmaS);

[SR,R] = simAssetPathsWithInterests(S0,Param_sigmaS,T,R0,gamma,alpha,sigmaR,nsteps,nsims,rho);

AssetPathsMean = mean(SR,2);
Rdt = R.*dt ;
DiscountR =  sum(Rdt,2);

aCallPayoff = max(AssetPathsMean-Param_K,0); % MEAN OF ROWNS !!!!

aCallPrice = aCallPayoff.*exp(-DiscountR); %Each row is discounted with mean of corresponding interest rate process
aCallPrice_mean = mean(aCallPrice);
aCallPrice_STDerror = std(aCallPrice)/sqrt(nsims);

% Geometric average MC price

b = alpha;
a = gamma * alpha;

AssetPathsGeoMean = geomean(SR,2);
gCallPayoff = max(AssetPathsGeoMean -Param_K,0);

gCallPrice = aCallPayoff.*exp(-DiscountR); 
gCallPrice_mean = mean(gCallPrice);
gCallPrice_STDerror = std(gCallPrice)/sqrt(nsims);
 
% Geometric average real price
D1= exp(S0);
D = log(D1) - 0.5 *sigmaS^2 +R0*exp(-a) + b*exp(-a);
C = R0*exp(-a) + b*exp(-a);
D2 = R0*exp(-a)+ b*exp(-a);

d1 = log(D/Param_K)+sigmaS^2;
d2 = log(S0/K)+0.5*sigmaS^2 + C;

gPriceReal = (normcdf(d1)-normcdf(d2))*(-0.5*sigmaS^2 +C+D2);