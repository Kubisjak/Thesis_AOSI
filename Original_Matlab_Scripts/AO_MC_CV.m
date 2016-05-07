function [PutPrice,CallPrice,PutErr,CallErr] = AO_MC_CV(S0,r,sigma,T,steps,nruns)
%--------------------------------------------------------------------------
% Generate potential future asset paths
S = AssetPaths(S0,r,sigma,T,steps,nruns);

%--------------------------------------------------------------------------
% Arithmetic average:
% calculate the payoff for each path for a Put: 
aPutPayoff = max(K-mean(S),0);

% calculate the payoff for each path for a Call:
aCallPayoff = max(mean(S)-K,0);

% discount back
aPutPrice = aPutPayoff*exp(-r*T);
aCallPrice = aCallPayoff*exp(-r*T);

%--------------------------------------------------------------------------
%Geometric average:
% Calculate exact asian option price using geometric average
[gCallPriceReal, gPutPriceReal] = geoAsianOpt(S0,sigma,K,r,T,steps);

% Calculate Monte Carlo geometric price
gPutPayoff = max(K-geomean(S),0);
gCallPayoff = max(geomean(S)-K,0);

% Discount back
gPutPrice = gPutPayoff*exp(-r*T);
gCallPrice = gCallPayoff*exp(-r*T);

%--------------------------------------------------------------------------
%Control variate coefficient:

cov_geometric_call = cov(gCallPayoff,aCallPayoff);
k_opt_call = -cov_geometric_call(1,2)/cov_geometric_call(2,2);

cov_geometric_put = cov(gPutPayoff,aPutPayoff);
k_opt_put = -cov_geometric_put(1,2)/cov_geometric_put(2,2);

%--------------------------------------------------------------------------
% Calculate the control variate (with geometric average) estimate:

% Put:
PutPrice = aPutPrice + k_opt_put*(gPutPrice - gPutPriceReal);
PutPrice_mean = mean(PutPrice);
PutPrice_STDerror = std(PutPrice)/sqrt(nruns);

% Call:
CallPrice = aCallPrice + k_opt_call*(gCallPrice - gCallPriceReal);
CallPrice_mean = mean(CallPrice);
CallPrice_STDerror = std(CallPrice)/sqrt(nruns);

PutPrice = PutPrice_mean;
CallPrice = CallPrice_mean;
PutErr = PutPrice_STDerror;
CallErr = CallPrice_STDerror;

end