function  W = kamil_fncAsianOption(nPaths, CPIPaths, Params, Settings)

% Reinsurer Share With and Without Index Clause by Monte Carlo
%
% USES: normfit
% 
% OUTPUT: 
% W - Reinsurer Share WITHOUT Index Clause
%
% INPUT: 
% nPaths   - simulated short-rate process for discounting
% CPIPaths - simulated CPI process
% 
% Params - structure
% Params.T - maturity of the contract
% Params.K - priority
% Params.a - intensity of annuities
% 
%
% Kamil Kladivko 
% Last Update: August 6, 2014 (for Jakub Kubis)

dt = Settings.dt;
K = Params.K;
 
% Without Index Clause
AvgCPI = mean(CPIPaths);
DF = exp(-dt.*sum(nPaths));
payoffW = max(AvgCPI-K,0);
DpayoffW = DF.*payoffW;
[W, ~, CIW] = normfit(DpayoffW);
fprintf('\n    Asian Call = %3.3f, CI = (%3.3f, %3.3f)\n', W, CIW(1), CIW(2));
               
end