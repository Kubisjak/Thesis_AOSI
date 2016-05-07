clear all;
clc;

Param_nruns = [100,200,500,1000,10000];
Param_sigma = [0.05, 0.1, 0.2, 0.3];
Param_K = [90, 100, 110];

S0 =100;       % Price of underlying today
r = 0.15;      % Risk free rate
T = 1; 
steps = 500;
nruns = 10000; % Number of simulated paths

file = fopen('MC_AO_Control_Variate_output.tex', 'w');
fprintf(file, '\\begin{tabular}{|rrrrrrr|}\\hline \n');
fprintf(file, '$\\sigma$ & $K$ & Ve\\v{c}e\\v{r}''s price & \\shortstack[r]{Geometric average \\\\ closed form solution} & \\shortstack[r]{Control variate \\\\ estimate} & Variance & STD Error \\\\ \\hline \\hline \n');


for i=1:length(Param_sigma)
        fprintf(file, '%8.2f',Param_sigma(i));
for j=1:length(Param_K)
    
[price_call,price_put] = Vecer_asiancontinuous(S0,Param_K(j),r,Param_sigma(i),T);
    
%--------------------------------------------------------------------------
% Generate potential future asset paths
S = AssetPaths(S0,r,Param_sigma(i),T,steps,nruns);

%--------------------------------------------------------------------------
% Arithmetic average:
% calculate the payoff for each path for a Put: 

% calculate the payoff for each path for a Call:
aCallPayoff = max(mean(S)-Param_K(j),0);

% discount back
aCallPrice = aCallPayoff*exp(-r*T);
aCallPrice_mean = mean(aCallPrice);
aCallPrice_STDerror = std(aCallPrice)/sqrt(nruns);
%--------------------------------------------------------------------------

%Geometric average:
% Calculate exact asian option price using geometric average
[gCallPriceReal, gPutPriceReal] = geoAsianOpt(S0,Param_sigma(i),Param_K(j),r,T,steps);

% Calculate Monte Carlo geometric price
gCallPayoff = max(geomean(S)-Param_K(j),0);

% Discount back
gCallPrice = gCallPayoff*exp(-r*T);
gCallPrice_mean = mean(gCallPrice);
gCallPrice_STDerror = std(gCallPrice)/sqrt(nruns);

%--------------------------------------------------------------------------
%Control variate coefficient:

cov_geometric_call = cov(aCallPayoff,gCallPayoff);
k_opt_call = -cov_geometric_call(1,2)/cov_geometric_call(2,2);

%--------------------------------------------------------------------------
% Calculate the control variate (with geometric average) estimate:

% Call:
CallPrice = aCallPrice + k_opt_call*(gCallPrice - gCallPriceReal);
CallPrice_mean = mean(CallPrice);
CallPrice_STDerror = std(CallPrice)/sqrt(nruns);


%--------------------------------------------------------------------------
% Output:
    fprintf(file, '    & %8.0f & %8.2f & %8.2f & %8.2f & %8.4f & %8.4f \\\\ ',Param_K(j),price_call,gCallPriceReal,CallPrice_mean,var(CallPrice),CallPrice_STDerror);
    fprintf(file, '\n');
    
    if j==length(Param_K) 
        if i~=length(Param_sigma)
        fprintf(file, '\\hline \\hline ');
        else fprintf(file, '\\hline');
        end
    end
end
end

fprintf(file, '\\end{tabular}\n');
fclose(file);
movefile('MC_AO_Control_Variate_output.tex','/Users/Kubisjak/Google Drive/Vyzkumak/Vypracovanie/Final/Matlab');