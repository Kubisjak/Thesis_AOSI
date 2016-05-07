clear all;
clc;

Param_nruns = [100,200,500,1000,10000];
Param_sigmaS = [0.05, 0.1, 0.2, 0.3];
Param_K = [90, 100, 110];

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


file = fopen('MC_AOSI_ControlVariate_output.tex', 'w');
fprintf(file, '\\begin{tabular}{|rrrrrrrr|}\\hline \n');
fprintf(file, '$\\sigma_s$ & $K$ & \\shortstack[r]{ Const. int. \\\\ Ve\\v{c}e\\v{r}''s price} &\\shortstack[r]{MC price with \\\\ control variate } & Variance & \\shortstack[r]{ STD \\\\ Error} & \\shortstack[r]{EO \\\\ MC price} & \\shortstack[r]{EO \\\\ real price}  \\\\ \\hline \\hline \n');

for i=1:length(Param_sigmaS)
        fprintf(file, '%8.2f',Param_sigmaS(i));
for j=1:length(Param_K)
    
[price_call,price_put] = Vecer_asiancontinuous(S0,Param_K(j),gamma,Param_sigmaS(i),T);
    
%--------------------------------------------------------------------------
% Generate potential future asset paths

[SR,R] = simAssetPathsWithInterests(S0,Param_sigmaS(i),T,R0,gamma,alpha,sigmaR,nsteps,nsims,rho);

% -------------------------------------------------------------------------
% European Option MC simulation

Rdt = R.*dt;
r = sum(Rdt,2);
discount = exp(-r.*T);
Emax = max(0,SR(:,nsteps+1)-Param_K(j));

EuropeanPayoff = Emax;
EuropeanPrices = exp(-r.*T).*max(0,SR(:,nsteps+1)-Param_K(j));
EuropeanPriceMC = mean(EuropeanPrices);

% -------------------------------------------------------------------------
% European Option Exact Price with stochastic interest rates

% Transformation of Vasicek model: 
% alpha(gamma-R(t))dt +sigma*dW(t) to
% (a-bR(t))dt + sigma*dW(t) 

b = alpha;
a = gamma * alpha;

% Calculation According to VU: Theorem 2.2 

lambda = (1/b)*(1-exp(-b*T));
A = (a/b)*T +(R0-(a/b))*lambda;
B = (sigmaR/b)*sqrt(T - lambda - 0.5*b*lambda^2);
nu = sqrt(Param_sigmaS(i)^2 + B^2 - 2*rho*Param_sigmaS(i)*B);

d1 = ((log(S0)/Param_K(j)) + A - 0.5*nu^2*T)/(Param_sigmaS(i)*T) + nu*sqrt(T);
d2 = ((log(S0)/Param_K(j)) + A - 0.5*nu^2*T)/(Param_sigmaS(i)*T) - B*sqrt(T);

I1 = normcdf(d1);
I2 = normcdf(d2);

EuropeanPriceReal = S0*I1 - Param_K(j)*exp(-A+0.5*B^2*T)*I2;

%--------------------------------------------------------------------------
AssetPathsMean = mean(SR,2);
Rdt = R.*dt ;
DiscountR =  sum(Rdt,2);

aCallPayoff = max(AssetPathsMean-Param_K(j),0); % MEAN OF ROWNS !!!!

aCallPrice = aCallPayoff.*exp(-DiscountR); %Each row is discounted with mean of corresponding interest rate process

aCallPrice_mean = mean(aCallPrice);
aCallPrice_STDerror = std(aCallPrice)/sqrt(nsims);
%--------------------------------------------------------------------------

%Control variate coefficient:

cov_geometric_call = cov(aCallPrice,EuropeanPrices);
k_opt_call = -cov_geometric_call(1,2)/cov_geometric_call(2,2);

%--------------------------------------------------------------------------
% Calculate the control variate (with European Option) estimate:

CallPrice = aCallPrice + k_opt_call*(EuropeanPrices - EuropeanPriceReal);
CallPrice_mean = mean(CallPrice);
CallPrice_STDerror = std(CallPrice)/sqrt(nsims);
%--------------------------------------------------------------------------
% Output:

    fprintf(file, '    & %8.0f & %8.2f & %8.2f & %8.4f & %8.4f & %8.2f & %8.2f \\\\ ',Param_K(j),price_call,CallPrice_mean, var(CallPrice),CallPrice_STDerror,EuropeanPriceMC,EuropeanPriceReal);
    fprintf(file, '\n');
    
    if j==length(Param_K) 
        if i~=length(Param_sigmaS)
        fprintf(file, '\\hline \\hline ');
        else fprintf(file, '\\hline');
        end
    end

end
end

fprintf(file, '\\end{tabular}\n');
fclose(file);
movefile('MC_AOSI_ControlVariate_output.tex','/Users/Kubisjak/Google Drive/Vyzkumak/Vypracovanie/Final/Matlab');