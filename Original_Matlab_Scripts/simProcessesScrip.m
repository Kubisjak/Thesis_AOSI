S0 =100;       % Price of underlying today
K = 95;       % Strike at expiry
sigmaS = 0.05;  % expected vol.
T = 1; 
nsteps = 100;
nsims = 10; % Number of simulated paths
r=0.15;


R0 = 0.15; %initial value
sigmaR = 0.05; % volatility
gamma = 0.15; %long term mean
alpha = 2; % rate of mean reversion


rho =0;




[S,R] = simProceses(S0,sigmaS,r,T,R0,gamma,alpha,sigmaR,nsteps,nsims,rho);

%plot(linspace(0,T,length(S)),S);

% plot(linspace(0,T,length(R)),R);
% str = fprintf('Ornstein-Uhlenbeck process. $\\sigma$= %8.2f, $\\gamma$= %8.2f, $\\alpha$= %8.2f',sigmaR,gamma,alpha);


 
SR = simAssetPathsWithInterests(S0,sigmaS,T,R0,gamma,alpha,sigmaR,nsteps,nsims,rho);
plot(linspace(0,T,length(SR)),SR);
str = sprintf('GBM with stochastic interest rates under OU process');
title(str);
xlabel('t');
ylabel('S(t)');

