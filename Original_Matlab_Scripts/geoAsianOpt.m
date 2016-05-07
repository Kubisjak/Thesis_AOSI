function [price_call, price_put] = geoAsianOpt(S0,sigma,K,r,T,steps)
%This is a closed form solution for geometric Asian options
%
%S0 = Current price of underlying asset
%sigma = Volatility
%K = Strike price
%r = Risk-free rate
%T = Time to maturity\
%steps = # of time steps
Nt = T*steps;

adj_sigma=sigma*sqrt((2*Nt+1)/(6*(Nt+1)));

rho=0.5*(r-(sigma^2)*0.5+adj_sigma^2);

d1 = (log(S0/K)+(rho+0.5*adj_sigma^2)*T)/(adj_sigma*sqrt(T));
d2 = (log(S0/K)+(rho-0.5*adj_sigma^2)*T)/(adj_sigma*sqrt(T));

price_call = exp(-r*T)*(S0*exp(rho*T)*normcdf(d1)-K*normcdf(d2));
price_put = exp(-r*T)*(K*normcdf(-d2)-S0*exp(rho*T)*normcdf(-d1));

end