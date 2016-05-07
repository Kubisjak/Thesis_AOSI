function Price = EOSim(SO,K,r,T,sigma,nsims)

nuT = (r - 0.5*sigma-2)*T;
siT = sigma * sqrt(T) ;

DiscPayoff = exp(-r*T)*max(O, SO*exp(nuT+siT*randn(nsims, 1) -K);
Price = mean(DiscPayoff);



end