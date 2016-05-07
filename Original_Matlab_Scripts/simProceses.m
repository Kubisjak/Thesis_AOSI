function [AssetPaths,InterestPaths] = simProceses(S0,sigmaS,r,T,R0,gamma,alpha,sigmaR,nsteps,nsims,rho)

AssetPaths = zeros(nsims,nsteps+1);
InterestPaths = zeros(nsims,nsteps+1);
AssetPaths(:,1) = S0;
InterestPaths(:,1) = R0;

W1 = randn(nsims,nsteps);
W2 = rho*W1 + sqrt(1-rho^2)*randn(nsims,nsteps);  

dt = T/nsteps;

nudt = (r-0.5*sigmaS^2)*dt;
sigmaRdt = sigmaS*sqrt(dt);

gammadt = gamma*(1-exp(-alpha*dt));


for i=1:nsims
    for j=1:nsteps
        AssetPaths(i,j+1)=AssetPaths(i,j)*exp(nudt + sigmaRdt*W2(i,j));
        InterestPaths(i,j+1)= InterestPaths(i,j)*exp(-alpha*dt)+ ...
            gammadt + sigmaR*sqrt((1-exp(-2*alpha*dt))/(2*alpha))*W1(i,j);
    end
end


end