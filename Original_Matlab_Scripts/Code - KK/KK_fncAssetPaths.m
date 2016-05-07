function [CPIPaths, nPaths]= kamil_fncAssetPaths(Params, Settings)
%
% Simulates the GBM for the CPI and an OU process for the short rate.
% Simulation - replications in columns, evolution in tine in rows, i.e., [NSteps, NRepl]
% 
% OUTPUT:
% CPIPaths - simulated CPI process as GBM
% nPaths - simulated nominal risk-free rate as OU
% 
% INPUT:
% Params - structure
%
% Settings - structure
%  .PlotSims: 'yes' plots simulation graphs
%
% Kamil Kladivko 
% Last Update: August 6, 2014 (for Jakub Kubis)

NRepl = Settings.NRepl;
dt = Settings.dt;

T = Params.T;
% CPI
CPI0 = Params.CPI0;
CPIdrift = Params.CPIdrift;
gamma = Params.gamma;
% Short-Rate
n0 = Params.n0;
kappa = Params.kappa;
mu = Params.mu;
sigma = Params.sigma;
% Correlation
rho = Params.rho;

NSteps = floor(T/dt);
fprintf('\n=====================================================================================\n')
fprintf('Simualating the GBM for the CPI and OU process for the short-rate\n'); 
fprintf('\n Replications = %3.0f, Steps = %3.0f, dt = %3.3f\n', NRepl, NSteps, dt);

CPIPaths = zeros(1+NSteps,NRepl);
nPaths = zeros(1+NSteps,NRepl);
CPIPaths(1,:) = CPI0;
nPaths(1,:) = n0;

W1 = randn(NSteps,NRepl);   
W2 = zeros(NSteps,NRepl); 

nudt = (CPIdrift-0.5*gamma^2)*dt;
gammadt = gamma*sqrt(dt);    
mudt = mu*(1-exp(-kappa*dt));

for j=1:NRepl      
   for i=1:NSteps          
      %nPaths(i+1,j)= (nPaths(i,j)-mu)*exp(-kappa*dt)+mu + sigma*sqrt((1-exp(-2*kappa*dt))/(2*kappa))*W1(i,j);
      nPaths(i+1,j)= nPaths(i,j)*exp(-kappa*dt)+ mudt + sigma*sqrt((1-exp(-2*kappa*dt))/(2*kappa))*W1(i,j);
      W2(i,j) = rho*W1(i,j) + sqrt(1-rho^2)*randn; 
      CPIPaths(i+1,j)=CPIPaths(i,j)*exp(nudt + gammadt*W2(i,j));
   end
end

if strcmp(Settings.PlotSims, 'yes')
    FontSize = 15;
    FontWeight = 'normal';
    figure
    xaxis = 1:NSteps+1;
    subplot(1,2,1)
    plot(xaxis, CPIPaths, 'b')    
    hold on
    plot(xaxis, mean(CPIPaths, 2), 'r', 'LineWidth', 2)
    xlim([1 NSteps]);
    set(gca, 'Xtick', [1, floor(NSteps/2), NSteps]);
    set(gca, 'XTickLabel', [0,T/2,T]);
    %legend('CPI')
    %legend('boxoff') 
    set(gca, 'FontSize', FontSize, 'FontName', 'Arial', 'FontWeight', FontWeight);
    title('CPI')
    hold off
    subplot(1,2,2)
    plot(xaxis, nPaths, 'b')
    hold on
    plot(xaxis, mean(nPaths, 2), 'r', 'LineWidth', 2)   
    xlim([1 NSteps]);
    set(gca, 'Xtick', [1, floor(NSteps/2), NSteps]);
    set(gca, 'XTickLabel', [0,T/2,T]);
    set(gca, 'FontSize', FontSize, 'FontName', 'Arial', 'FontWeight', FontWeight);
    title('Short-Rate')
    hold off
end   

% rhoSim = zeros(NRepl,1);
% for i=1:NRepl
%    rhotmp = corrcoef(CPIPaths(:,i), nPaths(:,i));
%    %rhotmp = corrcoef(W1(:,i), W2(:,i));
%    rhoSim(i,1) = rhotmp(1,2);  
% end
% fprintf('\n Average MC rho = %3.4f\n', mean(rhoSim));

end