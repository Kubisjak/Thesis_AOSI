function [price_call,price_put,pde_sol] = Vecer_asiancontinuous(S0,K,r,vol,T)
% ASIANCONTINUOUS - implementation of Vecer's PDE Method for Asion Option
% Pricing
% use: [price, pde_sol] = asiancontinuous(100, 110, .04, .12, .5)

%Mesh building ------------------------------------------------------------
N = 500;       %number of subintervals in space 
M = 500;       %number of subintervals in time

%more points -> higher precision, but slower

%Xmesh  x
xmin = -1;
xmax = 1;
x = linspace(xmin, xmax, N+1);

%Tspan
t = linspace(0, T, M+1);

m = 0;
% pdepe: Solve initial-boundary value problems for parabolic-elliptic PDEs in 1-D
sol = pdepe(m, @pdef, @pdeic, @pdebc, x, t);
pde_sol = struct('x', x, 't', t, 'u', sol(:,:,1), 'plot', @plot_sol);

%Compute price value ------------------------------------------------------
%Output of the value of the option
X_0 = (1-exp(-r*T))*S0/r/T - exp(-r*T)*K;
x0 = X_0/S0;

X_0_put = exp(-r*T)*K - (1-exp(-r*T))*S0/r/T ;
x0_put = X_0_put/S0;

%pdeval: Evaluate numerical solution of PDE using output of pdepe
uout = pdeval(m,x,sol(M+1,:),x0);
price_call = uout*S0;

uout = pdeval(m,x,sol(M+1,:),x0_put);
price_put = uout*S0;

fprintf( 'The price of Asian Option Call is %8.6f\n\n', price_call);
fprintf( 'The price of Asian Option Put is %8.6f\n\n', price_put);

%Description of PDE -------------------------------------------------------
function [c, f, s] = pdef(x, t, u, DuDx)
c = 1;

    f = 0.5*vol^2*( (1-exp(-r*t))/(r*T) - x )^2*DuDx;
    s = vol^2*((1-exp(-r*t))/(r*T) - x)*DuDx;
    
end

%Initial Condition --------------------------------------------------------
function u0 = pdeic(x) 
u0 = max(x, 0);
end

%Boundary Condition -------------------------------------------------------
function [pl, ql, pr, qr] = pdebc(xl, ul,  xr, ur, t) 
pl = ul;
ql = 0;
pr = ur - xr;
qr = 0;
end

%Plot Function ------------------------------------------------------------
function h = plot_sol
figure('Color',[0.9412 0.9412 0.9412 ]);
surf(pde_sol.x, pde_sol.t, pde_sol.u, 'edgecolor', 'none');
axis([min(pde_sol.x) max(pde_sol.x) min(pde_sol.t) max(pde_sol.t) min(min(pde_sol.u)) max(max(pde_sol.u))]);
xlabel('X');ylabel('t');
end


end
