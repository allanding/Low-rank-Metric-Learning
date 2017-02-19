function [P,Z,E] = FastLRMLa(Xs, P, D, options)

Sw = options.Sw;
St = options.St;
[n, m] = size(Xs);
% n is original dimension and m is the size.
np = size(D,1);  %% np is the low-dimension.
r = options.r;

my = size(D,2);%100;
gama = 0.01;
Lw = Sw-gama*St;
alpha = 1e-2;
lambda1 = 1e3;

lambda2 = 1e-2;

maxIter = 2;


%% initilize other parameters
max_mu = 10^6;
mu = 1e-7;
rho = 1.1;

scale = 1e-5;
Z = rand(my,m)*scale;
E = rand(np,m)*scale;

A =  rand(my,r)*scale;
B =  rand(r,m)*scale;

Y1 = rand(np,m)*scale; %%<P'X - YZ - E>
Y2 = rand(my,m)*scale;
for iter = 1: maxIter
    %% update P
    if iter>1
        P1 = 2*lambda1*Lw + mu*Xs*Xs' + mu*eye(n);
        P2 = mu*Xs*(D*Z-E)' - Xs*Y1';
        P = P1\P2;
        P = orth(P);
    end
    
    %% for Z
    Z1 = (2*lambda2*eye(my) + mu*eye(my)+ mu*D'*D);
    Z2 = mu*D'*(P'*Xs - E) +D'*Y1 + mu*A*B + Y2;
    Z = Z1\Z2;
    
    %% for E
    tmp_E = P' * Xs - D * Z + Y1/mu;
    E = max(0,tmp_E - alpha/mu)+min(0,tmp_E + alpha/mu);
    
    %% for A
    A1 = eye(r)+mu*B*B';
    A2 = mu*Z*B'-Y2*B';
    A = A2/A1;
    
    %% for B
    B1 = eye(r)+mu*A'*A;
    B2 = mu*A'*Z-A'*Y2;
    B = B1\B2;
    
    %% for Y1~Y2, mu
    leq1 = P'*Xs - D*Z - E;
    leq2 = A*B-Z;
    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;
    mu = min(rho*mu, max_mu);
    stopALM = norm(leq1,'fro');
    stopALM = max(stopALM,norm(leq2,'fro'));
    if stopALM < 1e-4
        break;
    end
end
end
