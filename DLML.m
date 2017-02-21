function [P, Z, E] = DLML(X,P,options)
%% DLML; Discriminative Low-ranl Metrix Leaning

%% Input:
%% X is the training data matrix, the row is the dimension and column is the sample size. 
%% P is the projection matrix, which can be learned using supervised or initialized randomly. 
%% Y is the pre-learned low-dimensional feature, the row is the dimension and column is the sample size. 
%% Sw is the intra-personal covariation matrix.
%% St is the total covariation matrix.
%% options - Struct value in Matlab. The fields in options
%                         that can be set:
%                         gama - the paramter between Sw and St
%                         beta - the paramter on P
%                         alpha - the paramter on E
%                         lambda - the paramter on the supervised
%                                  regularizer 
%                         maxIter - the maximum optimization iteration.
%                         Z_Method - the method to learn Z,
%                                    we define two kinds of methods: sparse and low-rank. 
%                                    (1). When the data is definitely large and each class only
%                                         has several samples, the sparse term can be used. 
%                                    (2). While the data size is not large and each class has many
%                                         samples, the low-rank term is used. 
%% Output:
%% P is the optimized projection matrix.
%% Z is the coefficient matrx.
%% E is the sparse error term.

Y = P'*X;
[n,m] = size(X); %% n is original dimension and m is the size.
np = size(Y,1);  %% np is the low-dimension.


if (~exist('options','var'))
   options = [];
end

gama = 0.7;
if isfield(options,'gama')
    gama = options.gama;
end
Sw = options.Sw;
St = options.St;

Lw = Sw-gama*St; 

alpha = 1e-2;
if isfield(options,'alpha')
    alpha = options.alpha;
end

lambda = 5;
if isfield(options,'lambda')
    lambda = options.lambda;
end

beta = 1e-1;
if isfield(options,'beta')
    beta = options.beta;
end

Z_Method = 'sparse';
if isfield(options,'Z_Method')
    Z_Method = options.Z_Method;
end

maxIter = 1e3;
if isfield(options,'maxIter')
    maxIter = options.maxIter;
end

%% initilize other parameters
max_mu = 10^6;
mu = 1e-4;
rho = 1.2;

Q = zeros(n,np);
Z = zeros(m,m);
J = zeros(m,m);
E = zeros(np,m);

Y1 = zeros(np,m); %%<P'X - YZ - E>
Y2 = zeros(m,m);  %%<Z-J>
Y3 = zeros(n,np); %% <P-Q>


for iter = 1: maxIter
    %% update P
    if(iter > 1)
        if options.optP==1
        P1 = 2*lambda*Lw + mu*X*X' + mu*eye(n);
        P2 = mu*X*(Y*Z-E)' - X*Y1' + mu*Q - Y3;        
        P = P1\P2;
        P = orth(P);
        elseif options.optP==2
            addpath('./FOptM')
            P = optimizingP(P,Y,Lw,X,Q,Z,n,lambda,E,Y1,Y3,mu,1000);
        end
    end
    
    %update Q
    tmp_Q = P + Y3/mu;
    [QU,Qs,QV] = svd(tmp_Q,'econ');
    Qs = diag(Qs);
    svp = length(find(Qs>beta/mu));
    if svp>=1
        Qs = Qs(1:svp)-beta/mu;
    else
        svp = 1;
        Qs = 0;
    end
    Q = QU(:,1:svp)*diag(Qs)*QV(:,1:svp)';


    %%update J
    %% two methods: low-rank and sparse
    tmp_Z = Z + Y2/mu; 
    if strcmp(Z_Method,'low-rank')
        [U,sigma,V] = svd(tmp_Z,'econ');
        sigma = diag(sigma);
        svp = length(find(sigma>1/mu));
        if svp>=1
            sigma = sigma(1:svp)-1/mu;
        else
            svp = 1;
            sigma = 0;
        end
        J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    end
    if strcmp(Z_Method,'sparse')
        J = max(0,tmp_Z - 1/mu)+min(0,tmp_Z + 1/mu);
    end

    
    %% for Z
    Z1 = (2*eye(m) + Y'*Y);
    Z2 = Y'*(P'*X - E) + J + (Y'*Y1 - Y2)/mu;
    Z = Z1\Z2;

    %% for E
    tmp_E = P' * X - Y * Z + Y1/mu;
    E = max(0,tmp_E - alpha/mu)+min(0,tmp_E + alpha/mu);

    
    %% for Y1~Y4, mu
    leq1 = P'*X - Y*Z - E;
    leq2 = Z - J;
    leq3 = P - Q;

    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;
    Y3 = Y3 + mu*leq3;

    mu = min(rho*mu, max_mu);   
    stopALM = norm(leq1,'fro');
    stopALM = max(norm(leq2,'fro'),stopALM);
    stopALM = max(norm(leq3,'fro'),stopALM);
    if stopALM < 1e-3
        break;
    end
end
end
