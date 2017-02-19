function P = FastLRML(Xs, Ys, P, options)

%%%%%%%%%%%%%%%%%%
%initialize dict
%%%%%%%%%%%%%%%%%%
Zs = P'*Xs;
D  =  [];
nClass = length(unique(Ys));
wayInit = 'random';
for ci = 1:nClass
    cdat = Zs(:,Ys==ci);
    dict = FDDL_INID(cdat,size(cdat,2),wayInit);
    D = [D dict];
end
options.r  = nClass;
maxIter = 1;
eta = 0.95;

for iter = 1:maxIter
    %% optimize P, Z, E, when fixing D
    [P, Z, E] = FastLRMLa(Xs, P, D, options);
    disp('dic opt!')
    Xu = P'*Xs-E;
    Du = Xu*Z'*(Z*Z');
    Do = D;
    D = eta*D+(1-eta)*Du;
    obj = max(max(D-Do));    
    if obj<1e-3
        break;
    end
end
