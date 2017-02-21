
clear
close all
%% This is an example code to run our LDA/fast JB + low-rank metric on LFW deep features



load LFW_a.mat;
nfea = [];
for i_fold = 1:10
    fea{i_fold} = double(fea{i_fold});
    nfea = [nfea, fea{i_fold}];
end

%% dim of PCA feature
dim = 220;
options.ReducedDim = dim;
P = PCA(nfea', options);
for i_fold=1:10
    fea{i_fold} = P'*fea{i_fold};
end
idx = 1:10;
warning off


%% parameters
options.init = 'lda'; %% subspace iniatlization method: jb is joint bayesian; or lda
options.optP = 2; %% optimization method of P: 1 or 2;



for i_fold = 1:10
    disp(i_fold)
    
    Xt = fea{i_fold,1};
    Xt = NormalizeFea(Xt,0);
    Xt = CentralizeFea(Xt,0);
    
    sidx = idx;
    sidx(i_fold)=[];
    Xs = [];
    Ys = [];
    for j = sidx
        Xs = [Xs, fea{j,1}];
        Ys = [Ys; gnd{j,1}];
    end
    Xs = NormalizeFea(Xs,0);
    Xs = CentralizeFea(Xs,0);
    
    if strcmp(options.init,'jb')
        %% JB learning
        %% para for JB learning
        max_iter = 200;
        theta = 1e-6;
        [mapping, running_time] = FastJointBayesian(Xs', Ys, theta, max_iter);
        
        S_mu = mapping.Su;
        S_eps = mapping.Sw;
        G = mapping.G;
        A = mapping.A;
        %% JB based subsapce
        [P, S, ~] = svd(-G);
        P = P(:, 1:dim);
        disp('JB finished')
        options.Sw = S_mu;
        options.St = S_eps;
        
    elseif strcmp(options.init,'lda')
        %% LDA based subsapce
        
        [P, Sw, St] = myLDA(Xs, Ys, dim);
        disp('LDA finished')
        
        options.Sw = Sw;
        options.St = St;
    end
    %% low-rank metric learning
    P = DLML(Xs,P,options);
    disp('Metric Learning finished')
    %% pos samples
    Xt = P'* Xt;
    for i = 1: 300
        y1 = Xt(:, (i-1)*2+1); y2 = Xt(:, (i-1)*2+2);
        scoreMatrix(i,i_fold) = dot(y1,y2)/norm(y1)/norm(y2);
    end
    %% neg samples
    for i = 301: 600
        y1 = Xt(:, (i-1)*2+1); y2 = Xt(:, (i-1)*2+2);
        scoreMatrix(i,i_fold) = dot(y1,y2)/norm(y1)/norm(y2);
    end
end

[~, ~, FAR, VR] = evaluationLFW(-scoreMatrix);
figure()
plot(FAR, VR);
