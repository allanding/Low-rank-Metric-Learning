%% This is an example code to run our fast LDA+low-rank metric on LFW deep features

clear
close all

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
    
    %% LDA
    [P, Sw, St] = myLDA(Xs, Ys, dim);
    disp('LDA finished')
    
    options.Sw = Sw;
    options.St = St;
    %% fast low-rank metric 
    P = LRML(Xs, Ys, P, options);
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
