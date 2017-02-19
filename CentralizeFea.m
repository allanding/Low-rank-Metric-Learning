function [ Xc, C] = CentralizeFea(X, row)
% If row is true then each row vector is a sample; otherwise each column is
% a sample
%  
[d1, d2] = size(X);
if(row)
    C = mean(X,1);
    Xc = X-repmat(C,[d1,1]);
else
    C = mean(X,2);
    Xc = X-repmat(C,[1,d2]);
end

