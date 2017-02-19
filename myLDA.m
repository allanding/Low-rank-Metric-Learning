function [M, Sb, Sw] = myLDA(fea, label, dim)
% generalized eigen problem, A*V = B*V*D A-S_b B-S_w
class_num = max(label);
feature_dim = size(fea,1);

Sw = zeros(feature_dim,feature_dim);
Sb = zeros(feature_dim,feature_dim);

for i = 1 : class_num
    A = fea(:,label == i);
    m_i = mean(A,2);
    Sw = Sw + (A - repmat(m_i,1,size(A,2)))*(A - repmat(m_i,1,size(A,2)))';
    Sb = Sb + m_i*m_i';
end

Sb = Sb/(class_num-1);
Sw = Sw/(size(dataset,2)-1);

Sb(isnan(Sb)) = 0; Sw(isnan(Sw)) = 0;
Sb(isinf(Sb)) = 0; Sw(isinf(Sw)) = 0;
[eigvec, eigval] = eig(Sw, Sb, 'qz');

eigval(isnan(eigval)) = 0;
eigval(isinf(eigval)) = 0;

[~, ind] = sort(diag(eigval), 'descend');
M = eigvec(:, ind(1:min([dim, size(eigvec, 2)])));
end