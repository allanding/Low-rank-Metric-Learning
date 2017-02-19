function [mapping, running_time] = FastJointBayesian(X, labels, theta, num_iter)
% Input: X - d x n training samples; labels - ground truth labels; theta -
% stopping criterion; num_iter - max iterations
%
% Output: mapping - a set including Smu, Sep, G, A; running time - elapse
% time of each iteration

% Make sure labels are nice
[classes, ~, labels] = unique(labels);
% number of classes
nc = length(classes);

% initilize a few variables
n = length(labels);
% dimension of feature
d = size(X,2);
% store the running time of each iteration
running_time = zeros(1, num_iter);
% store the face images of each class
face_class = cell(nc,1);
% a counter
withinCount = 0;
% memory the number of faces in each class, set to 1 if we've got such
% number of faces
numberCount = zeros(1000,1);

for i=1:nc
    % store all instances with class i
    face_class{i} = X(labels == i,:);
    % select classes with at least 2 instaces
    if size(face_class{i},1)>1
        withinCount = withinCount + size(face_class{i},1);
    end;
    if numberCount(size(face_class{i},1)) ==0
        numberCount(size(face_class{i},1)) = 1;
    end;
end;

% inter-class vector set
mu = zeros(d,nc);
% intra-class vector set
ep = zeros(d,withinCount);
% count the number of columns, used later
col_count = 1;

% initialize within- and between- scatter matrices
for i=1:nc
    % update inter-class vector set
    mu(:,i) = mean(face_class{i},1)';
    % update intra-class vector set
    if size(face_class{i},1)>1
        ep(:,col_count:col_count+ size(face_class{i}, 1)-1) = bsxfun(@minus,face_class{i}',mu(:,i));
        col_count = col_count + size(face_class{i}, 1);
    end
end

% between-class scatter
Smu = cov(mu');
% within-class scatter
Sep = cov(ep');

oldSep = Sep;
% temp variables, different from Su and Sw
Smu_temp = cell(1000,1);
Sep_temp = cell(1000,1);

% EM updating
for l=1:num_iter
    tic;
    F = inv(Sep);
    ep = zeros(d,n);
    col_count = 1;
    
    % following update procedure of ECCV2012 paper
    for i = 1:1000
        if numberCount(i)==1
            G = -1 .* (i .* Smu + Sep) \ Smu / Sep;
            Smu_temp{i} = Smu * (F + i.*G);
            Sep_temp{i} = Sep*G;
        end
    end
    for i=1:nc
        nnc = size(face_class{i}, 1);
        mu(:,i) = sum(Smu_temp{nnc} * face_class{i}',2);
        ep(:,col_count:col_count+ size(face_class{i}, 1)-1) = bsxfun(@plus,face_class{i}',sum(Sep_temp{nnc}*face_class{i}',2));
        col_count = col_count+ nnc;
    end
    
    Smu = cov(mu');
    Sep = cov(ep');
    
    running_time(l) = toc;
%     fprintf('iteration-%d: Sep appx error is %f || running time is %fs\n',l,norm(Sep - oldSep)/norm(Sep), running_time(l));
    
    if norm(Sep - oldSep)/norm(Sep)<theta
        break;
    end
    
    oldSep = Sep;
end

% collecting results
F = inv(Sep);
mapping = [];
mapping.G = -1 .* (2 * Smu + Sep) \ Smu / Sep;
mapping.A = inv(Smu + Sep) - (F + mapping.G);
mapping.F = F;
mapping.Sw = Sep;
mapping.Su = Smu;