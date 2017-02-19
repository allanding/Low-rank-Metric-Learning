function [vr1, far1, FAR, VR] = evaluationLFW(scoreMatrix)

ms = scoreMatrix(1:300,:);
mms = scoreMatrix(301:600,:);

score = [ms;mms];

minScore = min(min(score));
maxScore = max(max(score));

step = (maxScore-minScore)/10000;
threshold = [minScore:step:maxScore];

numtrial1 = size(mms, 1)*size(mms, 2);
numtrial2 = size(ms, 1)*size(ms, 2);
FAR=[]; fan=[]; 
FRR=[]; frn=[];
VR=[];
for i=1:length(threshold)
    % false acceptance number
    idx = find(mms<=threshold(i));
    fan(i) = length(idx);
    % false reject number
    idx = find(ms>threshold(i));
    frn(i) = length(idx);
    FAR(i) = fan(i)/numtrial1;
    FRR(i) = frn(i)/numtrial2;
    VR(i) = 1.0-FRR(i);
end

for i=1:length(threshold)
    if (FAR(i)>0.01 )
        far = 0.01;
        vr = (far-FAR(i-1))*(VR(i)-VR(i-1))/(FAR(i)-FAR(i-1))+VR(i-1);
        fprintf('VR: %0.2f%% at FAR: %0.2f%%\n', vr*100, far*100);%VR(i)*100, FAR(i)*100);
        far1 = FAR(i)*100;
        vr1 = VR(i)*100;
        break;
    end
end


% LFW 

err=[];
rg=[];
for i=1:length(threshold)
	idx = find(mms<=threshold(i));
	err(i) = length(idx);
	idx = find(ms>threshold(i));
	err(i) = err(i) + length(idx);
	rg(i) = 1.0-err(i)/(size(score, 1)*size(score, 2));
end
%fprintf('Average Recognition Rate: %0.2f%%\n', max(rg)*100);

[t1, idx] = max(rg);
th = threshold(idx);

rg2=[];
for i=1:size(ms,2)
	idx = find(mms(:,i)<=th);
	err(i) = length(idx);
	idx = find(ms(:,i)>th);
	err(i) = err(i) + length(idx);
	rg2(i) = 1.0-err(i)/(size(score, 1));
	fprintf('[%d] Recognition Rate: %0.2f%%\n', i, rg2(i)*100);
end
fprintf('Average Recognition Rate: %0.2f%%+%0.4f\n', mean(rg2)*100, std(rg2));

avg1 = mean(rg2);
std1 = std(rg2);
return