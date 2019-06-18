function [CM, acc] = comp_confmat(Ytrues, Ypreds,K)
% Input:
%   Ytrues : N-by-1 ground truth label vector
%   Ypreds : N-by-1 predicted label vector
% Output:
%   CM : K-by-K confusion matrix, where CM(i,j) is the number of samples whose target is the ith class that was classified as j
%   acc : accuracy (i.e. correct classification rate)
CM = zeros(K,K);
for i=1:size(Ytrues,1)
   CM(Ytrues(i)+1,Ypreds(i)+1) = CM(Ytrues(i)+1,Ypreds(i)+1) +1;
end
correct = sum(diag(CM));
total = sum(sum(CM));
acc=correct/total;
end
