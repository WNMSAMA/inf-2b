function [Ypreds] = run_knn_classifier(Xtrain, Ytrain, Xtest, Ks)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector (uint8) for Xtrain
%   Xtest  : N-by-D test data matrix (double)
%   Ks     : 1-by-L vector (integer) of the numbers of nearest neighbours in Xtrain
% Output:
%   Ypreds : N-by-L matrix (uint8) of predicted labels for Xtest
Ks=sort(Ks);
Ypreds = zeros(size(Xtest,1),size(Ks,2));
XX1 = [];
YY = [];
XX2 = [];
szx = size(Xtest,1);
szy = size(Xtrain,1);
for a = 1 : szy
   YY = cat(1,YY,Xtrain(a,:) * Xtrain(a,:)');
end
for c = 1 : fix(szx/2)
   XX1 = cat(1,XX1,Xtest(c,:) * Xtest(c,:)');
end
YY1 = repmat(YY,1,fix(szx/2));
XX = repmat(XX1,1,szy);
[~,idx] = sort((XX - 2 * Xtest(1:fix(szx/2),:)*Xtrain'+YY1'),2);
idx1 =idx(:,1:Ks(length(Ks)));
clear XX1;
clear YY1;
clear idx;
for d = fix(szx/2) + 1 : szx
   XX2 = cat(1,XX2,Xtest(d,:) * Xtest(d,:)');
end
XX = repmat(XX2,1,szy);
YY2 = repmat(YY,1,(szx-fix(szx/2)));
[~,idx] = sort((XX - 2 * Xtest(fix(szx/2)+1:szx,:)*Xtrain'+YY2'),2);
idx2 = idx(:,1:Ks(length(Ks)));
clear XX;
clear YY;
clear XX2;
clear YY2;
clear idx;
n = 1;
for k = Ks
   idx = cat(1,idx1(:,1:k),idx2(:,1:k));
   for j = 1 : size(Xtest,1)
       Ypreds(j,n) = mode(Ytrain(idx(j,:)));
   end
   n = n +1;
end
end
