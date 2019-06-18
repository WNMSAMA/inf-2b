function [Ypreds, Ms, Covs] = run_gaussian_classifiers(Xtrain, Ytrain, Xtest, epsilon)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector for Xtrain (uint8)
%   Xtest  : N-by-D test data matrix (double)
%   epsilon : A scalar variable (double) for covariance regularisation
% Output:
%  Ypreds : N-by-1 matrix (uint8) of predicted labels for Xtest
%  Ms     : K-by-D matrix (double) of mean vectors
%  Covs   : K-by-D-by-D 3D array (double) of covariance matrices

%YourCode - Bayes classification with multivariate Gaussian distributions.
szy = size(Xtrain,2);
Ms=[];
Covs = zeros(10,szy,szy);
for i = 1 : 10
    temparr = Xtrain((Ytrain == i - 1),:);
    x_mean = Mymean(temparr);
    Ms = cat(1,Ms,x_mean);
    X = bsxfun(@minus, temparr, x_mean);
    Covs(i,:,:) = (1/(size(X,1)) * (X' * X))+ eye(szy) * epsilon;
end
clear Xspace;
szt = size(Xtest,1);
gauprobs = zeros(szt,10);
for l = 1 : 10
    cprob = size((Ytrain == i - 1),1)/42115;
    gauprob = (-0.5 * (Xtest(1:fix(szt/2),:) - Ms(l,:)) * inv(squeeze(Covs(l,:,:))) * (Xtest(1:fix(szt/2),:)-Ms(l,:))') - 0.5 * logdet(squeeze(Covs(l,:,:))) + log(cprob);
    gauprobs(1:fix(szt/2),l) = diag(gauprob);
    clear gauprob;
    gauprob = (-0.5 * (Xtest(fix(szt/2) : szt,:) - Ms(l,:)) * inv(squeeze(Covs(l,:,:))) * (Xtest(fix(szt/2):szt,:)-Ms(l,:))') - 0.5 * logdet(squeeze(Covs(l,:,:))) + log(cprob);
    gauprobs(fix(szt/2) : szt,l) = diag(gauprob);
    clear gauprob;
end
[~, idx] = sort(gauprobs,2,'descend');
Ypreds = (idx(:,1) - 1);
end
function Mea = Mymean(X)
   Mea = zeros(1,size(X,2));
   for a = 1:size(X,2)
       msum = 0;
       for b = 1:size(X,1)
           msum = msum + X(b,a);
       end
       Mea(1,a) = msum / size(X,1);
   end
end
