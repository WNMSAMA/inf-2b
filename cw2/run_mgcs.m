function [Ypreds, MMs, MCovs] = run_mgcs(Xtrain, Ytrain, Xtest, epsilon, L)
% Input:
%   Xtrain : M-by-D training data matrix (double)
%   Ytrain : M-by-1 label vector for Xtrain (uint8)
%   Xtest  : N-by-D test data matrix (double)
%   epsilon : A scalar parameter for regularisation (double)
%   L      : scalar (integer) of the number of Gaussian distributions per class
% Output:
%  Ypreds : N-by-1 matrix of predicted labels for Xtest (integer)
%  MMs     : (L*K)-by-D matrix of mean vectors (double)
%  MCovs   : (L*K)-by-D-by-D 3D array of covariance matrices (double)
szx = size(Xtrain,1);
szy = size(Xtrain,2);
Xspace = zeros(szx,szy,10);
inds = ones(10,1);
MMs = [];
MCovs = zeros(L*10,szy,szy);
for i = 1 : szx
    n = inds(Ytrain(i,1)+1,1);
    Xspace(n,:,Ytrain(i,1)+1) = Xtrain(i,:);
    inds(Ytrain(i,1)+1,1) = n + 1;
end
inds = inds -1;
n = 1;
ind = ones(L*10,1);
for j = 1 : 10
   sizeX =size( Xspace(1:inds(j),:,j),1);
   [~, idx, ~] = my_kMeansClustering(Xspace(1:inds(j),:,j), L, Xspace(1:L,:,j));
   Cspace = zeros(sizeX,szy,L);
   for k = 1 : sizeX
       l = ind(n + idx(k) -1);
       Cspace(l,:,idx(k)) = Xspace(k,:,j);
       ind(n + idx(k) -1) = l + 1;
   end
   for m = 1 : L
    temparr = Cspace(1:ind(m,1),:,m);
    x_mean = Mymean(temparr);
    MMs = cat(1,MMs,x_mean);
    X = bsxfun(@minus, temparr, x_mean);
    cova = (1/(size(X,1)) * (X' * X))+ eye(szy) * epsilon;
    MCovs(n,:,:) = cova;
    n = n + 1;
   end
end
clear inds;
ind = ind - 1;
clear sizeX;
clear Cspace;
gauprobs = zeros(size(Xtest,1),10 * L);
for l = 1 : 10*L
    cprob = ind(l) / 42115;
    gauprob = (-0.5 * (Xtest - MMs(l,:)) * inv(squeeze(MCovs(l,:,:))) * (Xtest-MMs(l,:))') - 0.5 * logdet(squeeze(MCovs(l,:,:))) + log(cprob);
    gauprobs(:,l) = diag(gauprob);
    clear gauprob;
end
[~, idx] = sort(gauprobs,2,'descend');
Ypreds = (fix((idx(:,1) - 1)/L));
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

