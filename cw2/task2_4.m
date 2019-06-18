%
%
function [Corrs] = task2_4(Xtrain, Ytrain)
% Input:
%  Xtrain : M-by-D data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for X
% Output:
%  Corrs  : (K+1)-by-1 vector (double) of correlation $r_{12}$ 
%           for each class k = 1,...,K, and the last element holds the
%           correlation for the whole data, i.e. Xtrain.
Corrs = [];
[EVecs, ~] = comp_pca(Xtrain);
vecs = (EVecs' * Xtrain')';
for j = 1 : 10
    temparr = vecs((Ytrain == j-1),:);
    x_mean = Mymean(temparr);
    X = bsxfun(@minus, temparr, x_mean);
    cova = 1/(size(X,1)) * (X' * X);
    Corrs = cat(1,Corrs,cova(1,2)/sqrt(cova(1,1)*cova(2,2)));
end
total = [vecs(:,1),vecs(:,2)];
x_mean = Mymean(total);
X = bsxfun(@minus, total, x_mean);
cova = 1/(size(X,1)) * (X' * X);
Corrs = cat(1,Corrs,cova(1,2)/sqrt(cova(1,1)*cova(2,2)));
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