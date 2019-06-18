%
%
function [EVecs, EVals, CumVar, MinDims] = task1_3(X)
% Input:
%  X : M-by-D data matrix (double)
% Output:
%  EVecs, Evals: same as in comp_pca.m
%  CumVar  : D-by-1 vector (double) of cumulative variance
%  MinDims : 4-by-1 vector (integer) of the minimum number of PCA dimensions
%            to cover 70%, 80%, 90%, and 95% of the total variance.
[EVecs, EVals] = comp_pca(X);
CumVar = cumsum(EVals);
plot(CumVar);
MinDims = zeros(4,1);
n = CumVar(size(X,2),1);
MinDims(1,1) = findI(n*0.7,CumVar);
MinDims(2,1) = findI(n*0.8,CumVar);
MinDims(3,1) = findI(n*0.9,CumVar);
MinDims(4,1) = findI(n*0.95,CumVar);
save('task1_3_evecs.mat','EVecs');
save('task1_3_evals.mat','EVals');
save('task1_3_cumvar.mat','CumVar');
save('task1_3_mindims.mat','MinDims');
xlabel('number of dimensions'); 
ylabel('variance');
end
function res = findI(p,sz)
   res = 0;
   for i = 1:size(sz,1)
      if p < sz(i,1)
          res = i;
          break;
      end
   end
end
