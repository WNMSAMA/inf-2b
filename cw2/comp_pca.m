function [EVecs, EVals] = comp_pca(X)
% Input: 
%   X:  N x D matrix (double)
% Output: 
%   EVecs: D-by-D matrix (double) contains all eigenvectors as columns
%       NB: follow the Task 1.3 specifications on eigenvectors.
%   EVals:
%       Eigenvalues in descending order, D x 1 vector (double)
%   (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)
  %% TO-DO
    x_mean = Mymean(X);
    X = bsxfun(@minus, X, x_mean);
    cova = 1/(size(X,1)) * (X' * X);
    [V,D] = eig(cova);
    [d,ind] = sort(diag(D),'descend');
    Vs = V(:,ind);
    for j = 1:size(Vs,1)
       if Vs(1,j) <= 0
           Vs(:,j)=Vs(:,j)*(-1);
       end
    end
    
    
    EVecs = Vs;
    EVals = d;

end
function Mea = Mymean(X)
   Mea = zeros(1,size(X,2));
   for a = 1:size(X,2)
       sum = 0;
       for b = 1:size(X,1)
           sum = sum + X(b,a);
       end
       Mea(1,a) = sum / size(X,1);
   end
end

