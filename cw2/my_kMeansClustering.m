%
function [C, idx, SSE] = my_kMeansClustering(X, k, initialCentres, maxIter)
% Input
%   X : N-by-D matrix (double) of input sample data
%   k : scalar (integer) - the number of clusters
%   initialCentres : k-by-D matrix (double) of initial cluster centres
%   maxIter  : scalar (integer) - the maximum number of iterations
% Output
%   C   : k-by-D matrix (double) of cluster centres
%   idx : N-by-1 vector (integer) of cluster index table
%   SSE : (L+1)-by-1 vector (double) of sum-squared-errors

  %% If 'maxIter' argument is not given, we set by default to 500
  if nargin < 4
    maxIter = 500;
  end
  
  %% TO-DO
  n = size(X,1);
  D = zeros(k,n);
  vect = zeros(size(initialCentres));
  C = initialCentres;
  SSE = [];
          
 if k == 1
     ds = square_dist(X,C(1,:));
     SSE = cat(1,SSE,sum(ds));
     C(1,:) = Mymean(X);
     D(1,:) = square_dist(X,C(1,:));
     [Ds, d] = min(D);
     SSE = cat(1,SSE,sum(Ds));
     d = ones(1,size(X,1));
else
   for i  = 1:maxIter+1
       
        for c = 1:k
            D(c,:) = square_dist(X,C(c,:));
        end
        [Ds, d] = min(D);
        SSE = cat(1,SSE,sum(Ds));
        for c = 1:k
            C(c,:) = Mymean(X(d == c,:));
        end
        if (vect == C)
            break;
        end
        vect = C;
        
   end
end
  idx = d';
function sq_dist = square_dist(U, v)
   sq_dist = sum(bsxfun(@minus, U, v).^2, 2)';
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
end

