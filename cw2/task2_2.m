%
%
function Dmap = task2_2(Xtrain, Ytrain, k, MAT_evecs, MAT_evals, posVec, nbins)
% Input:
%  X   : M-by-D data matrix (double)
%  k   : scalar (integer) - the number of nearest neighbours
%  MAT_evecs : MAT filename of eigenvector matrix of D-by-D
%  MAT_evals : MAT filename of eigenvalue vector of D-by-1
%  posVec    : 1-by-D vector (double) to specity the position of the plane
%  nbins     : scalar (integer) - the number of bins for each PCA axis
% Output:
%  Dmap  : nbins-by-nbins matrix (uint8) - each element represents
%	   the cluster number that the point belongs to.
mean = Mymean(Xtrain);
file3 = load(MAT_evecs,'-mat');
Evecs = cell2mat(struct2cell(file3));
file4 = load(MAT_evals,'-mat');
Evals = cell2mat(struct2cell(file4));
eval1 = Evals(1,1);
eval2 = Evals(2,1);
range1 = mean(1) + 5 * sqrt(eval1);
range2 = mean(1) - 5 * sqrt(eval1);
range3 = mean(2) + 5 * sqrt(eval2);
range4 = mean(2) - 5 * sqrt(eval2);
Xax = linspace(range2,range1,nbins);
Yax = linspace(range4,range3,nbins);
[Xv Yv] = meshgrid(Xax, Yax);
gridX = [Xv(:), Yv(:)];
ngrid = zeros(size(gridX,1),784);
ngrid(:,1:2) = gridX(:,1:2);
corrvec = (inv(Evecs') * ngrid' + posVec')';
half = fix(size(gridX,1)/2);
[Ypreds1] = run_knn_classifier(single(Xtrain), Ytrain, single(corrvec(1:half,:)), [k]);
[Ypreds2] = run_knn_classifier(single(Xtrain), Ytrain, single(corrvec(half+1:size(gridX,1),:)), [k]);
Dmap = reshape(cat(1,Ypreds1,Ypreds2),nbins,nbins);
figure;
[~,h] = contourf(Xax(:), Yax(:), Dmap);
set(h,'LineColor','none');
colormap(random_colours(10));
save(sprintf('task2_2_dmap_%d.mat',k),'Dmap');
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
function Colours = random_colours(n)
    % Written by Bora M. Alper <m.b.alper@sms.ed.ac.uk>, 2019.
    % Generate k distinct (and easily distinguishable) colours:
    %     The trick is to use sin curves with 2*pi/3 phase difference for
    %     R, G, B colour channels, such that the sum of R G B is always 1.
    %     (which kind of* guarantees that the colours have similar 
    %     "lightness" [i.e. none is too bright or too dark than another]).
    %     *: Human vision is more complex than that, see CIELAB colour
    %        space and Helmholtz-Kohlrausch effect. =)
    %% Input
    %  n
    %      The number of random colours requested (must be a scalar).
    %% Output
    %  Colours
    %      n-by-3 matrix of RGB values (i.e. colors are stored at rows).
    
    %% Compute
    % Generate n+1 (instead of n) x values so that the first and the last 
    % colours won't be identical (remember that sin is periodic!) and
    % ignore the last point.
    x = linspace(0, 2*pi, n + 1);
    x = x(1:n); 
    
    Colours = [sin(x)', sin(x - 2*pi/3)', sin(x - 4*pi/3)'];
    
    % Shift upwards by one since negative RGB values are illegal. Divide by
    % two since the allowed range is [0, 1].
    Colours = (Colours + 1) / 2;
end