%
%
function Dmap = task1_7(MAT_ClusterCentres, MAT_M, MAT_evecs, MAT_evals, posVec, nbins)
% Input:
%  MAT_ClusterCentres: MAT filename of cluster centre matrix
%  MAT_M     : MAT filename of mean vectors of (K+1)-by-D, where K is
%              the number of classes (which is 10 for the MNIST data)
%  MAT_evecs : MAT filename of eigenvector matrix of D-by-D
%  MAT_evals : MAT filename of eigenvalue vector of D-by-1
%  posVec    : 1-by-D vector (double) to specify the position of the plane
%  nbins     : scalar (integer) to specify the number of bins for each PCA axis
% Output
%  Dmap  : nbins-by-nbins matrix (uint8) - each element represents
%	   the cluster number that the point belongs to.
Dmap = [];
file1 = load(MAT_ClusterCentres,'-mat');
centres = cell2mat(struct2cell(file1));
K = size(centres,1);
file2 = load(MAT_M,'-mat');
means = cell2mat(struct2cell(file2));
file3 = load(MAT_evecs,'-mat');
Evecs = cell2mat(struct2cell(file3));
file4 = load(MAT_evals,'-mat');
Evals = cell2mat(struct2cell(file4));
mean = means(11,:);
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
for b = 1:size(gridX,1)
    dists = square_dist(centres,corrvec(b,:));
    [~, idx] = sort(dists);
    Dmap = cat(2,Dmap,idx(1,1));
end
Dmap = reshape(Dmap,nbins,nbins);
figure;
[~,h] = contourf(Xax(:), Yax(:),Dmap);
set(h,'LineColor','none');
colormap(random_colours(K));
save(sprintf('task1_7_dmap_%d.mat',K),'Dmap');
end
function sq_dist = square_dist(U, v)
   sq_dist = sum(bsxfun(@minus, U, v).^2, 2)';
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