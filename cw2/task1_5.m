%
%
function task1_5(X, Ks)
% Input:
%  X  : M-by-D data matrix (double)
%  Ks : 1-by-L vector (integer) of the numbers of nearest neighbours

for k = Ks
   figure;
   time = tic;
   centres = [];
   for i = 1:k
      centres = cat (1,centres,X(i,:));
   end
   [C, idx, SSE] = my_kMeansClustering(X, k, centres);
   timend = toc(time);
   fname1=sprintf('task1_5_c_%d.mat',k);
   fname2=sprintf('task1_5_idx_%d.mat',k);
   fname3=sprintf('task1_5_sse_%d.mat',k);
   save(fname1,'C');
   save(fname2,'idx');
   save(fname3,'SSE');
   plot(SSE);
   xlabel('Iteration number'); 
   ylabel('SSE');
   title(sprintf('SSE for k=%d \n timetaken %d seconds.',k,timend));
end

end
