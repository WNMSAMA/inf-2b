%
%
function task2_1(Xtrain, Ytrain, Xtest, Ytest, Ks)
% Input:
%  Xtrain : M-by-D training data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for Xtrain
%  Xtest  : N-by-D test data matrix (double)
%  Ytest  : N-by-1 label vector (unit8) for Xtest
%  Ks     : 1-by-L vector (integer) of the numbers of nearest neighbours in Xtrain
time = tic();
[Ypreds] = run_knn_classifier(Xtrain, Ytrain, Xtest, Ks);
t = toc(time)
for i = 1 : size(Ks,2)
   [cm, acc] = comp_confmat(Ytest, Ypreds(:,i),10);
   save(sprintf('task2_1_cm%d.mat',Ks(1,i)),'cm');
   k = Ks(1,i)
   N = size(Ytest,1)
   Nerrs = sum(sum(cm)) - sum(diag(cm))
   acc
end

end
