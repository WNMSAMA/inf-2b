%
%
function task2_5(Xtrain, Ytrain, Xtest, Ytest, epsilon)
% Input:
%  Xtrain : M-by-D training data matrix (double)
%  Ytrain : M-by-1 label vector (unit8) for Xtrain
%  Xtest  : N-by-D test data matrix (double)
%  Ytest  : N-by-1 label vector (unit8) for Xtest
%  epsilon : a scalar variable (double) for covariance regularisation
time = tic();
[Ypreds, Ms, Covs] = run_gaussian_classifiers(Xtrain, Ytrain, Xtest, epsilon);
time = toc(time)
[CM, acc] = comp_confmat(Ytest, Ypreds,10);
save('task2_5_cm.mat','CM');
M10 = Ms(10,:); 
Cov10 = Covs(10,:,:);
save('task2_5_m10.mat','M10');
save('task2_5_cov10.mat','Cov10');
N = size(Xtest,1)
Nerrs = sum(sum(CM)) - sum(diag(CM))
acc = acc 
end
