%
%
function task1_4(EVecs)
% Input:
%  Evecs : the same format as in comp_pca.m
%
temp = [];
for i = 1 : 10
   temp = cat(3,temp,reshape(EVecs(:,i),28,28)');
end
montage(temp, 'DisplayRange', [-0.2, 0.2]);
end
