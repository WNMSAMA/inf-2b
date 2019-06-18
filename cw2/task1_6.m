%
%
function task1_6(MAT_ClusterCentres)
% Input:
%  MAT_ClusterCentres : file name of the MAT file that contains cluster centres C.
%       
% 
file = load(MAT_ClusterCentres,'-mat');
centres = cell2mat(struct2cell(file));
draw = [];
centres
for i = 1:size(centres,1)
   draw = cat(3,draw,reshape(centres(i,:),28,28)');

end
montage(draw);
  
end
