%
%
function task1_1(X, Y)
% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector (unit8)
    %colormap 'gray';
    for i = 1:10
        figure(i);
        k = 0;
        temp = [];
        for j = 1:size(X,1)
          if k == 10
              break;
          end
          if (Y(j) == i-1)
            temp = cat(3,temp,reshape(X(j,:),28,28)');
            k = k +1;
          end
        end
        montage(temp);
    end
    
end
