%
%
function M = task1_2(X, Y)
% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector (unit8)
% Output:
%  M : (K+1)-by-D mean vector matrix (double)
%      Note that M(K+1,:) is the mean vector of X.
Xax = size(X,1);
Yax = size(X,2);
temp = zeros(1,Yax,11);
num = zeros(11,1);
num(11,1) = Xax;
draw = [];
M = [];
  for a = 1:Xax
      for b = 1 : Yax
        temp(1,b,Y(a)+1) = temp(1,b,Y(a)+1) + X(a,b);
        temp(1,b,11) = temp(1,b,11) + X(a,b);
      end
      num(Y(a)+1) = num(Y(a)+1) + 1;
  end
  for x = 1:11
      for y = 1 : Yax
          temp(1,y,x) = temp(1,y,x)/num(x);
          
      end
      draw = cat(3,draw,reshape(temp(:,:,x),28,28)');
  end
  montage(draw);
  for i = 1 : 11
     M = cat(1,M,temp(:,:,i));
  end
  save('task1_2_M.mat','M');
end




