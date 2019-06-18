%
%
function task2_3(X, Y)
% Input:
%  X : M-by-D data matrix (double)
%  Y : M-by-1 label vector for X (unit8)
figure;
hold on;
xlabel('1st principal component');
ylabel('2nd principal component');
szx = size(X,1);
[EVecs, ~] = comp_pca(X);
corrvec = (EVecs' * X')';
tvec = zeros(szx,2);
for m = 1 : szx
   tvec(m,1) = corrvec(m,1);
   tvec(m,2) = corrvec(m,2);
end
for j = 1:10
   %figure;
   %hold on;
   %scatter(tvec(Y==j-1,1),tvec(Y==j-1,2));
   temparr = tvec((Y == j-1),:);
   x_mean = Mymean(temparr);
   temp = bsxfun(@minus, temparr, x_mean);
   cova = 1/(size(temp,1)) * (temp' * temp);
   th = atan(cova(1,2));
   M = [cos(th),-sin(th);
        sin(th),cos(th)];   
   a=sqrt(max(cova(1,1),cova(2,2))); % horizontal radius
   b=sqrt(min(cova(1,1),cova(2,2))); % vertical radius
   x1=x_mean(1,1); % x0,y0 ellipse centre coordinates
   y1=x_mean(1,2);
   x0 = 0;
   y0 = 0;
   t=-pi:0.01:pi;
   x=x0+a*cos(t);
   y=y0+b*sin(t);
   M = M * [x;y];
   plot(M(1,:)+x1,M(2,:)+y1);
   text(x1,y1,num2str(j-1));
end
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