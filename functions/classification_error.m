function [error,y_est]= classification_error(M,y)
M0 = ind2vec(y');
k= size(M,1);
y_est = kmeans(M',k);

if(length(y_est) ~= length(M))
    y_est = kmeans(M,k);
    y_est = y_est';
end


M = ind2vec(y_est');
P = Hungarian(-M*M0'); M = P'*M;
 

for i=1:k
    y_est(M(i,:)==1)=i;
end

index=find(y~=0);
y_est=y_est(index);
u=find(y_est~=y(index));
error = length(u)/length(y_est);



end