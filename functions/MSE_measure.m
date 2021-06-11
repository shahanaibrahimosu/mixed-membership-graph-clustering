function [MSE, A_est_order]= MSE_measure2(A_est,A_true)
% =====================================
% Input:
% A_est is the estimation of mixing matrix
% A_true is the true mixing matrix
% Output:
% A_est_order is the reordered (in column-wise manner) A_est 
% MSE is mean sqaure error between A_est_order and A_true
%======================================
% a0 = A_est;
[m,k] = size(A_true);
for kk=1:k
    A_true(:,kk)=A_true(:,kk)/norm(A_true(:,kk),2);
    A_est(:,kk)=A_est(:,kk)/max(norm(A_est(:,kk),2),1e-12);
end
IDX = isnan(A_est);
if sum(sum((IDX)))
    A_est(IDX) = 0;
end

[A_est_order]   = nearest(A_est,A_true);
MSE = norm(A_est_order-A_true,'fro')^2/(k);



%%%%%%%%%%%%%%%%%%%%%%%%%
%NEAREST 
% 
%  Nearest(Wnew, Wold) reshuffles order of columns of the n x n matrix Wnew 
%  and changes signs of the columns in the matrix, if need be, 
%  so that the resultant matrix the nearest one (in sense of the Frobenius norm) 
%  to the second input matrix Wold 
%  The function calls the function "maxmatching". 
function [A,M]=nearest(Wnew,Wold)
n=size(Wnew,2); 
W = zeros(n,n);
Sign = zeros(n,n);
for i=1:n 
     for j=1:n 
         [W(i,j), Sign(i,j)]=min([sum((Wnew(:,i)-Wold(:,j)).^2) sum((Wnew(:,i)+Wold(:,j)).^2)]); 
    end 
end
Sign=-sign(Sign-1.5); 
W=max(max(W))-W; 
M=maxmatching(W); 
M=M.*Sign;
A=Wnew*M;

%MAXMATCHING Maximum matching. 
% 
%Maxmatching(W) implements the Kuhn-Munkres algorithm 
%to find the maximum matching in a complete bipartite 
%graph G=(X U Y,X x Y), where X and Y have the same size, n. Elements 
%of matrix W (n x n) contain weights of edges between vertices of X 
%and Y. 
% 
%The result is a permutation matrix M of the size n x n, where 
%(i,j)-entry of M is equal to one, iff x_i is matched with y_j. 
% 
function M=maxmatching(W)
n=size(W,2); M=zeros(n,n); lx=max(W'); ly=zeros(1,n); 
Gl=double((lx'*ones(1,n)+ones(n,1)*ly)==W); 
M=diag(sum(Gl')==1)*Gl*diag(sum(Gl)==1); 
if (sum(sum(M))==0) 
    pom=find(Gl==1); 
    M(pom(1))=1; 
end 
while(sum(sum(M))~=n)  
%1 
    pom=find(sum(M')==0); 
    x=pom(1); S=[x]; T=[]; 
    run=1; y=1; 
    while ((sum(M(:,y))==1)|run) 
%2 
       run=0; 
       if (isempty(setdiff(find(sum(Gl(S,:),1)>0),T))) 
            pom=lx'*ones(1,n)+ones(n,1)*ly-W; 
           alfa=min(min(pom(S,setdiff(1:n,T)))); 
           lx(S)=lx(S)-alfa; ly(T)=ly(T)+alfa; 
           Gl=abs((lx'*ones(1,n)+ones(n,1)*ly)-W)<0.00000001; 
       end 
%3 
       pom=setdiff(find(sum(Gl(S,:),1)>0),T); 
       y=pom(1); 
       if (sum(M(:,y))==1) 
            z=find(M(:,y)==1); 
            S(length(S)+1)=z; 
            T(length(T)+1)=y; 
       end 
    end 
    S=augmentingpath(x,y,Gl,M); 
    M(S(1),S(2))=1; 
    for i=4:2:length(S) 
        M(S(i-1),S(i-2))=0; 
        M(S(i-1),S(i))=1; 
    end 
end 
function S=augmentingpath(x,y,Gl,M) 
n=size(Gl,2); 
cesty=zeros(n,2*n); 
cesty(1,1)=x; uroven=1; pocetcest=1; 
while (ismember(y,cesty(:,2:2:2*n))==0) 
     if (mod(uroven,2)) 
           pom=Gl-M; 
           k=2; 
     else 
           pom=M'; 
           k=1; 
     end 
     novypocetcest=pocetcest; 
     i=1; 
     while (i<=pocetcest) 
           sousedi=find(pom(cesty(i,uroven),:)==1); 
           pridano=0; 
           for j=1:length(sousedi) 
                   if (ismember(sousedi(j),cesty(:,k:2:2*n))==0) 
                          if (pridano==0) 
                                   cesty(i,uroven+1)=sousedi(j); 
                          else 
                                   novypocetcest=novypocetcest+1; 
                                   cesty(novypocetcest,1:uroven+1)=[cesty(i,1:uroven) sousedi(j)]; 
                          end 
                          pridano=pridano+1; 
                   end 
             end 
            if (pridano==0) 
                    novypocetcest=novypocetcest-1; 
                    cesty=[cesty([1:i-1, i+1:n],:);zeros(1,2*n)]; 
                    i=i-1; 
                    pocetcest=pocetcest-1; 
             end 
             i=i+1; 
     end 
     pocetcest=novypocetcest; 
     uroven=uroven+1; 
end 
pom=find(cesty(:,uroven)==y); 
S=cesty(pom(1),1:uroven); 





