function [M] = AnchorFree(B,M0)


[p,n] = size(B);
M = M0;


f = zeros(n,1);


mm = 0;
for ite = 1:10 
    for i = 1:n
        for j = 1:n
            indi = [1:i-1, i+1:n];
            indj = [1:j-1, j+1:n];
            f(j) = real( (-1)^(i+j) * det( M(indj, indi) ));
        end
  
        cvx_begin quiet
            variable x1(n)
            maximize( f'*x1 ) 
            subject to
                B*x1 >= 0
                x1'*B'*ones(p,1) == 1
        cvx_end
        
        f1 = cvx_optval;
        
        cvx_begin quiet
            variable x2(n) 
            minimize( f'*x2 )
            subject to
                B*x2 >= 0
                x2'*B'*ones(p,1) == 1
        cvx_end
        f2 = cvx_optval;
 

        if abs(f1) > abs(f2)
            M(:,i) = x1;
        else
            M(:,i) = x2;
        end    
        cost((ite-1)*n+i) = abs( det( M ) ); mm=mm+1;
    end
    
    if mm>1&& abs(cost(mm)-cost(mm-1))<1e-6
        break;
    end
    
end

