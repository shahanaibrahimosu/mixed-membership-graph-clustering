function [M,X] = CDMVS(B)

[k,n] = size(B);

k = k+1;
B = [B;ones(1,n)];
s = sum(B,2);

e = eye(k);

X = eye(k);
loss = 1/prod(s);
for itr = 1:10
    
    for l = 1:k
        
        f = X\e(:,l);
        
        x = linprog(-f, -B', zeros(n,1), s', 1 );
        
        
        if(~isempty(x))
            X(l,:) = x';
        end
        
    end
    
    loss = [ loss, abs(det(X)) ];
    
    if loss(end)-loss(end-1) < eps
        break
    end
    
end

% plot(loss)

d = X'\e(:,k);
X = diag(abs(d))*X;
% M = X*B;
M = max(0,X*B);
M = bsxfun(@times,M,1./sum(M));