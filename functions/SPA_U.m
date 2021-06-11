function [ theta,F] = SPA_U( U, K )


Up = U;
U = U';
F  = zeros(K);
for k=1:K
    [~,i] = max(sum(Up.^2));
    F(:,k) = U(i,:)';
    Q = orth(F(:,1:k)); P = eye(K) - Q*Q';
    Up = P*Up;
end

%B = F'*L*F;
theta = F\U';

% theta = max(eps,theta);
% theta = bsxfun( @times, theta, 1./sum(theta) );
% theta = rand(size(U'));
%theta = theta*diag(1./sum(theta,1));
opts.max_iter=20;
opts.tol=1e-12;
theta = ProjGD(U',F,theta,opts);
indices = find(sum(theta,1)~=0);


theta = max(eps,theta);
theta = bsxfun( @times, theta, 1./sum(theta) );
end