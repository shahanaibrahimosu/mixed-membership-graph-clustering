function [M] = generate_membership_matrix_dirichlet(N,K,alpha)

% The columns of M are generated using Dirichlet distribution
M = transpose(drchrnd(alpha,N));
K = size(M,1);
M = M*diag(1./sum(M,1));

end
