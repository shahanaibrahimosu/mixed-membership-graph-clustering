function [ theta, B ] = SPACL( A, K, prune )
%
% SPACL of MMSB inference
% Input: A: Adjacency matrix
%        K: number of clusters
%        prune: an indicator variable for deciding to use Prune algorihm or not. 
%               Set prune=1 if using Prune, and 0 if not.
% Output: theta: node-community matrix, theta_{ij} is the probability node i is in community j
%         B: community-community matrix, B_{ij} is the probability there is
%            an edge between a node in community i and a node in community
%            j. Note that sparisity paprameter rho is absorbed in B
                

% Author: Xueyu Mao
% Email: maoxueyu@gmail.com
% Last Update: June 07, 2020
%
% Reference: Mao, Xueyu, Purnamrita Sarkar, and Deepayan Chakrabarti. "Estimating mixed memberships with sharp eigenvector deviations." Journal of the American Statistical Association just-accepted (2020): 1-24. ArXiv: 1709.00407

[ V, ~ ] = eigs( A, K );

% Prune
I_outlier=[];
if prune
    I_outlier=Prune(V,10,0.75,0.05);
end 

% Find pure nodes set that has pure nodes of all communities
[ pure ] = SPA( V, K, I_outlier );

% Estimate theta and B
theta = V / V(pure,:);
tre = 1e-12;
theta(theta<tre) = 0;
[ theta ] = normalize_row_l1_( theta );
[ B ] = est_B_from_theta_A( theta, A );


end

%% Prune function
function [S,d]=Prune(X,r,q,epsilon)
X = real(X);
if nargin==1
    r = 10;
    q = 0.75;
    epsilon = 0.05;
end

% vnorm=sum(X.^2,2); % if vecnorm is not available
vnorm = vecnorm(X');
S0=find(vnorm>quantile(vnorm,q));

if isempty(S0)
    S = [];
    d = 0;
else
    [~,D]=knnsearch(X,X(S0,:),'K',r);
    d=mean(D,2);
    S=S0(d>quantile(d,1-epsilon));
end

end

%% SPA algorithm
function [ pure ] = SPA( X, K, I_outlier )
% An implementation of SPA algorithm [1]
% Input: X: n by K data matrix (eigenvectors in our setting), n is the number of nodes, K is the number of communities
%        I_outlier: the set of outliers that we do not choose pure nodes from
% Outpiut: pure: set of pure nodes indices

% [1] Gillis, Nicolas, and Stephen A. Vavasis. "Fast and robust recursive algorithmsfor separable nonnegative matrix factorization." IEEE transactions on pattern analysis and machine intelligence 36.4 (2013): 698-714.

pure = [];

% row_norm=sum(X.^2,2); % if vecnorm is not available
row_norm = vecnorm(X').^2';

for i = 1:K
    row_norm(I_outlier) = 0;
    [~,idx_tmp] = max(row_norm);
    pure = [pure idx_tmp];
    
    U(i,:) = X(idx_tmp,:);
    
    for j = 1 : i-1
        U(i,:) = U(i,:) - U(i,:)*(U(j,:)'*U(j,:));
    end
    % Normalize U(:,i)
    U(i,:) = U(i,:)/norm(U(i,:));
    
    row_norm = row_norm - (X*U(i,:)').^2;
end

end

%% Function for normalizing rows to theta to have unit l1 norm, for rows
% which are all zero, do not normalize
function [ theta_l1 ] = normalize_row_l1_( theta )

d_theta = sum(theta,2);
S = find(d_theta>0);
theta_l1 = theta;
theta_l1(S,:) = bsxfun(@times, theta(S,:), 1./(sum(theta(S,:), 2)));

end


%% Estimating B using A and estimated Theta
function [ B_est ] = est_B_from_theta_A( theta_est, A )

theta_T_theta_est = theta_est'*theta_est;
B_est=inv(theta_T_theta_est)*theta_est'*A*theta_est*inv(theta_T_theta_est);

B_est(B_est<0) = 0;
B_est(B_est>1) = 1;

end


