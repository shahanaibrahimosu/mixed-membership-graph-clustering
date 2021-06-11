function U = ProjGD(X,W,U,opts)
%U = rand(size(X));
%U=U*diag(1./sum(U,1));
epsilon=1/max(eig((W')*(W)));
U_prev=U;
for i=1:opts.max_iter
    U = U_prev- epsilon* ((W')*(W)*U_prev-W'*X);
    %U = ProjectOntoSimplex(U,1);
    U=max(U,0);   
    %U = U';
%     norm(X-W*U')/length(X)
     if((norm(U_prev-U)/length(X))<opts.tol)
         break;
     end
    U_prev=U;

end
end