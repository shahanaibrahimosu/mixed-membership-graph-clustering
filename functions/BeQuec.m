function [M, U_t, U]= SVD_SYS_QUERY_4(G,k,opts)
    %% Intialize the parameters
    L=opts.L;
    block_len=opts.block_len;
    l_array=opts.l_array;
    m_array=opts.m_array;
    U = cell(L,1);

    %% Starting position of the procedure
    T = floor(L/2);
    l = l_array(T);
    m = m_array(T);

    %% Run th algorithm
    C = G{l,m};
    opts.tol = 1e-12; opts.maxit = 1000;
    [U{l},Sigma1,V] = svds(C,k,'largest',opts); 
    U{l} = U{l}(:,1:k)*sqrt(Sigma1(1:k,1:k));
    U_sel = U{l};

    l_ref=l;
    U_ref=U_sel;
    
    
    for i=T+1:L
        l_dash = l_array(i);
        m = m_array(i-1);
        D = [G{m,l} G{m,l_dash}];        
        [~,Sigma2,V23]=svds(D,k,'largest',opts);
        V23 = V23(:,1:k);%*sqrt(Sigma2(1:k,1:k));
        V_l = V23(1:block_len,:);
        V_ldash = V23(block_len+1:end,:); 
        U{l_dash}= V_ldash*pinv(V_l)*U_sel;
        U_sel = U{l_dash};
        l = l_dash;
    end
    l=l_ref;
    U_sel=U_ref;
    for i=T-1:-1:1
        l_dash = l_array(i);
        m = m_array(i);
        D = [G{m,l} G{m,l_dash}];        
        [~,Sigma2,V23]=svds(D,k,'largest',opts);
        V23 = V23(:,1:k);%*sqrt(Sigma2(1:k,1:k));
        V_l = V23(1:block_len,:);
        V_ldash = V23(block_len+1:end,:); 
        U{l_dash}= V_ldash*pinv(V_l)*U_sel;
        U_sel = U{l_dash};
        l = l_dash;
    end
    U_t=[];
    for i=1:L
        U_t = [U_t; U{i}];
    end

    [M,F]=SPA_U(U_t',k);
    U_t=U_t';
end