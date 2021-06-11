function B=flip_matrix(B,p)
    B_t=B;
%     a=1-p-0.01;
%     b=1-p+0.01;
%     r = a + (b-a).*rand(size(B));
    prob_success=1-p;
    %prob_success=r;
    mask = binornd(1,prob_success,size(B));
    B(mask==0)= double(~B(mask==0));

end