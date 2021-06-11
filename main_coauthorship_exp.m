clear
clc
close all
addpath(genpath('functions'))
rng(1);
load datasets/networks.mat
%% Network Parameters
t_list={[2,3,4,5],[1,3,4,5],[1,2,4,5],[1,2,3,5],[1,2,3,4],[6],[7]};
nb = 10;

%% Simulation Parameters
maxitr = 20;
rho=1;

%% Baselines
f=0;
for t=1:7
    f=f+1;
%% Run the simulation
for itr = 1:maxitr   
    %% MMSB Model    
    A = []; C = [];
    for s = [t_list{t}]
        A = blkdiag(A,adjacency{s});
        C = blkdiag(C,community{s});
    end
    
    
    nodes = find(sum(A,1)>5);
    idx = randperm(length(nodes));
    A = A(nodes(idx),nodes(idx));
    C = C(nodes(idx),:);
    A=A+A';
    A(A==2)=1;
    
    
    M = C';
    N = size(A,1);
    

    k = size(C,2);
    block_len=floor(N/nb);
    N_eff=block_len*nb;
    M=M(:,1:N_eff);
    y = kmeans(M',k);
    
    %% Hiding   
    B = cell(nb,1);
    Y = cell(nb,nb);
    no_observed_entries = 0;
    for i = 1:nb
        B{i,i}=A((i-1)*block_len+1:i*block_len,(i-1)*block_len+1:(i)*block_len);
        if(i>1)
            no_observed_entries=no_observed_entries+numel(B{i,i});
        end
        if(i+1 <= nb)
            B{i,i+1}=A((i-1)*block_len+1:i*block_len,(i)*block_len+1:(i+1)*block_len);
            no_observed_entries=no_observed_entries+2*numel(B{i,i+1});
        end
    end

    
	l_array = 1:nb;
	m_array = 1:nb;    
    
    
	%% Run Proposed Algorithm
	tic
	opts.L=nb;
	opts.block_len=block_len;
	opts.l_array=l_array;
	opts.m_array=m_array;

	[M1, U1, U_cell]=BeQuec(B,k,opts);
	time_prop(f,itr)=toc;
	class_acc_M(f,itr)=1-classification_error(real(M1),y(1:length(M1))); 
	src_M(f,itr) = getSRC(M1,M);
 
	%% Baselines

	%GeoNMF
	tic
	err=0;
	M_geo=zeros(size(M));
	Pi= eye(k);
	for i=1:nb-1
	    J = [B{i,i} B{i,i+1}; B{i,i+1}' B{i+1,i+1}];
	    [M_12,~] = GeoNMF(J,k);
	    M_12=M_12';
	    M_to_permute = M_12(:,1:block_len);
	    if(i>1)
	        Pi  = Hungarian(-M_to_permute*M_baseline'); 
	    end
	    M_permuted  = Pi'*M_12;
	    M_baseline=M_permuted(:,block_len+1:2*block_len);
	    M_geo(:,(i-1)*block_len+1:(i+1)*block_len)=M_permuted;
	end
	time_geo(f,itr)=toc;	
	src_geo(f,itr) = getSRC(M_geo,M);
	class_acc_geo(f,itr)=1-classification_error(real(M_geo),y(1:length(M_geo))); 

	% SPACL
	tic
	err=0;
	Pi=eye(k);
	M_spacl= zeros(size(M));
	for i=1:nb-1
		J = [B{i,i} B{i,i+1}; B{i,i+1}' B{i+1,i+1}];
		[M_12,~] = SPACL(J,k,1);
		M_12=M_12';
		M_12(isnan(M_12))=eps;
		M_12(isinf(M_12))=eps;
		M_12(M_12==0)=eps;
		M_12 = M_12*diag(1./sum(M_12,1));
		M_to_permute = M_12(:,1:block_len);
		if(i>1)
			Pi  = Hungarian(-M_to_permute*M_baseline'); 
		end
		M_permuted  = Pi'*M_12;
		M_baseline=M_permuted(:,block_len+1:2*block_len);
		M_spacl(:,(i-1)*block_len+1:(i+1)*block_len)=M_permuted;
	end
	time_spacl(f,itr)=toc;
	src_spacl(f,itr) = getSRC(real(M_spacl),M);
	class_acc_spacl(f,itr)=1-classification_error(real(M_spacl),y(1:length(M_spacl))); 


 
	% NMF-commDetNMF
	tic
	err=0;
	flag=0;
	Pi= eye(k);
	M_nmf= zeros(size(M));
	for i=1:nb-1
	    J = [B{i,i} B{i,i+1}; B{i,i+1}' B{i+1,i+1}];
	    [M_12,~] = commDetNMF(J,k);
	    M_12=M_12';
	    if(size(M_12,1)~=k)
	        flag=1;
	        break;
	    end
	    M_to_permute = M_12(:,1:block_len);
	    if(i>1)
	        Pi  = Hungarian(-M_to_permute*M_baseline'); 
	    end
	    M_permuted  = Pi'*M_12;
	    M_baseline=M_permuted(:,block_len+1:2*block_len);
	    M_nmf(:,(i-1)*block_len+1:(i+1)*block_len)=M_permuted;
	end
	time_nmf(f,itr)=toc;	
	if(flag==0)
	    src_nmf(f,itr) = getSRC(real(M_nmf),M);
	    [err,y_est]=classification_error(real(M_nmf),y(1:length(M_nmf)));
	    class_acc_nmf(f,itr)=1-err;
	    nmi_nmf(f,itr)  = getNMI(y,y_est);
	else
	    src_nmf(f,itr) = NaN;
	    class_acc_nmf(f,itr)=NaN;
	end   
	
	
	
	
	% CDMVS-community detection
	tic
	err=0;
	Pi= eye(k);
	M_CDMVS= zeros(size(M));
	for i=1:nb-1
	    J = [B{i,i} B{i,i+1}; B{i,i+1}' B{i+1,i+1}];
	    H = randperm(2*block_len,k-1);
	    Y = full(J(:,H)'*J);
	    [M_12,~] = CDMVS(Y);
	    
	    M_12 = M_12(1:k,:);
	    M_to_permute = M_12(:,1:block_len);
	    if(i>1)
	        Pi  = Hungarian(-M_to_permute*M_baseline'); 
	    end
	    M_permuted  = Pi'*M_12;
	    M_baseline=M_permuted(:,block_len+1:2*block_len);
	    M_CDMVS(:,(i-1)*block_len+1:(i+1)*block_len)=M_permuted;
	end
	time_cdmvs(f,itr)=toc;
	src_cdmvs(f,itr) = getSRC(real(M_CDMVS),M);
	class_mse_cdmvs(f,itr)=1-classification_error(real(M_CDMVS),y(1:length(M_CDMVS))); 
	
	




end
end




