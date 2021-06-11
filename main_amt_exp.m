clear
clc
close all
addpath(genpath('functions'))
addpath datasets

rng(2)
%% error rate
error_rate_list=[0];
%% Pattern features
pattern_type =0; % 0 --> block-diagonal
                 % 1 --> randomized blocks
                 % 2 --> control reference node m
num_ref_nodes =4; % Applied when the pattern_type is 2
display_pattern=0;


%% Get the number of clusters and the nodes
K=5;
N=900;



%% Create the mask
L=15; %No of blocks in NxN graph
mask = zeros(N,N);
b = floor(N/L); 
for i=1:L
    mask((i-1)*b+1:i*b,(i-1)*b+1:i*b)=ones(b,b);
    if(i<L)
        mask((i-1)*b+1:i*b,(i)*b+1:(i+1)*b)=ones(b,b);
        mask((i)*b+1:(i+1)*b,(i-1)*b+1:(i)*b)=ones(b,b);
    end
end

%% Calculate the percent of data
[row,col] = find(mask==1);
indices = [row col];
[~,idx] = unique(sort(indices,2),'rows','stable');
pairs = indices(idx,:);
pairs (pairs(:,1)==pairs(:,2),:)=[];
percent_of_data = 2*length(pairs)/(N*(N-1))

for f=1:length(error_rate_list)

    error_rate= error_rate_list(f);

    %% Load the annotation data
    load('annotation_data_batch1.mat');
    G1=G;
    load('annotation_data_batch2.mat');
    G2=G;
    load('annotation_data_batch3.mat');
    G3=G;
    G_all = G1+G2+G3;
    load('single_label_class_5_ground_truth.mat');
    load('single_label_class_5_B_t.mat');

    A=G_all;
    A = A(1:N,1:N);
    M = ind2vec(ground_truth');
    M = M(:,1:N);
    N = size(A,1);
    y = ground_truth;
    y=y(1:N);
    maxitr=5;

    err_count=0;
    for ii=1:length(pairs)
        i=pairs(ii,1);
        j=pairs(ii,2);
        if(B_t(i,j)~=G_all(i,j))
            err_count=err_count+1;
        end
    end


    for itr = 1:maxitr   
        %% Create Pattern
        if(pattern_type==0)
            l_array = 1:L;
            m_array = 2:L;
        elseif(pattern_type==1)
            l_array = randperm(L);
            m_array = randi(L,1,L-1);
        elseif(pattern_type==2)
            if(num_ref_nodes==1)
                l_array = randperm(L);
                m_array = randi(L)*ones(1,L-1); 
            else
                l_array = randperm(L);
                m_sel = randsample(L,num_ref_nodes, false);
                m_array = randsample(m_sel,L-1,true);
            end
        end

        index_list=[];
        count=1;
        for i=1:L-1
            index_list(count,:) = [l_array(i),m_array(i)];
            count=count+1;
            if(l_array(i)~=m_array(i))
                index_list(count,:) = [m_array(i),l_array(i)];
                count=count+1;
            end
            index_list(count,:)= [m_array(i),l_array(i+1)];
            count=count+1;
            if(l_array(i+1)~=m_array(i))
                index_list(count,:) = [l_array(i+1),m_array(i)];
                count=count+1;
            end
        end

        index_list = unique(index_list,'rows');




        %% Acquire Data using the pattern
        block_len=floor(N/L);
        G = cell(L,L);
        observed_number_of_blocks=length(index_list)
        percent_of_data_observed = observed_number_of_blocks/(L^2)
        no_observed_entries = 0;
        for ii=1:length(index_list)
            l = index_list(ii,1);
            k = index_list(ii,2);
            G{l,k}=A((l-1)*block_len+1:l*block_len,(k-1)*block_len+1:(k)*block_len);
        end
        if(pattern_type==0)
            G{1,1}=zeros(block_len,block_len);
        end





		%% Run Proposed Algorithms
		tic
		opts={};
		opts.L=L;
		opts.block_len=block_len;
		opts.l_array=l_array;
		opts.m_array=m_array;
		[M1, U1, U_cell]=BeQuec(G,K,opts);
		time_spa(f,itr)=toc;
		[err,y_est]=classification_error(real(M1),y(1:length(M1)));
		class_acc_spa(f,itr)=1- err;
		src_spa(f,itr) = getSRC(M1,M);  
		nmi_spa(f,itr)  = getNMI(y,y_est);

		%% Baselines
		% Convex
		tic;
		lambda=0.1;
		for jj=1:length(A)
		 A(jj,jj)=1;
		end
		[L1, S, numIter]=inexact_alm_rpca(A, lambda,1e-7,500);
		A_est = L1;
		idx= spectral_cluster(A_est,K,2);
		M_convex=idx';
		time_convex=toc;
		[err,y_est]=classification_error(real(full(M_convex)),y(1:length(M_convex)));
		class_acc_convex(f,itr)=1- err;
		nmi_convex(f,itr) = getNMI(y,y_est);

		
		% GeoNMF
		nb=L;
		tic
		err=0;
		M_geo=zeros(size(M));
		Pi= eye(K);
		for i=1:nb-1
			J = [G{i,i} G{i,i+1}; G{i,i+1}' G{i+1,i+1}];
			[M_12,~] = GeoNMF(J,K);
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
		[err,y_est] = classification_error(real(M_geo),y(1:length(M_geo)));
		class_acc_geo(f,itr)=1- err;
		src_geo(f,itr) = getSRC(M_geo,M); 
		nmi_geo(f,itr)  = getNMI(y,y_est);
		

		% SPACL
		nb=L;
		tic
		err=0;
		Pi=eye(K);
		M_spacl= zeros(size(M));
		for i=1:nb-1
			J = [G{i,i} G{i,i+1}; G{i,i+1}' G{i+1,i+1}];
			[M_12,~] = SPACL(J,K,1);
			M_12=M_12';
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
		[err,y_est] = classification_error(real(M_spacl),y(1:length(M_spacl)));
		class_acc_spacl(f,itr)=1-err;
		src_spacl(f,itr) = getSRC(M_spacl,M); 
		nmi_spacl(f,itr)  = getNMI(y,y_est);
		
		
		% Spectral Clustering Unnorm
		tic
		err=0;
		Pi=eye(K);
		M_sc= zeros(size(M));
		for i=1:L-1
			J = [G{i,i} G{i,i+1}; G{i,i+1}' G{i+1,i+1}];
			idx= spectral_cluster(J,K,1);
			M_12=idx';
			M_to_permute = M_12(:,1:block_len);
			if(i>1)
				Pi  = Hungarian(-M_to_permute*M_baseline'); 
			end
			M_permuted  = Pi'*M_12;
			M_baseline=M_permuted(:,block_len+1:2*block_len);
			M_sc(:,(i-1)*block_len+1:(i+1)*block_len)=M_permuted;
		end
		time_sc(f,itr)=toc;
		[err,y_est] = classification_error(real(M_sc),y(1:length(M_sc)));
		class_acc_sc(f,itr)=1- err;
		nmi_sc(f,itr)  = getNMI(y,y_est);
 
 
		% Spectral Clustering norm    
		tic
		err=0;
		nb=L;
		Pi=eye(K);
		M_sc1= zeros(size(M));
		for i=1:L-1
			J = [G{i,i} G{i,i+1}; G{i,i+1}' G{i+1,i+1}];
			idx= spectral_cluster(J,K,2);
			M_12=idx';
			M_to_permute = M_12(:,1:block_len);
			if(i>1)
				Pi  = Hungarian(-M_to_permute*M_baseline'); 
			end
			M_permuted  = Pi'*M_12;
			M_baseline=M_permuted(:,block_len+1:2*block_len);
			M_sc1(:,(i-1)*block_len+1:(i+1)*block_len)=M_permuted;
		end
		time_sc1(f,itr)=toc;
		[err,y_est] = classification_error(real(M_sc1),y(1:length(M_sc1)));
		class_acc_sc1(f,itr)=1- err;
		nmi_sc1(f,itr)  = getNMI(y,y_est);


		% k means   
		tic
		err=0;
		Pi=eye(K);
		M_kmeans= zeros(size(M));
		for i=1:L-1
			J = [G{i,i} G{i,i+1}; G{i,i+1}' G{i+1,i+1}];
			sumd_min = Inf;
			for ii = 1:20                    
				[idx_tmp,~,sumd] = kmeans(J,K);
				if sum(sumd) < sumd_min
					sum_min = sum(sumd);
					idx = idx_tmp;
				end
			end
			M_12=ind2vec(idx');
			M_to_permute = M_12(:,1:block_len);
			if(i>1)
				Pi  = Hungarian(-M_to_permute*M_baseline'); 
			end
			M_permuted  = Pi'*M_12;
			M_baseline=M_permuted(:,block_len+1:2*block_len);
			M_kmeans(:,(i-1)*block_len+1:(i+1)*block_len)=M_permuted;
		end
		time_kmeans(f,itr)=toc;
		[err,y_est] = classification_error(real(M_kmeans),y(1:length(M_kmeans))); 
		class_acc_kmeans(f,itr)=1-err;
		nmi_kmeans(f,itr)  = getNMI(y,y_est);

 
 
 
		% NMF-commDetNMF
		tic
		err=0;
		flag=0;
		Pi= eye(K);
		for i=1:nb-1
			J = [G{i,i} G{i,i+1}; G{i,i+1}' G{i+1,i+1}];
			[M_12,~] = commDetNMF(J,K);
			M_12=M_12';
			if(size(M_12,1)~=K)
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
		Pi= eye(K);
		for i=1:L-1
			J = [G{i,i} G{i,i+1}; G{i,i+1}' G{i+1,i+1}];
			H = randperm(2*block_len,K-1);
			Y = full(J(:,H)'*J);
			[M_12,~] = CDMVS(Y);
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
		[err,y_est] =classification_error(real(M_CDMVS),y(1:length(M_CDMVS))); 
		class_acc_cdmvs(f,itr)=1-err;
		nmi_cdmvs(f,itr)  = getNMI(y,y_est);
 


    end
end






