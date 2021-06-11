clear
clc
close all
addpath(genpath('functions'))


%% Network Parameters
N_list=[2000,4000, 8000, 10000];
K = 5; % Total number of communities in the network

L =10;
pattern_type =0; % 0 --> block-diagonal
                 % 1 --> randomized blocks
                 % 2 --> control reference node m
num_ref_nodes =4; % Applied when the pattern_type is 2
display_pattern=0;

%% Simulation Parameters
maxitr = 20;

%% Model Parameters
alpha = ones(1,K)/K;
flag_pure_node = 0;
flag_noise = 1;
sparse_factor=0;
eta = 0.1; % diagonal dominance of B

%% Baselines
f=0;
for N = N_list
    f=f+1;
    
    %% Create Pattern
    if(pattern_type==0)
        l_array = 1:L;
        m_array = 1:L;
    elseif(pattern_type==1)
        l_array = randperm(L);
        m_array = randi(L,1,L);
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
    for i=1:L
        index_list(count,:) = [l_array(i),m_array(i)];
        count=count+1;
        if(l_array(i)~=m_array(i))
            index_list(count,:) = [m_array(i),l_array(i)];
            count=count+1;
        end
        if(i<L)
            index_list(count,:)= [m_array(i),l_array(i+1)];
            count=count+1;
            if(l_array(i+1)~=m_array(i))
                index_list(count,:) = [l_array(i+1),m_array(i)];
                count=count+1;
            end
        end
    end

    index_list = unique(index_list,'rows');

    
    %% Display Pattern
    if(display_pattern)
        [X,Y]=meshgrid(1:L+1);
        fig=figure; 
        hold on;
        plot(X,Y,'k');
        plot(Y,X,'k');axis off
        Z = ones(L+1);
        C = ones(L+1,L+1,3);
        color_rgb = [0 0.4470 0.7410];
        for ii=1:length(index_list)
            C(L-index_list(ii,1)+1,index_list(ii,2),:)=color_rgb;
        end
        surface(X,Y,Z,C);
    end
	%% Run the simulation
	for itr = 1:maxitr   
		%% MMSB Model
		B = eta*tril(rand(K,K));
		B = B+ B';
		for kk=1:K
			B(kk,kk) = 0.8+0.2*rand;
		end    

		
		M = generate_membership_matrix_dirichlet(N,K,alpha);
		P = M'*B*M; % Bernouli parameter matrix
		if(~flag_noise)
			A=P;
		else
			A = sparse(binornd(1,P));% Adjacency Matrix
	        A= tril(A)+tril(A)';
		end
		n_list=1:N;
		

	 
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
			G{1,1}=A(1:block_len,1:block_len);
		end
		   
		
		%% Run Proposed Algorithm
		tic
		opts={};
		opts.L=L;
		opts.block_len=block_len;
		opts.l_array=l_array;
		opts.m_array=m_array;
		[M1, U1, U_cell]=BeQuec(G,K,opts);
		time_spa(f,itr)=toc;
		mse_spa(f,itr)= MSE_measure(M1',M')
		rel_err_spa(f,itr)= RMSE_measure(M1',M'); 
		src_spa(f,itr) = getSRC(real(M1),M);
		dist(f,itr) = getSubDistance(M,U1);
		
		
		%% Baselines
		% CommDetNMF
		tic
		err=0;
		flag=0;
		Pi= eye(K);
		nb=L;
		M_nmf=zeros(size(M));
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
			mse_nmf(f,itr)=MSE_measure(real(M_nmf'),M');
			rel_err_nmf(f,itr)= RMSE_measure(real(M_nmf'),M'); 
			src_nmf(f,itr) = getSRC(real(M_nmf),M);
		else
			mse_nmf(f,itr)=NaN; 
			src_nmf(f,itr) = NaN;
		end   


		
		% GeoNMF
		tic
		M_geo=zeros(size(M));
		Pi= eye(K);
		for i=1:L-1
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
		mse_geo(f,itr)= MSE_measure(M_geo',M');
		rel_err_geo(f,itr)= RMSE_measure(M_geo',M'); 
		src_geo(f,itr) = getSRC(real(M_geo),M);
		
		
		% SPACL
		tic
		M_spacl=zeros(size(M));
		Pi= eye(K);
		for i=1:L-1
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
		mse_spacl(f,itr)= MSE_measure(M_spacl',M'); 
		rel_err_spacl(f,itr)= RMSE_measure(M_spacl',M'); 
		time_spacl(f,itr)=toc;
		
    
		% % CDMVS-community detection
		tic
		err=0;
		Pi= eye(K);
		M_CDMVS=zeros(size(M));
		for i=1:L-1
			J = [G{i,i} G{i,i+1}; G{i,i+1}' G{i+1,i+1}];
			sumH = sum(J);
			[sumH_sorted,index]=sort(sumH,'descend');
			H = index(1:K-1);
			%H = randperm(2*block_len,K-1);
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
		mse_cdmvs(f,itr)=MSE_measure(real(M_CDMVS'),M') 
		rel_err_cdmvs(f,itr)= RMSE_measure(M_CDMVS',M');   
		src_cdmvs(f,itr) = getSRC(real(M_CDMVS),M);



end
end

