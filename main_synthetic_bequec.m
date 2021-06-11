clear
clc
close all
addpath(genpath('functions'))


%% Network Parameters
N_list=[2000,4000, 8000, 10000];
K = 5; % Total number of communities in the network
rng(1);

L =10;
pattern_type =1; % 0 --> block-diagonal
                 % 1 --> randomized blocks
                 % 2 --> control reference node m
num_ref_nodes =4; % Applied when the pattern_type is 2
display_pattern=0;

%% Simulation Parameters
maxitr = 20;

%% Model Parameters
alpha = ones(1,K)/K; % Dirichlet paramater
flag_noise = 0; % 0- ideal case, 1- binary case
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
	        A(A==2)=1;
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
		
		



end
end

