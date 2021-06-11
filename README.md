This code is developed for "Mixed Membership Graph Clustering via Systematic Edge Query" - Shahana Ibrahim, Xiao Fu.         

If you have troubles running the code, or find any bugs, please report to Shahana Ibrahim (ibrahish@oregonstate.edu)
=========================================================================================================================

Main Files : 

main_synthetic_bequec.m
----------------------
-> This script is used to test the proposed method under a non-diagonal EQP pattern (Table II).
-> The flag "display_pattern" can be turned on to display the EQP pattern. 
-> The flag "flag_noise" can be set to 0 for the ideal case and 1 for the binary observation case. 


main_synthetic_all.m
----------------------
-> This script is used to test the proposed method and the baselines under the diagonal EQP pattern (Table III & Table IV).
-> The flag "display_pattern" can be turned on to display the EQP pattern. 
-> The flag "flag_noise" can be set to 0 for the ideal case and 1 for the binary observation case. 
-> The parameter "alpha" can be changed to get different Dirichlet distribution parameters for membership matrix columns. 
-> The parameter "eta" can be changed to get different cluster interaction levels for matrix B; higher the "eta", the more cluster-cluster interaction levels. 


main_amt_exp.m
----------------------
-> This script is used to test the proposed method and the baselines on crowdclustering using dog breed dataset. (Table V).
-> The dataset is the annotations acquired using AMT platforms using a diagonal EQP pattern. More details of the dataset can be found in the paper.
-> The folder "dataset" contains the images of the dogs used in this experiment and the acquired annotation data.



main_amt_exp_semireal.m
----------------------
-> This script is used to test the proposed method and the baselines on crowdclustering using dog breed dataset. (Table VI).
-> The dataset is the annotations acquired using AMT platforms using a diagonal EQP pattern. More details of the dataset can be found in the paper.
-> In the acquired annotations, some annotations are injected with error to increase the total number of errors in the annotations. 
-> The list "error_rate_list" can be changed to test different levels of error rate in the annotations.

main_coauthorship_exp.m
----------------------
-> This script is used to test the proposed method and the baselines on community detection tasks using DBLP and MAG datasets. (Table VII).
-> The datasets DBLP and MAG used is from https://www.cs.utexas.edu/~xmao/coauthorship.html.
-> Diagonal EQP and L=10 are used here.


main_coauthorship_exp_varying_L.m
----------------------
-> This script is used to test the proposed method and the baselines on community detection tasks using DBLP and MAG datasets. (Table IX).
-> The EQP pattern is diagonal.
-> The list "nb_list" can be changed to get different number of node groups (L) in the EQP

