function [ class_no_vec,class_id_mat ] = igecco_plus_output_all_K_one_alpha( X,Y,Z,fs_weight,alpha,alpha1_vec,alpha2_vec,alpha3_vec)
%%% Output cluster assignments for a sequence of gamma 
%%% alpha fixed
%%% We don't know desired number of clusters 

K = 10;  % don't know optimal K, choose maximum K to consider
[p1,n] = size(X);
p1 = size(X,1); p2 = size(Y,1); p3 = size(Z,1);

alpha1_vec = alpha1_vec * alpha;
alpha2_vec = alpha2_vec * alpha;
alpha3_vec = alpha3_vec * alpha;


%%% Select K and phi in weights
[K_best,phi,w] = select_K_target_gower( [X;Y;Z], fs_weight,@knn_weight_gower_weighted_dense,@knn_weight_gower_concat_3data_weighted_tune,1);
w = w ./ sum(w);

[U_output_adapt,Z_output_adapt,THETA_output_adapt] = igecco_plus_3type_ARP_fast(X,Y,Z,w,alpha1_vec,alpha2_vec,alpha3_vec,K);


%% Rand Index

V_round = round(Z_output_adapt,3);

len_V = size(V_round,3);


class_id_mat = zeros(len_V,n);
class_no_vec = zeros(len_V,1);
for i = 1:len_V 
    [class_no_vec(i), class_id_mat(i,:)] = group_assign_vertice(V_round(:,:,i),w,n);
end




end

