function [class_id,act_feature] = igecco_plus_output_all_K_and_feature( X,Y,Z,fs_weight,K_optimal,alpha,alpha1_vec,alpha2_vec,alpha3_vec)
%%% Output cluster assignments and selected features for a sequence of gamma 
%%% alpha fixed
%%% Assume we know desired number of clusters

K = K_optimal;

[p1 n] = size(X);

n = size(X,2);
p1 = size(X,1); p2 = size(Y,1); p3 = size(Z,1);

alpha1_vec = alpha1_vec * alpha;
alpha2_vec = alpha2_vec * alpha;
alpha3_vec = alpha3_vec * alpha;


%%% Select K and phi in weights
[K_best,phi,w] = select_K_target_gower( [X;Y;Z], fs_weight,@knn_weight_gower_weighted_dense,@knn_weight_gower_concat_3data_weighted_tune,1);
w = w ./ sum(w);

[U_output_adapt,Z_output_adapt,THETA_output_adapt] = igecco_plus_3type_ARP(X,Y,Z,w,alpha1_vec,alpha2_vec,alpha3_vec,K);


%% Rand Index

V_round = round(Z_output_adapt,3);
len_V = size(V_round,3);



class_id_mat = zeros(len_V,n);
class_no_vec = zeros(len_V,1);
for i = 1:len_V 
    [class_no_vec(i), class_id_mat(i,:)] = group_assign_vertice(V_round(:,:,i),w,n);
end


act_feature = zeros(len_V,p1+p2+p3);
for i = 1:len_V 
    theta_hat = THETA_output_adapt(:,:,i);
    theta_vec = vecnorm(theta_hat');
    act_feature(i,:) = (theta_vec > 1e-2);
    
    if isempty(act_feature(i,:))
        act_feature(i,:) = (theta_vec > 1e-9);
    end
    
    
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_K_filter = 20;
class_no_vec_unique = unique(class_no_vec);
class_no_vec_unique = class_no_vec_unique(class_no_vec_unique < max_K_filter & class_no_vec_unique > 1);

class_id_mat_trun = zeros(length(class_no_vec_unique),n);
act_feature_mat_trun = zeros(length(class_no_vec_unique),p1+p2+p3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% get unique
for s = 1:length(class_no_vec_unique)
    class_id_mat_trun(s,:) = class_id_mat(min(find(class_no_vec == class_no_vec_unique(s))),:);
    act_feature_mat_trun(s,:) = act_feature(min(find(class_no_vec == class_no_vec_unique(s))),:);
end



class_id = class_id_mat_trun( class_no_vec_unique  == K_optimal,:);
act_feature = act_feature_mat_trun( class_no_vec_unique  == K_optimal,:);


if isempty(class_id) % if it does not K_optimal clusters
    index = min(find( abs(class_no_vec_unique - K_optimal) == min(abs(class_no_vec_unique - K_optimal ))));
    class_id = class_id_mat_trun( index,:);
    act_feature = act_feature_mat_trun( index,:);
    
    K_new = max(class_id);
    if K_new > K_optimal
        class_id( ismember(class_id, (K_optimal + 1):K_new )) = 1;  % assign those addtional clusters to cluster 1 to make it become K_optimal classes
    end
    
    
end



end

