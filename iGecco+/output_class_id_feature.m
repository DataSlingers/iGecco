function [ class_id_adapt,active_set,active_set1,active_set2,active_set3 ] = output_class_id_feature( X,Y,Z,K,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha )
%%% iGecco+, output cluster assignment and features selected
[p1,n] = size(X);[p2,n] = size(Y);[p3,n] = size(Z);

[K_best,phi,w] = select_K_target_gower( [X;Y;Z], fs_weight,@knn_weight_gower_weighted_dense,@knn_weight_gower_concat_3data_weighted_tune,1);
w = w ./ sum(w);

alpha1_vec = alpha1_vec  * alpha; 
alpha2_vec = alpha2_vec  * alpha;
alpha3_vec = alpha3_vec  * alpha;

[U_output_adapt,Z_output_adapt,THETA_output_adapt] = igecco_plus_3type_ARP(X,Y,Z,w,alpha1_vec,alpha2_vec,alpha3_vec,K);

V_round = round(Z_output_adapt,3);



%% Cluster Assignment
[class_id_adapt,iter_cut_adapt] = get_cluster_assignment(V_round,w,n,K);


len_V = size(V_round,3);
if length(unique(class_id_adapt)) == 1
    class_id_mat = zeros(len_V,n);
    class_no_vec = zeros(len_V,1);
    for i = 1:len_V 
        [class_no_vec(i), class_id_mat(i,:)] = group_assign_vertice(V_round(:,:,i),w,n);
    end

    iter_cut_adapt = max(find(class_no_vec > 1));
    class_id_adapt = class_id_mat(iter_cut_adapt,:);
end


edges = unique(class_id_adapt);
counts = histc(class_id_adapt, edges);

%%% force no abrupt fusion or singleton cluster
if length(unique(class_id_adapt)) < K || min(counts) == 1
    V_round = round(Z_output_adapt,0);
    [class_id_adapt_new,iter_cut_adapt_new] = get_cluster_assignment(V_round,w,n,K);
    if length(unique(class_id_adapt_new)) == K
        class_id_adapt = class_id_adapt_new;
        iter_cut_adapt = iter_cut_adapt_new;
    end
end


%%% Feature Selection

%%% Use U for feature selection
U_hat = U_output_adapt(:,:,iter_cut_adapt);
% active_set = find(var(U_hat') > 1e-3);

%% Use theta for feature selection
theta_hat = THETA_output_adapt(:,:,iter_cut_adapt);
theta_vec = vecnorm(theta_hat');
active_set = find(theta_vec > 1e-2);




%%%%%%%%%%%%%%%%%%%%%%%%
active_set1 = intersect(active_set,1:p1);
active_set2 = intersect(active_set,(p1+1):(p1+p2));
active_set3 = intersect(active_set,(p1+p2+1):(p1+p2+p3));

















end