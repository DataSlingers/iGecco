function [K_optimal,alpha_optimal,act_feature_prev] = choose_paras_BIC(X,Y,Z,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,BIC_type,alpha_list)
%%% Choose K and alpha using BIC
p1 = size(X,1); p2 = size(Y,1); p3 = size(Z,1);
n = size(X,2);

alpha_optimal = 1;
alpha_previous = 10;
K_optimal = 1;
K_previous = 10;
act_feature_prev = 1:(p1+p2+p3);

alpha_store = [];
K_store = [];
no_of_feature_store = [];
stop_search = 0;

while (alpha_optimal ~= alpha_previous || K_optimal ~= K_previous) && (stop_search == 0)
    K_previous = K_optimal;
    alpha_previous = alpha_optimal;

    %%% Search K with alpha fixed
    [class_no_vec,class_id_mat ] = igecco_plus_output_all_K_one_alpha( X,Y,Z,fs_weight,alpha_optimal,alpha1_vec,alpha2_vec,alpha3_vec);
    % act_feature: p_all * iter

    % Calculate BIC
    [ BIC,K_cluster ] = BIC_for_K( X,Y,Z,class_no_vec,class_id_mat,act_feature_prev,BIC_type);
    % plot(K_cluster,BIC)
    K_optimal = K_cluster(find(BIC == min(BIC)));

    
    if K_optimal == K_previous
        stop_search = 1;
    else
        len_alpha = length(alpha_list);

        class_id_mat = zeros(len_alpha,n);
        feature_mat = zeros(len_alpha,p1+p2+p3);

        for i = 1:len_alpha
            [class_id,act_feature] = igecco_plus_output_all_K_and_feature( X,Y,Z,fs_weight,K_optimal,alpha_list(i),alpha1_vec,alpha2_vec,alpha3_vec );
            class_id_mat(i,:) = class_id;
            feature_mat(i,:) = act_feature;   
        end

        %%%%%%%%
        % Calculate BIC
        [BIC] = BIC_for_alpha( X,Y,Z,class_id_mat,feature_mat,K_optimal,alpha_list);

        active_set_count = zeros(1,len_alpha);
        for i = 1:len_alpha
            active_set_count(i) = sum(feature_mat(i,:) ~= 0);
        end
%         plot(active_set_count,BIC)
%         xlabel("Number of Features")
%         ylabel("BIC")

        % Avoid the case if it selects no feature
        BIC = BIC(active_set_count~=0);
        active_set_count = active_set_count(active_set_count~=0);
        feature_mat = feature_mat(active_set_count~=0,:);

        % Calculate BIC
        alpha_optimal = alpha_list(max(find(BIC == min(BIC))))  ;
        act_feature_prev = find(feature_mat(max(find(BIC == min(BIC))),:) ~= 0 );

    end
    
    
    K_store = [K_store K_optimal];
    alpha_store = [alpha_store alpha_optimal];
    no_of_feature_store = [no_of_feature_store length(act_feature_prev)];

    
    %%% stop if has local minima
    if (length(K_store) > 4) && (K_store(end) == K_store(end-2)) && (K_store(end-1) == K_store(end-3))
        K_optimal = K_previous;  % as the optimal K for alpha_optimal is end+1, i.e., end-1
        stop_search = 1;
    end

end








end

