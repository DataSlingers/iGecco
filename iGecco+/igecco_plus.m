function [class_id_adapt,active_set] = igecco_plus(X,Y,Z,K,target_p)


    %%% Adaptive iGecco+ to choose fusion/feature weights
    [ fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,class_id ] = adaptive_weight( X,Y,Z,K);  
    %%% Adaptive iGecco+ with alpha = 1
    alpha = 1;    
    [ class_id_adapt,active_set ] = output_class_id_feature( X,Y,Z,K,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha);   

    %%% Feature Selection, stop until iGecco+ finds oracle number of features
    while length(active_set) > target_p
        alpha = alpha * 5;
        [ class_id_adapt,active_set ] = output_class_id_feature( X,Y,Z,K,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha);       
    end
    alpha_upper = alpha; 
      

    while length(active_set) < target_p
        alpha = alpha / 2;
        [ class_id_adapt,active_set ] = output_class_id_feature( X,Y,Z,K,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha);        
    end
    alpha_lower = alpha;
    
    while (abs(alpha_upper - alpha_lower)) > 1 && (length(active_set) ~= target_p)
        alpha_mid = (alpha_lower + alpha_upper)/2;
        alpha = alpha_mid;
        [ class_id_adapt,active_set,active_set1,active_set2,active_set3 ] = output_class_id_feature( X,Y,Z,K,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha);        

        if length(active_set) < target_p
            alpha_upper =  alpha;
        elseif length(active_set) > target_p
            alpha_lower =  alpha;
        elseif length(active_set) == target_p
            alpha_upper = alpha_lower;  % stop while loop
        end
        
    end 
    
    
    % force no singleton cluster; feature penalty alpha might be the too large and produce singleton
    
    edges = unique(class_id_adapt);
    counts = histc(class_id_adapt, edges);  % count the size of cluster for each cluster
    
    if min(counts) == 1 % if there is a singleton
        active_set_old = active_set;
        active_set_back = active_set;
        while (length(active_set_back) == length(active_set_old)) && (alpha > 1e-2)  %%% while the number of feature is the same, make alpha smaller; also set stopping rule for alpha small enough
            alpha = alpha / 1.1;
            [ class_id_adapt_back,active_set_back,active_set1_back,active_set2_back,active_set3_back ] = output_class_id_feature( X,Y,Z,K,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha);        
            if length(active_set_back) == length(active_set_old) || length(active_set_back) == target_p
                class_id_adapt = class_id_adapt_back;
                active_set = active_set_back;
                active_set1 = active_set1_back;   
                active_set2 = active_set2_back;    
                active_set3 = active_set3_back;    
            end
        end
    end
    
    
    
    
 


end

% demo code
% load('X_whole_S6.mat')
% load('Y_whole_S6.mat')
% load('Z_whole_S6.mat')
% X = X_whole(:,:,1);
% Y = Y_whole(:,:,1);
% Z = Z_whole(:,:,1);
% [class_id_adapt,active_set] = igecco_plus(X,Y,Z,3,30);



