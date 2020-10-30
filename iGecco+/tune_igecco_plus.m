function [K_optimal,alpha_optimal,class_id_adapt,active_set] = tune_igecco_plus(X,Y,Z,alpha_list,method,choice)

% X,Y,Z,alpha_list,method(BIC/consens), choice(BIC_type for BIC to choose K
% or sampling method in stability selection


if nargin == 4  % default, use BIC with extended BIC
    [ fs_weight,alpha1_vec,alpha2_vec,alpha3_vec ] = adaptive_weight( X,Y,Z,2);
    [K_optimal,alpha_optimal] = choose_paras_BIC(X,Y,Z,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,"extended_BIC",alpha_list);
    [ class_id_adapt,active_set] = output_class_id_feature( X,Y,Z,K_optimal,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha_optimal);       
end

if nargin == 5
    
    if method == "BIC"  % default for BIC method; use extended BIC
        [ fs_weight,alpha1_vec,alpha2_vec,alpha3_vec ] = adaptive_weight( X,Y,Z,2);
        [K_optimal,alpha_optimal] = choose_paras_BIC(X,Y,Z,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,"extended_BIC",alpha_list);
        [ class_id_adapt,active_set] = output_class_id_feature( X,Y,Z,K_optimal,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha_optimal);   

    end
    
    if method == "stability"  % default for stability selection; use bootstrap + extended BIC
        [ fs_weight,alpha1_vec,alpha2_vec,alpha3_vec ] = adaptive_weight( X,Y,Z,2);
        [K_optimal,~] = choose_K_stability_selection(X,Y,Z,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,'bootstrap');
        %%%%%%%%% BIC for alpha
        [alpha_optimal,active_set,class_id_adapt] = choose_alpha_BIC(X,Y,Z,K_optimal,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha_list);
    end       
end


if nargin == 6
    if method == "stability"
        [ fs_weight,alpha1_vec,alpha2_vec,alpha3_vec ] = adaptive_weight( X,Y,Z,2);
        [K_optimal,~] = choose_K_stability_selection(X,Y,Z,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,choice);
        %%%%%%%%% BIC for alpha
        [alpha_optimal,active_set,class_id_adapt] = choose_alpha_BIC(X,Y,Z,K_optimal,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha_list);

    end


    if method == "BIC"
        [ fs_weight,alpha1_vec,alpha2_vec,alpha3_vec ] = adaptive_weight( X,Y,Z,2);
        [K_optimal,alpha_optimal] = choose_paras_BIC(X,Y,Z,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,choice,alpha_list);
        [ class_id_adapt,active_set] = output_class_id_feature( X,Y,Z,K_optimal,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha_optimal);   

    end    
    
end










end

