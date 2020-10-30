function [alpha_optimal,active_set,class_id_optimal] = choose_alpha_BIC(X,Y,Z,K_optimal,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,alpha_list)
%%% Choose alpha using BIC
p1 = size(X,1); p2 = size(Y,1); p3 = size(Z,1);
n = size(X,2);

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
% plot(active_set_count,BIC)
% xlabel("Number of Features")
% ylabel("BIC")


% Find optimal alpha which minimizes BIC
alpha_optimal = alpha_list(max(find(BIC == min(BIC))))  ;
active_set = find(feature_mat(max(find(BIC == min(BIC))),:) ~= 0 );
class_id_optimal = class_id_mat(max(find(BIC == min(BIC))),:);




end

