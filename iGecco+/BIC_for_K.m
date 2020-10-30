function [ BIC,K_cluster ] = BIC_for_K( X,Y,Z,class_no_vec,class_id_mat,act_feature_org,BIC_type)
%%% Calculate BIC for K
n = size(X,2);
p1 = size(X,1);
p2 = size(Y,1);
p3 = size(Z,1);
p_act = length(act_feature_org);

max_K_filter = 20;
class_no_vec_unique = unique(class_no_vec);
class_no_vec_unique = class_no_vec_unique(class_no_vec_unique < max_K_filter & class_no_vec_unique > 1);

class_id_mat_trun = zeros(length(class_no_vec_unique),n);
act_feature_mat_trun = zeros(length(class_no_vec_unique),p1+p2+p3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% get unique
for s = 1:length(class_no_vec_unique)
    class_id_mat_trun(s,:) = class_id_mat(min(find(class_no_vec == class_no_vec_unique(s))),:);
    % act_feature_mat_trun(s,:) = act_feature(min(find(class_no_vec == class_no_vec_unique(s))),:);
end


BIC = zeros(1,length(class_no_vec_unique));
K_cluster = zeros(1,length(class_no_vec_unique));



 for i = 1:length(class_no_vec_unique)
        select_feature1 = intersect(act_feature_org ,1:p1);
        select_feature2 = intersect(act_feature_org,(p1+1):(p1+p2)) - p1;
        select_feature3 = intersect(act_feature_org,(p1+p2+1):(p1+p2+p3)) - (p1+p2);
    
        noise_feature1 = setdiff(1:p1,select_feature1);
        noise_feature2 = setdiff(1:p2,select_feature2);
        noise_feature3 = setdiff(1:p3,select_feature3);
             
        tildeX = zeros(p1, class_no_vec_unique(i));
        for s = 1:class_no_vec_unique(i)
            tildeX(:,s) = mean(X(:,class_id_mat_trun(i,:) == s),2);
        end

        X_hat = zeros(p1,n);
        for q = 1:n
            X_hat(:,q) = tildeX(:,class_id_mat_trun(i,q));
        end

        X_hat(noise_feature1,:) = repmat(tildeX(noise_feature1)',[1 n]);
        
        RSS_X = (norm(X(select_feature1,:) - X_hat(select_feature1,:),'fro')^2);
        Sigma_X = max((norm(X(select_feature1,:) - mean(X(select_feature1,:),2),'fro')^2)/n ,eps);


        %%%%%%%%%%%%%%%%%%%%%%%%%%
        tildeY = zeros(p2, class_no_vec_unique(i));
        for s = 1:class_no_vec_unique(i)
            tildeY(:,s) = median(Y(:,class_id_mat_trun(i,:) == s),2);
        end
        Y_hat = zeros(p2,n);
        for q = 1:n
            Y_hat(:,q) = tildeY(:,class_id_mat_trun(i,q));
        end 
        
        Y_hat(noise_feature2,:) = repmat(tildeY(noise_feature2)',[1 n]);
        
        RSS_Y = sum(sum(abs(Y(select_feature2,:)-Y_hat(select_feature2,:))));
        Sigma_Y = max(sum(sum(abs(Y(select_feature2,:) - median(Y(select_feature2,:),2))))/n,eps);
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tildeZ = zeros(p3, class_no_vec_unique(i));
        for s = 1:class_no_vec_unique(i)
            tildeZ(:,s) = mean(Z(:,class_id_mat_trun(i,:) == s),2);
        end
        tildeZ(tildeZ == 0) = 1e-5;
        tildeZ(tildeZ == 1) = 1 - 1e-5;
        tildeZ = log (tildeZ ./  (1 - tildeZ) );

        Z_hat = zeros(p3,n);
        for q = 1:n
            Z_hat(:,q) = tildeZ(:,class_id_mat_trun(i,q));
        end 

        
        Z_hat(noise_feature3,:) = repmat(tildeZ(noise_feature3)',[1 n]);
        
        RSS_Z = -sum(sum(Z(select_feature3,:) .* Z_hat(select_feature3,:))) + sum(sum(log(1+exp(Z_hat(select_feature3,:)))));
        RSS_Z = 2 * RSS_Z;


        logit_Z = log ( mean(Z,2) ./ (1 - mean(Z,2)))  ;
        logit_Z_mat = repmat(logit_Z, [1 n]);
        
        Sigma_Z = -sum(sum(Z(select_feature3,:) .* logit_Z_mat(select_feature3,:))) + sum(sum(log(1+exp(logit_Z_mat(select_feature3,:)))));
        Sigma_Z = max(2 * Sigma_Z /n , eps);

        K = class_no_vec_unique(i);
        
        if BIC_type == "BIC" 
            BIC(i) = ((RSS_X) / Sigma_X + (RSS_Y) /  Sigma_Y  + (RSS_Z) /  Sigma_Z )+ K * log(n * (p_act));
        elseif BIC_type == "extended_BIC"
            BIC(i) = ((RSS_X) / Sigma_X + (RSS_Y) /  Sigma_Y  + (RSS_Z) /  Sigma_Z )+ 3 * K * log(n * (p_act)); % extended BIC
        end
        
        K_cluster(i) = K;
     
     
     
       

end














end


