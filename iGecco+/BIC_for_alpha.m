function [BIC] = BIC_for_alpha( X,Y,Z,class_id_mat,feature_mat,K_optimal,alpha_list)
%%% Calculate BIC for alpha
n = size(X,2);
p1 = size(X,1);
p2 = size(Y,1);
p3 = size(Z,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BIC = zeros(1,length(alpha_list));

 for i = 1:length(alpha_list)
        select_feature1 = intersect(find(feature_mat(i,:) == 1),1:p1);
        select_feature2 = intersect(find(feature_mat(i,:) == 1),(p1+1):(p1+p2)) - p1;
        select_feature3 = intersect(find(feature_mat(i,:) == 1),(p1+p2+1):(p1+p2+p3)) - (p1+p2);
    
        noise_feature1 = setdiff(1:p1,select_feature1);
        noise_feature2 = setdiff(1:p2,select_feature2);
        noise_feature3 = setdiff(1:p3,select_feature3);
     
        p_act = sum(feature_mat(i,:) == 1);

        tildeX = zeros(p1, K_optimal);
        for s = 1:K_optimal
            tildeX(:,s) = mean(X(:,class_id_mat(i,:) == s),2);
        end

        X_hat = zeros(p1,n);
        for q = 1:n
            X_hat(:,q) = tildeX(:,class_id_mat(i,q));
        end

        X_hat(noise_feature1,:) = repmat(tildeX(noise_feature1)',[1 n]);
        
        RSS_X_feature = sum((X - X_hat).^2,2);
        Sigma_X_feature = sum( (X - mean(X,2)).^2,2)/n;

        %%%%%%%%%%%%%%%%%%%%%%%%%%
        tildeY = zeros(p2, K_optimal);
        for s = 1:K_optimal
            tildeY(:,s) = median(Y(:,class_id_mat(i,:) == s),2);
        end
        Y_hat = zeros(p2,n);
        for q = 1:n
            Y_hat(:,q) = tildeY(:,class_id_mat(i,q));
        end 
        
        Y_hat(noise_feature2,:) = repmat(tildeY(noise_feature2)',[1 n]);
                
        RSS_Y_feature = sum( (abs(Y-Y_hat)),2);
        Sigma_Y_feature = sum( abs(Y-median(Y,2)),2)/n;
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tildeZ = zeros(p3, K_optimal);
        for s = 1:K_optimal
            tildeZ(:,s) = mean(Z(:,class_id_mat(i,:) == s),2);
        end
        tildeZ(tildeZ == 0) = 1e-5;
        tildeZ(tildeZ == 1) = 1 - 1e-5;
        tildeZ = log (tildeZ ./  (1 - tildeZ) );

        Z_hat = zeros(p3,n);
        for q = 1:n
            Z_hat(:,q) = tildeZ(:,class_id_mat(i,q));
        end 

        
        Z_hat(noise_feature3,:) = repmat(tildeZ(noise_feature3)',[1 n]);
        
        logit_Z = log ( mean(Z,2) ./ (1 - mean(Z,2)))  ;
        logit_Z_mat = repmat(logit_Z, [1 n]);
        
        RSS_Z_feature = -sum(Z .* Z_hat,2) + sum(log(1+exp(Z_hat)),2);
        RSS_Z_feature = 2 * RSS_Z_feature;
        Sigma_Z_feature = -sum(Z .* logit_Z_mat,2) + sum(log(1+exp(logit_Z_mat)),2);
        Sigma_Z_feature = 2 * Sigma_Z_feature/n;
     
        BIC(i) = sum(RSS_X_feature ./ Sigma_X_feature) + sum(RSS_Y_feature ./ Sigma_Y_feature) + sum(RSS_Z_feature ./ Sigma_Z_feature) +  K_optimal * p_act * log(n);
     
       

end














end



