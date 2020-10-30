function [w] = knn_weight_gower_weighted_dense(X,gamma,fs_weight,phi)

% Total length of w
% len_w = n*(n-1)/2;

tmp = gower_weighted(X,fs_weight);
        
% calculate w based on distance 'tmp'
% use gaussian kernel
w = gamma *  exp(-phi*tmp);
    
    

end