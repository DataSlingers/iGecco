function [w] = knn_weight_gower_concat_3data_weighted_tune(X,gamma,fs_weight,K,phi)
% Pick 1:K nearest neighbors in distance, for same distance, select
% randomly one

[~,n]  = size(X);
% [p2 n]  = size(Y);

% Total length of w
len_w = n*(n-1)/2;

% KNN method
    
    tmp = gower_weighted(X,fs_weight);        
    
    % Make it a distance matrix
    D = squareform(tmp);
    % Distance matrix with sparse outputs(only contain K nearest neighbor)
    M = zeros(n,n); 

    %%% For each node i, pick its K nearest neighbor
    for i = 1:n
        
        % sort_ind: the distance index from smallest to largest
        [sort_dist, sort_ind] = sort(D(i,:),'ascend');     
        
        %%% Select even values in distance with the Kth one
        even_index = [];
        even_index = find( D(i,:) == D(i,sort_ind(K+1)) ); % select index which equals to the Kth minimum distance
        
        if length(even_index) > 1  % If there are ties
            strict_less_count = sum( D(i,:) < D(i,sort_ind(K+1)));  % Find those who is strict nearest neighbor
            sort_ind(strict_less_count+1:K+1) = randsample( even_index, K+1 - strict_less_count);  % For those who have ties, randomly select some so we have K nearest neigobur
            % update the K-nearest neighbor index
        end
        %%%            
        
        % choose 2 to K+1 since the first(smallest) is 0 , distance with itself D(i,i) is also included 
        M(i,sort_ind(2:K+1)) = D(i,sort_ind(2:K+1));
               
    end
    
    % make M symmetric, since for previous one, i is j's K nearest neighbor
    % does not mean j is i's K nearest neighbor
    for i = 1:n
        for j = 1:n
            if M(i,j) ~= 0
                M(j,i) = M(i,j);
            end
        end
    end
    
    
% get sparse distance vector    
d = squareform(M);

% calculate w based on distance
% use gaussian kernel
w = gamma * (d~=0) .* exp(-phi*d);
    
    

end