function [ K_best,phi,w ] = select_K_target_gower( X, fs_weight,dense_funname,tune_funname,G)

[p n] = size(X);
gamma = 1;
phi = select_phi( X,fs_weight,dense_funname);
K_list = [3 4 5 6 7 8 9 10];
len_K = length(K_list);
connected = zeros(1,len_K);

for i = 1:len_K
    [w] = tune_funname(X,gamma,fs_weight, K_list(i),phi);
    W = squareform(w);
    
    %%
     g = digraph(W);
     bins = conncomp(g, 'Type', 'weak');
     % disp(max(bins));
     connected(i) = all(bins <= G);
    % The weight matrix has graph which has G separate components
    % G = 1 means we have fully connected graph
end

index_K = min(find(connected == 1));    
% index_K = find(connected,1,'first');
% https://www.mathworks.com/help/matlab/ref/find.html

K_best = K_list(index_K);
if isempty(K_best)
    K_best = 3;
    disp("max K bigger than 10");
end
[w] = tune_funname(X,gamma, fs_weight, K_best,phi);

end

