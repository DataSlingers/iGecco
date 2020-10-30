function [ fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,class_id ] = adaptive_weight(X,Y,Z,K)
%%% Adaptive iGecco+
% Fit iGecco+ with uniform weights, then get adaptive feature weights from
% U hat

[p n] = size(X);
alpha_1 = 1/ norm(X - mean(X,2),'fro')^2;
alpha_2 = 1/sum(sum(abs(Y - median(Y,2))));
logit_vec = log ( mean(Z,2) ./  (1 -  mean(Z,2)) ) ; 
null_dev_Z = sum(sum( - Z .* logit_vec )) + n * sum(log(  1 + exp( logit_vec) ) )  ; 
null_dev_Z = 2 * null_dev_Z;
alpha_3 = 1 / null_dev_Z;
alpha_scale = min([alpha_1,alpha_2,alpha_3]);
alpha_1 = alpha_1 / alpha_scale;
alpha_2 = alpha_2 / alpha_scale;
alpha_3 = alpha_3 / alpha_scale;


fs_weight = ones(1,size([X;Y;Z],1));
[K_best,phi,w] = select_K_target_gower( [X;Y;Z], fs_weight,@knn_weight_gower_weighted_dense,@knn_weight_gower_concat_3data_weighted_tune,1);
w = w ./ sum(w);

[p1 n] = size(X);
[p2 n] = size(Y);
[p3 n] = size(Z);

pen_alpha = 1;
%%% iGecco+ with uniform weight
[U_output,Z_output] = igecco_plus_3type_ARP_full(X,Y,Z,w,pen_alpha * ones(1,p1),pen_alpha * ones(1,p2),pen_alpha * ones(1,p3),K);

%% Rand Index
V_round = round(Z_output,3);
% V_round = Z_output;
[class_id,iter_cut] = get_cluster_assignment(V_round,w,n,K);
len_V = size(V_round,3);
if length(unique(class_id)) == 1
    class_id_mat = zeros(len_V,n);
    class_no_vec = zeros(len_V,1);
    for i = 1:len_V 
        [class_no_vec(i), class_id_mat(i,:)] = group_assign_vertice(V_round(:,:,i),w,n);
    end

    iter_cut = max(find(class_no_vec > 1));
    class_id = class_id_mat(iter_cut,:);
end



edges = unique(class_id);
counts = histc(class_id, edges);

%%% force no abrupt fusion or singleton cluster
if length(unique(class_id)) < K || min(counts) == 1
    V_round = round(Z_output,0);
    [class_id_new,iter_cut_new] = get_cluster_assignment(V_round,w,n,K);
    if length(unique(class_id_new)) == K
        class_id = class_id_new;
        iter_cut = iter_cut_new;
    end 
end

% disp(sparse_rand2_large);


%% Weighted
U_iter_cut = U_output(:,:,iter_cut); % p-by-n
% X also p-by-n
U1_iter_cut = U_iter_cut(1:p1,:);
U_mat = U1_iter_cut - repmat(mean(X,2),[1 n]);
% Compute alpha_j (pen_p) 
pen_p2 = 1 ./ (0.01 + vecnorm(U_mat,2,2));
% same
lambda = 1;
alpha1_vec = lambda * pen_p2;
w_pen1 = (vecnorm(U_mat,2,2)) ;
w_pen1 = (w_pen1 - min(w_pen1)) / (max(w_pen1)-min(w_pen1));
w_pen1 = w_pen1 /alpha_1;


U2_iter_cut = U_iter_cut( (p1+1):(p1+p2),:);
U_mat = U2_iter_cut - repmat(median(Y,2),[1 n]);
% Compute alpha_j (pen_p) 
pen_p2 = 1 ./ (0.01 + vecnorm(U_mat,2,2));
% same
alpha2_vec = lambda * pen_p2;
w_pen2 = (vecnorm(U_mat,2,2)) ;
w_pen2 = (w_pen2 - min(w_pen2)) / (max(w_pen2)-min(w_pen2));
w_pen2 = w_pen2 /alpha_2;


%%% Chunk 3
X_logit = log ( mean(Z,2) ./  (1 -  mean(Z,2)) ) ; 
U3_iter_cut = U_iter_cut( (p1+p2+1):(p1+p2+p3),:);
U_mat = U3_iter_cut - repmat(X_logit,[1 n]);
% Compute alpha_j (pen_p) 
pen_p2 = 1 ./ (0.01 + vecnorm(U_mat,2,2));
% same
alpha3_vec = lambda * pen_p2;
w_pen3 = (vecnorm(U_mat,2,2));
w_pen3 = (w_pen3 - min(w_pen3)) / (max(w_pen3)-min(w_pen3));

w_pen3 = w_pen3 / alpha_3;

%%
fs_weight = [w_pen1' w_pen2' w_pen3'];


fs_penalty = 1;
alpha1_vec = alpha1_vec / norm(alpha1_vec)  * fs_penalty; 
alpha2_vec = alpha2_vec / norm(alpha2_vec)  * fs_penalty;
alpha3_vec = alpha3_vec / norm(alpha3_vec)  * fs_penalty;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Scale feature penalty by signal
[x,y] = meshgrid(1:n, 1:n);
A = [x(:) y(:)];
A = A(y(:)>x(:),:);

signal1 = max(vecnorm(U1_iter_cut(:,A(:,1)) - U1_iter_cut(:,A(:,2)),2,1)) * sqrt(p1);
signal2 = max(vecnorm(U2_iter_cut(:,A(:,1)) - U2_iter_cut(:,A(:,2)),2,1)) * sqrt(p2);
signal3 = max(vecnorm(U3_iter_cut(:,A(:,1)) - U3_iter_cut(:,A(:,2)),2,1)) * sqrt(p3);


signal_scale = min([signal1 signal2 signal3]);
signal1 = signal1 / signal_scale;
signal2 = signal2 / signal_scale;
signal3 = signal3 / signal_scale;


alpha1_vec = alpha1_vec * signal1; 
alpha2_vec = alpha2_vec * signal2;
alpha3_vec = alpha3_vec * signal3;





end