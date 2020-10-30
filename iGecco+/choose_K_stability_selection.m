function [best_K,min_best_K,select_counter,occur_counter,min_select_counter] = choose_K_stability_selection(X,Y,Z,fs_weight,alpha1_vec,alpha2_vec,alpha3_vec,sampling_method)
%%% Choose alpha using Stability Selection
[p1 n] = size(X); [p2 n] = size(Y); [p3 n] = size(Z);

dist_gower_mat = gower_weighted([X;Y;Z],fs_weight);
dist_gower_mat = squareform(dist_gower_mat); % n by n distance matrix

% Bootstrap samples
B = 10;
KNN = 3; % nearest neignbor to predict validation samples

% Set max number of clusters to consider
max_K = 11;
select_counter = zeros(1,max_K-1); % number of instablity counts for each run
occur_counter = zeros(1,max_K-1);  % number of times each K appears in the boostrap

min_select_counter = zeros(1,max_K-1); % indicator of which K has the minimun instability counts for each run
min_occur_counter = zeros(1,max_K-1); % number of times each K appears in the boostrap

for b = 1:B
    % bootstrap samples
   
    if sampling_method == "bootstrap"
        fold1 = randsample(n,n,true);  % bootstrap pair 1
        fold2 = randsample(n,n,true);  % bootstrap pair 2

        fold1 = unique(fold1); % remove duplicates, faster in computation
        fold2 = unique(fold2);
    elseif sampling_method == "subsample"
        fold1 = datasample(1:n,floor(n*0.9),'Replace',false);
        fold2 = datasample(1:n,floor(n*0.9),'Replace',false);
    end

    n1 = length(fold1);
    n2 = length(fold2);
    
    % make training set
    train_X_1 = X(:,fold1);
    train_X_2 = X(:,fold2);

    train_Y_1 = Y(:,fold1);
    train_Y_2 = Y(:,fold2);

    train_Z_1 = Z(:,fold1);
    train_Z_2 = Z(:,fold2);
    

    % Run SCC on two training set; fix alpha = 1;
    alpha = 1;
    [ class_no_vec_1,class_id_mat_1 ] = igecco_plus_output_all_K_one_alpha( train_X_1,train_Y_1,train_Z_1,fs_weight,alpha,alpha1_vec,alpha2_vec,alpha3_vec);
    [ class_no_vec_2,class_id_mat_2 ] = igecco_plus_output_all_K_one_alpha( train_X_2,train_Y_2,train_Z_2,fs_weight,alpha,alpha1_vec,alpha2_vec,alpha3_vec);
 
    max_K_filter = 20;
    class_no_vec_1_unique = unique(class_no_vec_1);
    class_no_vec_2_unique = unique(class_no_vec_2);
    class_no_vec_1_unique = class_no_vec_1_unique(class_no_vec_1_unique < max_K_filter & class_no_vec_1_unique > 1);
    class_no_vec_2_unique = class_no_vec_2_unique(class_no_vec_2_unique < max_K_filter & class_no_vec_2_unique > 1);
    
    class_id_mat_1_trun = zeros(length(class_no_vec_1_unique),n1);
    class_id_mat_2_trun = zeros(length(class_no_vec_2_unique),n2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pair 1
    for s = 1:length(class_no_vec_1_unique)
        class_id_mat_1_trun(s,:) = class_id_mat_1(min(find(class_no_vec_1 == class_no_vec_1_unique(s))),:);
    end
    
    class_no_vec_1_unique_adjust_single = zeros(1,length(class_no_vec_1_unique));
    for s = 1:length(class_no_vec_1_unique)
        edges = unique(class_id_mat_1_trun(s,:));
        counts = histc(class_id_mat_1_trun(s,:), edges);
        class_no_vec_1_unique_adjust_single(s) = sum(counts>1); % only clusters with more than 1 obs are identified as a cluster
    end
    
    class_no_vec_1_unique_adjust_single_unique = unique(class_no_vec_1_unique_adjust_single);
    class_id_mat_1_trun_single = zeros(length(class_no_vec_1_unique_adjust_single_unique),n1);
    
    for s = 1:length(class_no_vec_1_unique_adjust_single_unique)
        class_id_mat_1_trun_single(s,:) = class_id_mat_1_trun(min(find(class_no_vec_1_unique_adjust_single == class_no_vec_1_unique_adjust_single_unique(s))),:);
    end
    
    class_no_vec_1 = class_no_vec_1_unique_adjust_single_unique;
    class_id_mat_1 = class_id_mat_1_trun_single;
    
    %%%%%%%%%%%%%%%%%%%%%%%%% Pair 2
    for s = 1:length(class_no_vec_2_unique)
        class_id_mat_2_trun(s,:) = class_id_mat_2(min(find(class_no_vec_2 == class_no_vec_2_unique(s))),:);
    end
    
    class_no_vec_2_unique_adjust_single = zeros(1,length(class_no_vec_2_unique));
    for s = 1:length(class_no_vec_2_unique)
        edges = unique(class_id_mat_2_trun(s,:));
        counts = histc(class_id_mat_2_trun(s,:), edges);
        class_no_vec_2_unique_adjust_single(s) = sum(counts>1); % only clusters with more than 1 obs are identified as a cluster
    end
    
    class_no_vec_2_unique_adjust_single_unique = unique(class_no_vec_2_unique_adjust_single);
    class_id_mat_2_trun_single = zeros(length(class_no_vec_2_unique_adjust_single_unique),n2);
    
    for s = 1:length(class_no_vec_2_unique_adjust_single_unique)
        class_id_mat_2_trun_single(s,:) = class_id_mat_2_trun(min(find(class_no_vec_2_unique_adjust_single == class_no_vec_2_unique_adjust_single_unique(s))),:);
    end
    
    class_no_vec_2 = class_no_vec_2_unique_adjust_single_unique;
    class_id_mat_2 = class_id_mat_2_trun_single;    
    
    
    % Find number of class that has overlapping over both runs
    common_K = intersect(class_no_vec_1,class_no_vec_2);
    common_K = common_K(common_K > 1 & common_K < max_K);
    
    % iter number that gives the number of class K
    set1_iter_index = zeros(1,length(common_K));
    set2_iter_index = zeros(1,length(common_K));

    % Find iters and cluster assignment that give overlapping number of
    % class
    % record the clustering label of training
    set1_class_id_K = zeros(length(common_K),n1);
    set2_class_id_K = zeros(length(common_K),n2);

    for k = 1:length(common_K)
        % find iters in SCC runs such that the iters gives the number of class
        % desired
        set1_iter_index(k) = min(find(class_no_vec_1 == common_K(k)));
        set2_iter_index(k) = min(find(class_no_vec_2 == common_K(k)));
        % find the cluster assignment
        set1_class_id_K(k,:) = class_id_mat_1( set1_iter_index(k),:);
        set2_class_id_K(k,:) = class_id_mat_2( set2_iter_index(k),:);

    end

    % get cluster assignment on all obs for two clustering results for each
    % K
    set1_class_id_K_val = zeros(length(common_K),n);
    set2_class_id_K_val = zeros(length(common_K),n);

    % use nearest neighbor
    for s = 1:n
        distance_for_s = dist_gower_mat(s,:);
        
        if ( ismember(s,fold1))
            % if indice is in the training sample, use its cluster label
            set1_class_id_K_val(:,s) = set1_class_id_K(:,ismember(fold1, s));
        else
            % if indice is not in the training sample, find its KNN
            % nearest neighbor in the training set
            distance_for_s_fold1 = distance_for_s;
            distance_for_s_fold1(setdiff(1:n,fold1)) = inf;
            % one nearest nb
            % nearest_nb_index_raw_X_fold1 = find( distance_for_s_fold1 == min(distance_for_s_fold1));
            % set1_class_id_K_val(:,s) = set1_class_id_K(:,find(fold1 == nearest_nb_index_raw_X_fold1));
            [xs, index] = sort(distance_for_s_fold1);
            % mode for each row
            set1_class_id_K_val(:,s) = mode(set1_class_id_K(:,ismember(fold1, index(1:KNN))),2);
        end
        
        if ( ismember(s,fold2))
            set2_class_id_K_val(:,s) = set2_class_id_K(:,ismember(fold2, s));
        else
            distance_for_s_fold2 = distance_for_s;
            distance_for_s_fold2(setdiff(1:n,fold2)) = inf;
            % nearest_nb_index_raw_X_fold2 = find( distance_for_s_fold2 == min(distance_for_s_fold2));
            % set2_class_id_K_val(:,s) = set2_class_id_K(:,find(fold2 == nearest_nb_index_raw_X_fold2));
            [xs, index] = sort(distance_for_s_fold2);
            % mode for each row
            set2_class_id_K_val(:,s) = mode(set2_class_id_K(:,ismember(fold2, index(1:KNN))),2);
        end
        
    end
    


    cluster_accuracy = zeros(1,length(common_K));
    for k = 1:length(common_K)
        % cluster_accuracy(k) = cluster_distance(set1_class_id_K_val(k,:), set2_class_id_K_val(k,:));
        cluster_accuracy(k) = 1 - rand_index(set1_class_id_K_val(k,:), set2_class_id_K_val(k,:),'adjusted');
    end

    % voting rule
    % best clustering result is one which minimize the distance
    min_select_counter(common_K(cluster_accuracy == min(cluster_accuracy))) = min_select_counter(common_K(cluster_accuracy == min(cluster_accuracy))) + 1;
    min_occur_counter(common_K) = min_occur_counter(common_K) + 1;

    % average rule
    select_counter(common_K) = select_counter(common_K) + cluster_accuracy;
    occur_counter(common_K) = occur_counter(common_K) + 1;
    

end




% voting rule
min_best_K = find(min_select_counter == max(min_select_counter));

% average rule
freq_K = find(occur_counter > B/2-1);
best_K = freq_K(find(select_counter(freq_K)./occur_counter(freq_K) == min(select_counter(freq_K)./ occur_counter(freq_K))));




end

