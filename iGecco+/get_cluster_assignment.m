function [class_id,iter_cut_adapt] = get_cluster_assignment(V,w,n,K)
% Get cluster assignment which gives K number of clusters from ARP solution
[~,~,len_V] = size(V);
iter_cut_adapt = [];

% Graph Vertice Method
for k = 1:len_V
    [no_class,class_id] = group_assign_vertice(V(:,:,k),w,n);

    if no_class <= K && k > 1
        iter_cut_adapt = k;
        break;
    end
end


if isempty(iter_cut_adapt)
     [no_class,class_id] = group_assign_vertice(V(:,:,end),w,n);
     iter_cut_adapt = k;
end
















end

