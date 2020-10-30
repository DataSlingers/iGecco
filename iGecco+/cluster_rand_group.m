function [ ri ] = cluster_rand_group(group,truth)
%%% calculate adjust rand index from two clusterings
n = length(group);
group = group';

ri = rand_index(truth,group,'adjusted'); % ARI
% ri = rand_index(truth,group); % RI

end