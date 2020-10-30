function [ cluster_distance ] = cluster_distance(cluster1,cluster2)

n = length(cluster1);

tmp = 0;

for i = 1:n
    for j = 1:n
        tmp = tmp + abs(sum( cluster1(i) == cluster1(j) ) - sum( cluster2(i) == cluster2(j) ));
    end
end

cluster_distance = tmp;

end
