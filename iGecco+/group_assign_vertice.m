function [ no_class,class_id ] = group_assign_vertice( V,w,n )
% Calculate cluster assignment from a sequence of ARP solution V
[x,y] = meshgrid(1:n, 1:n);
A = [x(:) y(:)];

A = A(y(:)>x(:),:);
A_whole = A;
w_whole = w;

active = find(w~=0);
A = A(active,:);

tmp = A(sum(abs(V),1) ==0,:);


% Construct graph with connected edges
G = graph(tmp(:,1)',tmp(:,2)',[],n);
bins = conncomp(G);

class_id = bins;
no_class = max(class_id);


if isempty(tmp)
    class_id = 1:n;
    no_class = n;
end

    


end

