function dd = gower_weighted(X,fs_weight)
% X: p by n

[p n] = size(X);
X_raw = X';
D1 = zeros(n,n);

for i = 1:p
    % d1 = pdist(X_raw(:,i),'cityblock')/max(pdist(X_raw(:,i),'cityblock'));
    d1 = pdist(X_raw(:,i),'cityblock')/max(eps,max(pdist(X_raw(:,i),'cityblock')));
    d1 = d1 *  fs_weight(i);
    D1 = squareform(d1) + D1;
end



dd = squareform(D1)/p ;
