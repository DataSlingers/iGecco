function z = group_shrinkage(x, kappa)
    % https://web.stanford.edu/~boyd/papers/admm/group_lasso/group_lasso.html
    z = pos(1 - kappa/norm(x))*x;
end
