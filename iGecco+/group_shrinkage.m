function z = group_shrinkage(x, kappa)
    z = pos(1 - kappa/norm(x))*x;
end
