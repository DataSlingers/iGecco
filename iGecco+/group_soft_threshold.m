function z = group_soft_threshold(a, kappa)
    tmp = 1 - kappa/norm(a) ;
    z = (tmp >0) * tmp *a;
end