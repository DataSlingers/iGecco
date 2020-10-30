function [ phi_best ] = select_phi( X,fs_weight,funname)

% https://www.mathworks.com/matlabcentral/answers/265862-how-to-put-function-as-an-input-to-another-function

[p n] = size(X);

phi_list = [0 1e-5 0.1 0.5 1 2 5];

gamma = 1;
len_phi = length(phi_list);
len_w = n * (n-1) / 2;
w_mat = zeros(len_phi, len_w);
var_w = zeros(1,len_phi);

for i = 1:len_phi
    w = funname(X,gamma,fs_weight,phi_list(i));
    w = w / sum(w);
    var_w(i) = var(w);
end
    
best_index  = find(var_w == max(var_w));

phi_best = phi_list(best_index);


end

