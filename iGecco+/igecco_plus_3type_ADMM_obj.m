function [obj] = igecco_plus_3type_ADMM_obj(X1,X2,X3,U1,U2,U3,w,A,alpha_1,alpha_2,alpha_3,gamma1_vec,gamma2_vec,gamma3_vec)

[p1,n] = size(X1); [p2,n] = size(X2); [p3,n] = size(X3);


data_fidelity1 = 1/2 *  (norm(X1-U1,'fro')^2);

data_fidelity2 = 1 *  sum(sum(abs(X2-U2)));

data_fidelity3 = -sum(sum(X3 .* U3)) + sum(sum(log(1+exp(U3))));

P = [U1;U2;U3];

center_distances = P(:,A(:,1)) - P(:,A(:,2));

tmp = zeros(1,length(w));


for i = 1:length(w)
    tmp(i) = norm(center_distances(:,i),2);
end

penalty = sum(w .* tmp) ;



%%% Sparsity: X1
Xmean = mean(X1,2);

tmp2 = zeros(1,p1);

for j = 1:p1
    tmp2(j) = norm(U1(j,:)-Xmean(j),2);
end

sparse_penalty1 =  sum(gamma1_vec .* tmp2);


%%% Sparsity: X2
Xmedian = median(X2,2);

tmp2 = zeros(1,p2);

for j = 1:p2
    tmp2(j) = norm(U2(j,:)-Xmedian(j),2);
end

sparse_penalty2 =  sum(gamma2_vec .* tmp2);



%%% Sparsity: X3
Xlogit = log ( mean(X3,2) ./  (1 -  mean(X3,2)) );

tmp2 = zeros(1,p3);

for j = 1:p3
    tmp2(j) = norm(U3(j,:)-Xlogit(j),2);
end

sparse_penalty3 =  sum(gamma3_vec .* tmp2);


%%% 
obj = alpha_1 * data_fidelity1 + alpha_2 * data_fidelity2 + alpha_3 * data_fidelity3 + penalty + sparse_penalty1 + sparse_penalty2 + sparse_penalty3  ;
  
 
end

