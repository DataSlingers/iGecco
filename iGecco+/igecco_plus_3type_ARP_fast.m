function [U_output,Z_output,THETA_output] = igecco_plus_3type_ARP_fast(X1,X2,X3,w,lambda1_vec,lambda2_vec,lambda3_vec,target_K)

[p1,n] = size(X1);[p2,n] = size(X2);[p3,n] = size(X3);

alpha_1 = 2/ norm(X1 - mean(X1,2),'fro')^2;

alpha_2 = 1/sum(sum(abs(X2 - median(X2,2))));

logit_vec = log ( mean(X3,2) ./  (1 -  mean(X3,2)+eps) + eps) ; 

null_dev_Z = sum(sum( - X3 .* logit_vec )) + n * sum(log(  1 + exp( logit_vec) ) )  ; 

alpha_3 = 1 / null_dev_Z;


alpha_scale = min([alpha_1,alpha_2,alpha_3]);
alpha_1 = alpha_1 / alpha_scale;
alpha_2 = alpha_2 / alpha_scale;
alpha_3 = alpha_3 / alpha_scale;

[x,y] = meshgrid(1:n, 1:n);
A = [x(:) y(:)];

A = A(y(:)>x(:),:);
A_whole = A;
w_whole = w;

active = find(w~=0);
A = A(active,:);

[len_l,~] = size(A);

% Remove Redundant edges
w = w(w~=0);

l1_mat_org = zeros(len_l,n);
l2_mat_org = zeros(len_l,n);

for i = 1:n
    l1_mat_org(:,i) = (A(:,1) == i);
end

for i = 1:n
    l2_mat_org(:,i) = (A(:,2) == i);
end


D = l1_mat_org - l2_mat_org;
D = D';

% Stroage of Output
MAX_ITER  = 500;

U1 = zeros(p1,n);
U2 = zeros(p2,n);
U3 = zeros(p3,n);

%%% Non-Differentiable Loss
R = zeros(p2,n);
KSI = zeros(p2,n);

%%% Fusion
Z1 = zeros(p1,len_l);
Z2 = zeros(p2,len_l);
Z3 = zeros(p3,len_l);
Z = zeros(p1+p2+p3,len_l);

Lambda1 = zeros(p1,len_l);
Lambda2 = zeros(p2,len_l);
Lambda3 = zeros(p3,len_l);

%%% Variable Selection
THETA1 = zeros(p1,n);
THETA2 = zeros(p2,n);
THETA3 = zeros(p3,n);


ETA1 = zeros(p1,n); 
ETA2 = zeros(p2,n); 
ETA3 = zeros(p3,n); 

%%%
M1 = inv(alpha_1 * eye(n) + D * D' + eye(n));
M2 = inv( 2*eye(n) + D * D');
M3 = inv(alpha_3 * 1/4 * eye(n) + D * D' + eye(n));


gamma = 0.1;
pen_t = 1.1;
k = 1;


U_output = rand(p1+p2+p3,n,MAX_ITER);
Z_output = rand(p1+p2+p3,len_l,MAX_ITER);
U_output(:,:,1) = [X1 ; X2 ; X3];
Z_output(:,:,1) = rand(p1+p2+p3,len_l);
THETA_output = zeros(p1+p2+p3,n,MAX_ITER);


%%% Loss-specific center
X_mean = mean(X1,2);
X_median = median(X2,2);
X_logit = log ( mean(X3,2) ./  (1 -  mean(X3,2)+eps) +eps) ; 

tildeX1 = repmat(X_mean, [1 n]);
tildeX2 = repmat(X_median, [1 n]);
tildeX3 = repmat(X_logit, [1 n]);



while norm(Z_output(:,:,k)) ~= 0
   
    Z_output(:,:,1) = zeros(p1+p2+p3,len_l);
           
    % Fast solve
    if k < 5
        MAX_ITER_INNER = 5;
    else
        MAX_ITER_INNER = 2;
    end
    
    for m = 1:MAX_ITER_INNER
        
        % U1 update
        U1 = (alpha_1 * X1 + (Z1+Lambda1) * D' + tildeX1 + THETA1 - ETA1 ) * M1;
        
        % U2 update
        U2 = ( X2  - R + KSI + (Z2+Lambda2) * D'  + tildeX2 + THETA2 - ETA2 ) * M2;
        
        % R update
        R = shrinkage(X2-U2+KSI,alpha_2);
        
        % KSI upate
        KSI = KSI + X2 - U2 - R;
        
        % U3 update
        U3 = (1/4* alpha_3 * U3 + alpha_3 * X3 - alpha_3 * exp(U3)./(1+exp(U3))+ (Z3+Lambda3) * D' + tildeX3 + THETA3 - ETA3 ) * M3;
        
        % THETA Update
        for j = 1:p1
            THETA1(j,:) = group_shrinkage( U1(j,:) - tildeX1(j,:)  + ETA1(j,:),lambda1_vec(j));
        end
        
        for j = 1:p2
            THETA2(j,:) = group_shrinkage( U2(j,:) - tildeX2(j,:)  + ETA2(j,:),lambda2_vec(j));
        end
        
        for j = 1:p3
            THETA3(j,:) = group_shrinkage( U3(j,:) - tildeX3(j,:)  + ETA3(j,:),lambda3_vec(j));
        end
        
        % ETA Update
        ETA1 = ETA1 + U1 - tildeX1 - THETA1;
        ETA2 = ETA2 + U2 - tildeX2 - THETA2;
        ETA3 = ETA3 + U3 - tildeX3 - THETA3;
        
        
        % Z-update and Lambda-update:
        for l = 1:len_l
            tmp = A(l,1);
            tmp2 = A(l,2);
            
            Z1(:,l) = U1(:,tmp) - U1(:,tmp2) - Lambda1(:,l);
            Z2(:,l) = U2(:,tmp) - U2(:,tmp2) - Lambda2(:,l);
            Z3(:,l) = U3(:,tmp) - U3(:,tmp2) - Lambda3(:,l);
            
            Z(:,l) = group_soft_threshold([Z1(:,l);Z2(:,l);Z3(:,l)], gamma * w(l));
            
            Z1(:,l) = Z(1:p1,l);
            Z2(:,l) = Z((p1+1):(p1+p2),l);
            Z3(:,l) = Z((p1+p2+1):(p1+p2+p3),l);
            
            Lambda1(:,l) = Lambda1(:,l) + (Z1(:,l) - U1(:,tmp) + U1(:,tmp2));
            Lambda2(:,l) = Lambda2(:,l) + (Z2(:,l) - U2(:,tmp) + U2(:,tmp2));
            Lambda3(:,l) = Lambda3(:,l) + (Z3(:,l) - U3(:,tmp) + U3(:,tmp2));
            
        end
        
    end
    
    % Get Cluster Assignment
    [no_class,class_id] = group_assign_vertice(Z,w_whole,n);
   
    
    if no_class == target_K
        U_output(:,:,k+1) = [U1;U2;U3];
        Z_output(:,:,k+1) = Z;
        THETA_output(:,:,k+1) = [THETA1;THETA2;THETA3];
        k = k + 1;
        gamma_lower = gamma;
        gamma = gamma * pen_t;
        target_K = target_K - 1;
        
        Z1_old = Z1;
        Z2_old = Z2;
        Z3_old = Z3;
        Lambda1_old = Lambda1;
        Lambda2_old = Lambda2;
        Lambda3_old = Lambda3;
        R_old = R;
        KSI_old = KSI;
        THETA1_old = THETA1;
        THETA2_old = THETA2;
        THETA3_old = THETA3;
        ETA1_old = ETA1;
        ETA2_old = ETA2;
        ETA3_old = ETA3;
    elseif no_class < target_K
        gamma = (gamma_lower + gamma)/2;
        Z1 = Z1_old;
        Z2 = Z2_old;
        Z3 = Z3_old;
        Lambda1 = Lambda1_old;
        Lambda2 = Lambda2_old;
        Lambda3 = Lambda3_old;
        R = R_old;
        KSI = KSI_old;
        THETA1 = THETA1_old;
        THETA2 = THETA2_old;
        THETA3 = THETA3_old;
        ETA1 = ETA1_old;
        ETA2 = ETA2_old;
        ETA3 = ETA3_old;

        if abs(gamma - gamma_lower) < 1e-3
            gamma = gamma * pen_t;
            target_K = target_K - 1;
        end
    elseif no_class > target_K
        U_output(:,:,k+1) = [U1;U2;U3];
        Z_output(:,:,k+1) = Z;
        THETA_output(:,:,k+1) = [THETA1;THETA2;THETA3];
        k = k + 1;
        gamma_lower = gamma;
        gamma = gamma * pen_t;
        Z1_old = Z1;
        Z2_old = Z2;
        Z3_old = Z3;
        Lambda1_old = Lambda1;
        Lambda2_old = Lambda2;
        Lambda3_old = Lambda3;
        R_old = R;
        KSI_old = KSI;
        THETA1_old = THETA1;
        THETA2_old = THETA2;
        THETA3_old = THETA3;
        ETA1_old = ETA1;
        ETA2_old = ETA2;
        ETA3_old = ETA3;
    end
    
end


U_output = U_output(:,:,1:k);
Z_output = Z_output(:,:,1:k);
THETA_output = THETA_output(:,:,1:k);


end