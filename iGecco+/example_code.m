%%% Assume oracle numeber of clusters and features
% Input:
X = ; Y = ; Z = ;
% X: Continuous/Gaussian data: p1 * n
% Y: Count data:    p2 * n
% Z: Binary/Proportional data: p3 * n
K = ;  % Desired Number of Clusters
target_p = ; % Desired Number of Features
[class_id,active_set] = igecco_plus(X,Y,Z,K,target_p); % iGecco+
% Output:
% class_id: cluster assignment
% active_set: features that are selected


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Example Code; Example data from simulation in paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Spherical Data
load('X_whole_S3.mat')
load('Y_whole_S3.mat')
load('Z_whole_S3.mat')

X = X_whole(:,:,1);
Y = Y_whole(:,:,1);
Z = Z_whole(:,:,1);

% X: p1 * n, Y: p2 * n, Z: p3 * n
K = 3; target_p = 30;
[class_id,active_set] = igecco_plus(X,Y,Z,K,target_p);

%%% Visualize Clustering result
subplot(1,3,1)
[coeff,score,latent] = pca(X');
n = length(class_id);
c = zeros(n,3); 
c(class_id == 1,1) = 1; c(class_id == 2,2) = 1; c(class_id == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
labels = cellstr(num2str([1:n]'));
title("PCA plot in X1")

subplot(1,3,2)
[coeff,score,latent] = pca(Y');
c = zeros(n,3); 
c(class_id == 1,1) = 1; c(class_id == 2,2) = 1; c(class_id == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
labels = cellstr(num2str([1:n]'));
title("PCA plot in X2")

subplot(1,3,3)
[coeff,score,latent] = pca(Z');
c = zeros(n,3); 
c(class_id == 1,1) = 1; c(class_id == 2,2) = 1; c(class_id == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
labels = cellstr(num2str([1:n]'));
title("PCA plot in X3")


%%%%% Tuning Parameter Selection
%%%% Number of Clusters and features estimated from data
tic
alpha_list = [30 40 50];  %%% range of feature penalty to consider
% alpha_list = [20 30 100 250 500 1000];  %%% range of feature peanlty to consider
%%% tuning parameter selection, default is BIC
[K_optimal,alpha_optimal,class_id,active_set] = tune_igecco_plus(X,Y,Z,alpha_list); 
time = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Half Moon Data
load('X_whole_S6.mat')
load('Y_whole_S6.mat')
load('Z_whole_S6.mat')

X = X_whole(:,:,1);
Y = Y_whole(:,:,1);
Z = Z_whole(:,:,1);

% X: p1 * n, Y: p2 * n, Z: p3 * n
K = 3; target_p = 30;
[class_id,active_set] = igecco_plus(X,Y,Z,K,target_p); % iGecco+

%%% Visualize Clustering result
subplot(1,3,1)
[coeff,score,latent] = pca(X');
n = length(class_id);
c = zeros(n,3); 
c(class_id == 1,1) = 1; c(class_id == 2,2) = 1; c(class_id == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
title("PCA plot in X1")

subplot(1,3,2)
[coeff,score,latent] = pca(Y');
c = zeros(n,3); 
c(class_id == 1,1) = 1; c(class_id == 2,2) = 1; c(class_id == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
title("PCA plot in X2")

subplot(1,3,3)
[coeff,score,latent] = pca(Z');
c = zeros(n,3); 
c(class_id == 1,1) = 1; c(class_id == 2,2) = 1; c(class_id == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
labels = cellstr(num2str([1:n]'));
title("PCA plot in X3")


%%%%% Tuning Parameter Selection
%%%% Number of Clusters and features estimated from data
alpha_list = [20 100 250 500 1000];  %%% range of feature peanlty to consider
%%% tuning parameter selection, default is BIC
[K_optimal,alpha_optimal,class_id,active_set] = tune_igecco_plus(X,Y,Z,alpha_list); 