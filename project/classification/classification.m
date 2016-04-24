clear all; close all; clc;

%% LOAD THE DATA FROM TOLEDO

% generate my data from studentnr 0639870
% digit 0 gives me (C+,C-) = (5,6) of the white wine
% these classes are in the last column
datatable = importdata('data/winequality-white.csv');
data = datatable.data;
pos = data(data(:,end) == 5,:); % 5 is positive
neg = data(data(:,end) == 6,:); % 6 is negative
Npos = size(pos,1);
Nneg = size(neg,1);

% put these in more useful training formats (all in one)
% input are all columns, except for the last one, which is target
% we set the target values to +/-1 instead of 5 and 6
X = [pos(:,1:(end-1)) ; neg(:,1:(end-1))]';
T = [ones(Npos,1) ; -ones(Nneg,1)]';
N = Npos + Nneg;

% shuffle and divide the data
[trainInd, valInd, testInd] = dividerand(N); % default is 0.7, 0.15, 0.15

%% CREATING A NEURAL NETWORK TO CLASSIFY THE DATA
% create a network and train it
net = feedforwardnet(20, 'trainlm');
% classification values are between -1 and 1, hence, we can use the tangent
% sigmoid function in the output layer as well
net.layers{1}.transferFcn = 'tansig'; % hidden layer
net.layers{2}.transferFcn = 'tansig'; % output layer
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
net.trainParam.max_fail = 50; % may vary this
net.trainParam.min_grad = 10^-15; % may vary this
[net, tr] = train(net,X,T);

%% PERFORMANCE CHECKS
predVal = sim(net, X(:,valInd));
CCRval = sum(sign(predVal) == T(valInd))*100/length(valInd)

%% PCA
% preprocess the data to get zero mean
Ntrain = size(X(trainInd),2)
mean = X(:, trainInd) - repmat(mean(x,2), 1, N);
zeromean =  - repmat(mean(x,2), 1, N)
zeromean = mapstd(X(:,trainInd)) - repmat(mean(x,2), 1, N);
% create 11 x 11 covariance matrix
covMat = cov(zeromean'); % every row must be data entry
% calculate the eigv and eigd
[eigvecs, eigvals] = eig(covMat)
eigvals = diag(eigvals)
sum(eigvals)
% plot the eigenvalues
bar(eigvals)