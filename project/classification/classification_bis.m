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
rng(111);
[trainInd, valInd, testInd] = dividerand(N); % default is 0.7, 0.15, 0.15
stdX = mapstd(X);


%% CREATING A NEURAL NETWORK TO CLASSIFY THE DATA
% create a network and train it
rng(111);
net = feedforwardnet([20 20], 'trainlm'); 
% classification values are between -1 and 1, hence, we can use the tangent
% sigmoid function in the output layer as well
net.layers{1}.transferFcn = 'tansig'; % hidden layer
net.layers{2}.transferFcn = 'tansig'; % output layer
net.layers{3}.transferFcn = 'tansig'; % output layer
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
net.trainParam.max_fail = 50; % may vary this
net.trainParam.min_grad = 10^-15; % may vary this
net = train(net,stdX,T);

%% PERFORMANCE CHECKS
predVal = sim(net, X(:,valInd));
CCRval = sum(sign(predVal) == T(valInd))*100/length(valInd);
predTest = sim(net, stdX(:,testInd));
CCRtest = sum(sign(predTest) == T(testInd))*100/length(testInd);
fprintf('CCRval= %f and CCRtest %f .\n',CCRval,CCRtest);

%% PCA
% preprocess the data to get zero mean = 0 and stddev = 1 for all
% properties
Ntrain = size(X(:,trainInd),2); % number of training data points
[~,~,eigvals,~,~,~] = doPCA(X(:,trainInd)',11); % use all 11

% plot the eigenvalues
figure;
bar(eigvals/max(eigvals)); hold on; % shows that one needs 'only' 10 basis vectors
plot(1:11, 1-cumsum(eigvals/sum(eigvals)), 'r-', 'LineWidth', 2)
ylabel('\lambda_k'); xlabel('k');
axis([0 12 0 1]);
savefig('eigenvalues.fig');

% we project the vectors onto the restricted eigenbasis (columns of eigvecs)
numBasisVecs=8; % choose the number of eigenvectors
[PCABasis,redXTrain,eigvals,meanTrain,stddevTrain,stdXTrain] = doPCA(X(:,trainInd)',numBasisVecs);

% project also validation and test set, but first standardize them, use
% part of my doPCA function for this
% note that we project with the PCA basis of the training set, as required
% by the assignment
[~,~,~,meanVal,stddevVal,stdXVal] = doPCA(X(:,valInd)',numBasisVecs);
[~,~,~,meanTest,stddevTest,stdXTest] = doPCA(X(:,testInd)',numBasisVecs);
redXVal = stdXVal*PCABasis;
redXTest = stdXTest*PCABasis;

% now reconstruct
redXTrain = redXTrain*PCABasis';
redXVal = redXVal*PCABasis';
redXTest = redXTest*PCABasis';

% create a new input matrix X from these components
%X = zeros(numBasisVecs,N);
X(:,trainInd) = redXTrain';
X(:,valInd) = redXVal';
X(:,testInd) = redXTest';

% we can now use them for training
%% CREATING A NEURAL NETWORK TO CLASSIFY THE DATA
% create a network and train it
rng(1);
net = feedforwardnet(11, 'trainlm');
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
CCRval = sum(sign(predVal) == T(valInd))*100/length(valInd);
predTest = sim(net, X(:,testInd));
CCRtest = sum(sign(predTest) == T(testInd))*100/length(testInd);
fprintf('CCRval= %f and CCRtest %f .\n',CCRval,CCRtest);