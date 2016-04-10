clear all; close all; clc;

%% LOAD THE DATA FROM TOLEDO

% load the data from Toledo
load 'data/Data_Problem1_regression.mat'

% generate my data from studentnr 0639870
d1 = 9;
d2 = 8;
d3 = 7;
d4 = 6;
d5 = 3;
T = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);
X = [X1 X2]; % useful for running sim
N = size(T,1);

% a check
assert((size(X1,1) == N) && (size(X2,1) == N));

%% DIVIDE THE DATA IN SUBSETS

% randomize the data, shuffle it, also transpose it 
shuffledInd = randperm(size(T,1));
X1 = X1(shuffledInd);
X2 = X2(shuffledInd);
T = T(shuffledInd);

% divide the data in subsets (use the first 3000, they are already shuffled)
useN = 3000;
[trainInd,valInd,testInd] = dividerand(useN,1/3,1/3,1/3);

% easier variables for training the network
xtrain = [X1(trainInd)'; X2(trainInd)'];% input
ytrain = T(trainInd,:)'; % target
xval = [X1(valInd)'; X2(valInd)'];
yval = T(valInd)';
xtest = [X1(testInd)'; X2(testInd)'];
ytest = T(testInd)';

%% VIZUALISE THE DATA

% create a regular grid for the interpolant (which is not the nets)
ndim = 50;
[Xmesh, Ymesh, ~, ~] = meshinterpolate(X1, X2, T, ndim);
% now interpolate
trainInterpolate = TriScatteredInterp(X1(trainInd), X2(trainInd), T(trainInd)); hold on; % plot the interpolator
Zmesh = trainInterpolate(Xmesh, Ymesh);
% show the surface
figure;
surface(Xmesh, Ymesh, Zmesh); hold on;
% also show the data
scatter3(X1(trainInd), X2(trainInd), T(trainInd), '.', 'MarkerEdgeColor','red', 'MarkerFaceColor', 'red');
xlabel('x_1'); ylabel('x_2'); zlabel('z');
savefig('train_surface.fig');

% show that the data is well distributed and divided randomly
surface(Xmesh, Ymesh, Zmesh); hold on; % plot the interpolator
scatter3(X1(trainInd), X2(trainInd), T(trainInd), 15, 'MarkerFaceColor','g'); hold on;
scatter3(X1(valInd), X2(valInd), T(valInd), 15, 'MarkerFaceColor',[1 .5 0]); hold on;
scatter3(X1(testInd), X2(testInd), T(testInd), 15, 'MarkerFaceColor','r'); hold off;

%% CREATE THE VALIDATION PLOT

% create some vectors to plot
nhvals = [1,2,3,4,5,10,15,20,25,30,35,40,45,50];
mseVal = zeros(length(nhvals),1);
mseTrain = zeros(length(nhvals),1);
mseTest = zeros(length(nhvals),1);

for nhIt = 1:length(nhvals)
    
    % number of neurons in the hidden layer
    nh = nhvals(nhIt);
    disp(['Training with ' num2str(nh) ' neurons in hidden layer']);
    
    % train the network
    net = feedforwardnet(nh,'trainlm');
    net.divideFcn = 'dividetrain'; % Use the whole training set for training
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    net.trainParam.epochs = 5000;
    
    % train the network
    [net,tr] = train(net, xtrain, ytrain);
    nntraintool('close');
    
    % show number of epochs used
    disp(['Used ' num2str(tr.num_epochs) ' epochs']);
    
    % make predictions for the validation set
    ytrainPred = sim(net, xtrain);
    yvalPred = sim(net, xval);
    ytestPred = sim(net, xtest);
    
    % calculate the mse for these points
    perfTrain = perform(net,ytrain,ytrainPred);
    perfVal  = perform(net,yval,yvalPred);
    perfTest = perform(net,ytest,ytestPred);
    
    % store it
    mseTrain(nhIt) = perfTrain;
    mseVal(nhIt) = perfVal;
    mseTest(nhIt) = perfTest;
end

%% SELECT A NETWORK ARCHITECTURE FROM THE RESULTS
nhfinal = 35;

%% SHOW PERFORMANCE ON VALIDATION SET
figure
semilogy(nhvals, mseTrain, 'Color','b', 'LineWidth',2); hold on;
semilogy(nhvals, mseVal, 'Color','g', 'LineWidth',2); hold on;
semilogy(nhvals, mseTest, 'Color','r', 'LineWidth',2);
xlabel('Neurons in the hidden layer'); ylabel('MSE');
savefig('performance_val.fig');

%% PLOT THE RESULT OF THE OPTED ARCHITECTURE
% train the network
net = feedforwardnet(nhfinal,'trainlm');
net.divideFcn = 'divideind'; % also give it access to the other sets for plotting
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
net.layers{1}.transferFcn = 'tansig'; % hidden layer
net.layers{2}.transferFcn = 'purelin'; % output layer
net.trainParam.epochs = 5000; % set really high, so it can decide itself
net.trainParam.max_fail = 50; % set really high, so it can decide itself
[net,tr] = train(net, xtrain, ytrain);

% calculate its output on the meshgrid
ZmeshNN = net([Xmesh(:) Ymesh(:)]');
ZmeshNN = reshape(ZmeshNN,size(Xmesh,1),size(Xmesh,2));

% map the interpolator of the test set
testInterpolant = TriScatteredInterp(X1(testInd),X2(testInd),T(testInd));
ZmeshTest = testInterpolant(Xmesh, Ymesh);

% plot it toghether with the interpolant of the TEST set
figure
surface(Xmesh, Ymesh, ZmeshNN,'FaceColor', 'g'); hold on; % plot the NN 
surface(Xmesh, Ymesh, ZmeshTest, 'FaceColor','r'); % plot the interpolator
xlabel('x_1'); ylabel('x_2'); zlabel('z');
savefig('NN_and_testsurf.fig');

% make the error plot
ZmeshError = ZmeshNN - ZmeshTest;
figure
contourf(Xmesh, Ymesh, ZmeshError);
xlabel('x_1'); ylabel('x_2'); colorbar;
h = colorbar; ylabel(h, 'Error');
savefig('NN_test_error.fig');

% plot the regression for the test set
figure;
plotregression(sim(net, xtest),ytest);