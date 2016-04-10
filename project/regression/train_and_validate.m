%% preliminaries
close all
clear all

%% load the data sets
% load the full dataset
load 'data/function_data'
% load the division indices
load 'data/tvt_set_indices'

% easier variables
trainInd = indices.trainInd;
valInd = indices.valInd;
testInd = indices.testInd;
xtrain = [data.X1(trainInd)' ; data.X2(trainInd)'];
ytrain = data.T(trainInd)';
xval = [data.X1(valInd)' ; data.X2(valInd)'];
yval = data.T(valInd)';

% create some vectors to plot
nhmax = 20;
mse_vector = zeros(nhmax,1);

%% create the validation plot
parfor nh = 1:nhmax
    
    disp(['Training ' num2str(nh) ' of ' num2str(nhmax)]);
    
    % train the network
    net = feedforwardnet(nh,'trainlm');
    
    % tell it which data to use for training, validation and testing
    net.divideFcn = 'divideind'; % Divide data by indices (i.e. not randomly)
    net.divideParam.trainInd = trainInd;
    net.divideParam.valInd = valInd;
    net.divideParam.testInd = testInd;
    net.params.epochs = 10000; % set really high, so it can decide itself
    
    % train the network
    net = train(net, xtrain, ytrain);
    
    % make predictions for the validation set
    ypred = sim(net, xval);
    
    % calculate the mse for these points
    perf  = perform(net,yval,ypred);

    % store it
    mse_vector(nh) = perf;
    
end

%% display the plot
figure
plot(1:nhmax, mse_vector);
xlabel('Neurons in the hidden layer');
ylabel('MSE validation set');
savefig('hidden_neurons.fig');