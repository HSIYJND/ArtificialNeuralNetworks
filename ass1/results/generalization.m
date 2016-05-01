clear all
close all


% my parameters.
sigma=0.3; % noise
NPoints=50; % number of function points as input
 %generation of examples and targets
x=linspace(0,3*pi,NPoints); y=sin(x);
% add noise to the data
yoriginal=y; % save this


% loop over all neural networks
neuronNrs=[1 2 3 4 5 6 7 8 9 10 11 12 15 17 20];
MaxIterations=1000;
NDataSets=100;
trainAlgos=char('traingd','traingda','traingdm','traincgf','traincgp','trainbfg','trainlm','trainbr')
biasResults=zeros(length(trainAlgos), length(neuronNrs));
varianceResults=zeros(length(trainAlgos), length(neuronNrs));


figure;
hold on;

biasPlot=subplot(1,2,1);
title('Bias');
xlabel('Number of neurons');
ylabel('Bias');
hold on;
variancePlot=subplot(1,2,2);
title('Variance');
xlabel('Number of neurons');
ylabel('Variance');
hold on;
Legend=cell(1, size(trainAlgos,1));

for algoNr = 1:size(trainAlgos,1)
    rng(123) % set the same seed for all algos
    
    algoName=char(strcat(trainAlgos(algoNr,:)))
    Legend{algoNr}=algoName;
    
    parfor neuronNr = 1:length(neuronNrs)
        
        numberOfNeurons=neuronNrs(neuronNr)
        disp(sprintf('Using %i neurons...', numberOfNeurons));
        
        ymean = zeros(size(yoriginal));
        variance = zeros(size(yoriginal));
        
        
        % generate a number of data sets
        for dataSetNr = 1:NDataSets
            
            % generate a new data set, randomly
            y=yoriginal+randn(size(yoriginal))*sigma;
            % convert the data to a useful format
            p=con2seq(x); t=con2seq(y); 
            
            % make and train the network
            net=feedforwardnet(numberOfNeurons, char(algoName));
            
            % set some parameters
            net.trainParam.epochs = MaxIterations;  % set the number of epochs for the training
            net.trainParam.showWindow = false;
            net.divideFcn = 'dividetrain';
            net=train(net,p,t); % train the networks
        
            % now update the values of bias and variance
            ynet = net(x); % made an error here in previous version!
            ymean = ymean + ynet;
            variance = variance + ynet.^2;
        end
        
        ymean = ymean / NDataSets;
        variance = sum(variance / NDataSets - ymean.^2);
        bias = sum((ymean - yoriginal).^2);
        
        % store the results
        biasResults(algoNr, neuronNr) = bias;
        varianceResults(algoNr, neuronNr) = variance;
    end
    hold on;
    subplot(1,2,1);
    semilogy(neuronNrs,biasResults(algoNr, :), 'DisplayName', algoName, 'LineWidth', 2)
    hold on;
    subplot(1,2,2);
    semilogy(neuronNrs,varianceResults(algoNr, :), 'LineWidth', 2)
    hold on;
end

subplot(1,2,1);
set(gca,'yscale','log');
grid on
legend(gca,'show')
hold on;
subplot(1,2,2);
set(gca,'yscale','log');
grid on
savefig('bias_and_variance.fig');
save('my_workspace_loops.mat');