cd%% Create data
NPoints=50; % number of function points as input
%generation of examples and targets
x=linspace(0,3*pi,NPoints); y=sin(x);
% add noise to the data
yoriginal=y; % save this
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%% mapping the performance of a number of training rules
trainAlgos=char('traingd','traingda','traingdm','traincgf','traincgp','trainbfg','trainlm','trainbr');
Legend=cell(1, size(trainAlgos,1));
MaxIterations=500;
numberOfNeurons=5;

%% figure stuff
figure;
hold on;

timePlot=subplot(1,2,1);
title('Performance ifo time');
xlabel('Time (s)');
ylabel('Mean-Square Error (MSE)');
hold on;
epochPlot=subplot(1,2,2);
title('Performance ifo epochs');
xlabel('Epoch');
ylabel('Mean-Square Error (MSE)');
hold on;

%% some stuff to keep, and return in the end
for algoNr = 1:size(trainAlgos,1)
    
    % take an algo name
    algoName=char(strcat(trainAlgos(algoNr,:)))
    Legend{algoNr}=algoName;
    
    % make and train the network
    net=feedforwardnet(numberOfNeurons, char(algoName));
    init(net);
    
    % set some parameters
    net.trainParam.epochs=MaxIterations;  % set the number of epochs for the training
    net.trainParam.showWindow = false;
    [net,tr]=train(net,p,t); % train the networks
    
    % getting time, epochs and performence for those
    times = tr.time;
    size(times)
    perf = tr.perf;
    size(perf)
    epochs = tr.epoch;
    size(epochs)
    
    % make the plots
    hold on;
    subplot(1,2,1);
    loglog(times(2:end),perf(2:end), 'DisplayName', algoName, 'LineWidth', 2);
    hold on;
    subplot(1,2,2);
    loglog(epochs,perf, 'LineWidth', 2)
    hold on;
end

subplot(1,2,1);
set(gca,'yscale','log');
set(gca,'xscale','log');
legend(gca,'show')
hold on;
subplot(1,2,2);
set(gca,'yscale','log');
set(gca,'xscale','log');

savefig('performance_plot.fig');