%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt 
%%%%%%%%%%%

% my parameters.
sigma=0.3; % noise
NEvals=1000; % number of evaluations (take 1000 for figs)
NPoints=50; % number of function points as input
NHiddenNeurons=15;

%generation of examples and targets
x=linspace(0,3*pi,NPoints); y=sin(x);
% add noise to the data
yoriginal=y; % save this
y=y+randn(size(y))*sigma;
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%creation of networks
net1=feedforwardnet(NHiddenNeurons,'traingda');
net2=feedforwardnet(NHiddenNeurons,'trainbr');
%net1=newff([0 3*pi],[5 1],{'tansig','purelin'},'traingd');
%net2=newff([0 3*pi],[5 1],{'tansig','purelin'},TrainingAlgo);
net2.iw{1,1}=net1.iw{1,1}; %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};

net1.trainParam.epochs=NEvals;
net2.trainParam.epochs=NEvals;
net1=train(net1,p,t);
net2=train(net2,p,t);
a13=sim(net1,p); a23=sim(net2,p);


MyLineWidth=2;
MyMarkerSize=5;
%plots
figure
plot(x,yoriginal,'ko',x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g', 'LineWidth',MyLineWidth, 'MarkerSize',MyMarkerSize);
legend('sin(x)','noisy data','trainlm','trainbr',4);
xlabel('x')
ylabel('sin(x)')
title('1000 epochs');
%legend('target','trainlm','traingd',4);
