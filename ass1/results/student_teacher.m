%% Perceptron: Student learning from teacher
% In this demo we use a teacher perceptron to generate a dataset
% which has to be learned by a student perceptron. 
%
% Rob Heylen, February 2007

%% Create a teacher net
% create the net teacher with some learning rule. 
% its one-neuron perceptron
nett = newp([-1 1; -1 1], 1);
% use only a single pass
nets=newp([-1 1; -1 1],1);
% randomize its parameters
net.IW{1}=rands(1,2);
net.b{1}=rands(1);

%% Creation of the teacher's data set
% generate n datapoints in a small interval, with one outlier (making it 101)
n=100;
p=rands(2,n);

% now get the teacher's output
t=sim(nett,p);

%plot this
hold on;
plotpv(p,t);
% plot the underlying teacher boundary
plotpc(nett.IW{1,1}, nett.b{1});

%% Create the student
nets.adaptParam.passes=1;
% now train the student
[nets, outputs, errors]=adapt(nets,p,t);
%plot its decision boundary
linehandle=plotpc(nets.IW{1,1}, nets.b{1});

%% Do more passes

while (sum(abs(errors)) ~= 0)
    Disp('passing')
    [nets, outputs, errors]=adapt(nets, p, t);
    linehandle=plotpc( nets.IW{1,1}, nets.b{1}, linehandle);
    pause(0.5)
end


