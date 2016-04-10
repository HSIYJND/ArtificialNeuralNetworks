%% Perceptron: Student learning from teacher
% In this demo we use a teacher perceptron to generate a dataset
% which has to be learned by a student perceptron. 
%
% Rob Heylen, February 2007

%% Creation of teacher perceptron
% Create a single-neuron perceptron "nett" with two inputs in [-1,1]
% Change the weights and bias to arbitrary chosen numbers
nett=newp([-1 1;-1 1],1);
nett.IW{1,1}=[-0.3 2];
nett.b{1}=0.4;

%% Creation of student perceptron
% Create a student perceptron with random weights and bias
nets=newp([-1 1;-1 1],1);
nets.IW{1,1}=rands(1,2);
%nets.b{1}=rands(1);
nets.b{1}=rand;

%% Creation of dataset
% We generate n random inputs, and use the teacher perceptron to provide
% targets
% We then plot this dataset, along with the teacher's decision boundary

n=100;           % Number of examples
p=rands(2,n);   % Create n inputs randomly in [-1,1]
t=sim(nett,p);  % Create outputs using the teacher network
%t=double(t);    % Turn the resulting logical array into a double array
hold on;        % Plot and keep everything in this figure
plotpv(p,t);    % Plot dataset
plotpc(nett.IW{1,1},nett.b{1}); % Plot teacher decision boundary

%% Teaching the student network
% We train the student network with a single pass through the dataset, and
% plot the resulting decision boundary
nets.adaptParam.passes=1;       % Configure the network for a single pass 
[nets,a,e]=adapt(nets,p,t);     % Adapt the student network
linehandle=plotpc(nets.IW{1,1},nets.b{1}); % Plot the decision boundary

%% More passes through the dataset
% When we pass through the dataset multiple times we improve the result
% considerably, until all dots are classified perfectly

while sum(abs(e))~=0
    [nets,a,e]=adapt(nets,p,t);     
    linehandle=plotpc(nets.IW{1,1},nets.b{1},linehandle); 
    pause(0.5);
end