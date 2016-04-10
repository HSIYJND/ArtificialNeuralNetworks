clear all
close all

%% load my data set
load 'data/function_data'
N = size(data.X1,1);

%% plot the surface first
% create a regular grid for the interpolant (which is not the nets)
gridn = 50;
X1lin = linspace(min(data.X1),max(data.X1),gridn);
X2lin = linspace(min(data.X2),max(data.X2),gridn);
% make a meshgrid from the vectors
[X1mesh,X2mesh] = meshgrid(X1lin,X2lin);
% create a simple interpolant to guide the eye
f = scatteredInterpolant(data.X1,data.X2,data.T);
Z = f(X1mesh,X2mesh); % map it
% Plot the interpolant and the data
figure
mesh(X1mesh,X2mesh,Z) %interpolated
axis tight; hold on
plot3(data.X1,data.X2,data.T,'.','MarkerSize',15) %nonuniform
hold off

% randomize the data, shuffle it
shuffledInd = randperm(size(data.T,1));
data.X1 = data.X1(shuffledInd);
data.X2 = data.X2(shuffledInd);
data.T = data.T(shuffledInd);

% divide the data in subsets (use the first 3000, they are already shuffled)
useN = 3000;
[trainInd,valInd,testInd] = dividerand(useN,1/3,1/3,1/3);

% saving the indices is more than enough
% create a struct containing all indices and save it
indices.trainInd = trainInd;
indices.valInd = valInd;
indices.testInd = testInd;
save('data/tvt_set_indices.mat', 'indices');

% plot again with colors
figure
mesh(X1mesh,X2mesh,Z) %interpolated
axis tight; hold on
plot3(data.X1(indices.trainInd),data.X2(indices.trainInd),data.T(indices.trainInd),'.','MarkerSize',15,'Color','g')
plot3(data.X1(indices.valInd),data.X2(indices.valInd),data.T(indices.valInd),'.','MarkerSize',15,'Color',[1 .5 0])
plot3(data.X1(indices.testInd),data.X2(indices.testInd),data.T(indices.testInd),'.','MarkerSize',15,'Color','r')
hold off