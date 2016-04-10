%A script to test if 2d Hopfield network can recognize letter of alphabet

%clear
%clc
%close all

function [] = hopdigit(noiselevel,num_iter)

load digits; clear size
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X)); 
maxx=max(max(X));

%Values must be +1 or -1
X(X==0)=-1;
%-------------------------------------------------------------------------

%Attractors of the Hopfield network

zero = X(1,:); %to visualize: digit=reshape(X(1,:),15, 16)'; -> imshow(digit); 
one = X(21,:);
two = X(41,:);
three = X(61,:);
four = X(81,:);
five = X(101,:);
six = X(121,:);
seven = X(141,:);
eight = X(161,:);
nine = X(181,:);
size(X)

index_dig = [1,21,41,61,81,101,121,141,161,181];
num_dig = 10;
%--------------------------------------------------------------------------

T = [zero;one;two;three;four;five;six;seven;eight;nine]';

%Create network
net = newhop(T);

%Check if digits are attractors
[Y,Pf,Af] = sim(net,num_dig,[],T);
Y = Y';

fig1 = figure;

subplot(num_dig,1,1);

for i = 1:num_dig
digit = Y(i,:);
digit = reshape(digit,15,16)'; 
subplot(num_dig,1,i);
imshow(digit)
hold on
end

ax = axes('position',[0,0,1,1],'visible','off');
tx = text(0.5,0.95,'Attractors');
set(tx,'fontweight','bold');

%The plots show that they are attractors.

%------------------------------------------------------------------------


fig2 = figure;

% Add noise to the digit maps

%noisefactor = 0.2; %>=0

noisefactor = noiselevel;

noise = noisefactor*maxx; % sd for Gaussian noise

Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

%Show noisy digits:

subplot(num_dig,1,1);

for i = 1:num_dig
digit = Xn(index_dig(i),:);
digit = reshape(digit,15,16)';
subplot(num_dig,1,i);
imshow(digit)
hold on
end


ax = axes('position',[0,0,1,1],'visible','off');
tx = text(0.5,0.95,'Noisy digits');
set(tx,'fontweight','bold');



%------------------------------------------------------------------------

%See if the network can correct the corrupted digits 

fig3 = figure;

%num_steps = 1000;

num_steps = num_iter;

Xn = Xn';
Tn = {Xn(:,index_dig)};
[Yn,Pf,Af] = sim(net,{num_dig num_steps},{},Tn);
Yn = Yn{1,num_steps};
Yn = Yn';

subplot(num_dig,1,1);

for i = 1:num_dig
digit = Yn(i,:);
digit = reshape(digit,15,16)';
subplot(num_dig,1,i);
imshow(digit)
hold on
end


ax = axes('position',[0,0,1,1],'visible','off');
tx = text(0.5,0.95,'Reconstructed noisy digits');
set(tx,'fontweight','bold');

%-----------------------------------------------------------------------

