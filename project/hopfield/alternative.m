close all; clc; clear all;

%% CREATE THE ATTRACTOR STATES
% get all the letters of the alphabet in CAPITALS
[ALPHABET, ~]=prprob();
% now the unique lower case letters of my name
name=GenerateName;
% add the ALPHABET and name
allLetters=[name';ALPHABET']';
% rescale from -1 to 1 instead of 0 and 1
allLetters = 2*allLetters - 1;

%% CREATE THE FULL SET OF POSSIBILITIES
% make the set of the first 25 letters
letters = allLetters(:,1:25);
% make some possibilities of distortion of 3 pixels
numDist = 100;
X = zeros(size(letters,1),size(letters,2)*numDist);
T = zeros(size(letters,1),size(letters,2)*numDist);
for il = 1:size(letters,2)
    letter = letters(:,il);
    for id = 1:numDist
        X(:,(il-1)*numDist+id) = DistortImage(letter);
        T(il,(il-1)*numDist+id) = 1; % only that one is 1
    end
end

%% TRAIN A NEURAL NETWORK
% create network with 25 neurons
net = feedforwardnet(25);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'softmax';
net = train(net, X, T); % 25 dimensional output (https://www.youtube.com/watch?v=iZ3e_cifP7Y)
% all letters have same distance of 2

%% NOW CHECK ITS CAPABILIIES
Nit = 100;
% store number of wrong results
Nwrong = 0;
% loop over Nit distored images per letter and calculate the error
for letterNr = 1:size(letters,2)
    letter = letters(:,letterNr);
    for it = 1:Nit
        distImage = DistortImage(letter);
        % use the net to retrieve
        [Y,~,~] = net(letter);
        
        % one-hot representation
        [~,ind] = max(Y); ind = ind(1); % one-hot encoding
        
        % 
        
        % check if it's the correct one
        Nwrong = Nwrong + (ind ~= letterNr);
    end
end
Nwrong = Nwrong / (Nit*P*size(allLetters,1)); % normalize over number of states that we generated and nr of pixels
