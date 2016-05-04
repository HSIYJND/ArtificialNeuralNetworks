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

% loop over number of patterns stored
MaxNpatt=25;
icf = zeros(1,MaxNpatt);
for P=1:size(icf,2)
    %% CREATE THE FULL SET OF POSSIBILITIES
    % make the set of the first 25 letters
    letters = allLetters(:,1:P);

    % make some possibilities of distortion of 3 pixels
    % this is used for training
    numDist = 10; % number of distorted images per letter
    X = zeros(size(letters,1),size(letters,2)*numDist);
    T = zeros(size(letters,2),size(letters,2)*numDist);
    for il = 1:size(letters,2)
        letter = letters(:,il);
        for id = 1:numDist
            X(:,(il-1)*numDist+id) = DistortImage(letter);
            T(il,(il-1)*numDist+id) = 1; % only that one is 1
        end
    end

    %% TRAIN A NEURAL NETWORK
    % create network with 25 neurons
    net = feedforwardnet(30);
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'softmax';
    net.trainParam.showWindow=0;
    net = train(net, X, T,'useParallel','yes'); % 25 dimensional output (https://www.youtube.com/watch?v=iZ3e_cifP7Y)
    % all letters have same distance of 2

    %% NOW CHECK ITS CAPABILIIES
    Nit = 1000;
    % store number of wrong results
    Nwrong = zeros(1,size(letters,2));
    % loop over Nit distored images per letter and calculate the error
    parfor letterNr = 1:size(letters,2)
        letter = letters(:,letterNr);
        fprintf('Starting with letter %i out of %i...\n',letterNr,size(letters,2))
        for it = 1:Nit
            distImage = DistortImage(letter);
            % use the net to retrieve
            [Y,~,~] = net(distImage);

            % one-hot representation
            [~,ind] = max(Y); ind = ind(1); % one-hot encoding

            % check if it's the correct one
            %Nwrong(letterNr) = Nwrong(letterNr) + sum(letters(:,ind) ~= letter);
            Nwrong(letterNr) = Nwrong(letterNr) + (ind ~= letterNr);
        end
    end
    %icf(P) = sum(Nwrong) / (Nit*size(letters,2)*size(letters,1)); % normalize
    icf(P) = sum(Nwrong) / (Nit*size(letters,2)); % normalize
    fprintf('%i patterns give a normalized pixel error of %f\n',P,icf(P));
end

figure;
plot(1:P,icf,'r-','LineWidth',2);
xlabel('Number of patterns stored'); ylabel('Pixel error');
savefig('alternative_error.fig');