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

% plot the first 10 letters of my data set
figure;
colormap(gray);
for letterNr=1:10
    subplot(2,5,letterNr);
    imagesc(reshape(allLetters(:,letterNr),5,7)','CDataMapping','scaled'); % 5x7 bit maps transposed to a 7x5
end
savefig('letters.fig');
hold off; close all;

%% CREATE A HOPFIELD RECURRENT NETWORK, TYPE 1
% type 1 retrieves 5 first letters
T = allLetters(:,1:5);
net = newhop(T);

%% DISTORT AND RETRIEVE IMAGES
% check the correct retrieval rate, output states that are spurious
% even though we can do the following exactly, we do a Monte Carlo
% simulation. We create 1000 distorted images of each of the 5 letters and
% use the Hopfield network to retrieve the original states.  If the number
% of attempts is large enoug, we should find all spurious states. Note
% however that there are 35!/(3!32!)= 6545 possible distorted images of
% each of the letters

% Compared to the assignment on Hopfields, the distorted image now has
% discrete values for the pixels. Hence, it's now more feasible to end up
% in a spurious state

% catch all the wrong states
wrongStates=cell(1,1);originalStates=cell(1,1);
Nwrong = 0;
timesteps = 1000;
for letterNr = 1:5
    fprintf('Start with letter nr : %i\n', letterNr)
    letter = T(:,letterNr);
    for it = 1:1000
        distImage = DistortImage(letter);
        Ai = {distImage};
        [Y,Pf,Af] = net({1 timesteps},{},Ai); % five time steps
        if ~isequal(Y{end},letter)
            any(isequal(wrongStates,Y{end}))
            Nwrong = Nwrong + 1;
            (Y{end} - letter)'
            fprintf('  Wrong state at iteration: %i\n',it);
            wrongStates{Nwrong} = Y{end};
            originalStates{Nwrong} = letter;
        end
    end
end

figure;
for wrongNr = 1:Nwrong
    subplot(2,Nwrong,wrongNr);
    imagesc(reshape(wrongStates{wrongNr},5,7)','CDataMapping','scaled'); hold on;
    subplot(2,Nwrong,Nwrong+wrongNr);
    imagesc(reshape(originalStates{wrongNr},5,7)','CDataMapping','scaled');
    hold off;
end
savefig('wrong_states.fig')
        