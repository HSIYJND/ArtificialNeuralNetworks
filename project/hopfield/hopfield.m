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
Nwrong = 0;
timesteps = 1000;
for letterNr = 1:5
    fprintf('Start with letter nr : %i\n', letterNr)
    letter = T(:,letterNr);
    for it = 1:1000
        distImage = DistortImage(letter);
        [Y,~,~] = net({1 timesteps},{},{distImage});
        if ~isequal(Y{end},letter)
            if (Nwrong ~= 0 && sum(ismember(Y{end}', wrongStates', 'rows'))==1) % avoid doubles
                fprintf('Same state.\n')
            else
                Nwrong = Nwrong + 1;
                fprintf('  Wrong state at iteration: %i\n',it);
                wrongStates(:,Nwrong) = Y{end};
                originalStates(:,Nwrong) = Y{end};
            end
        end
    end
end

figure;
colormap(gray)
for wrongNr = 1:Nwrong
    subplot(2,Nwrong,wrongNr);
    imagesc(reshape(wrongStates(:,wrongNr),5,7)','CDataMapping','scaled'); hold on;
    subplot(2,Nwrong,Nwrong+wrongNr);
    imagesc(reshape(originalStates(:,wrongNr),5,7)','CDataMapping','scaled'); hold on;
end
savefig('wrong_states.fig'); hold off;

%% MAPPING ERROR IFO P
Nit = 100;
% store number of wrong results
Nwrong = zeros(1,size(allLetters,2));
% loop over the number of stored patterns P
for P = 1:size(allLetters,2)
    fprintf('Simulating with P = %i patterns stored.\n',P);
    % take P attractors
    T = allLetters(:,1:P);
    % create a hopfield net
    net = newhop(T);
    % loop over Nit distored images per letter and calculate the error
    for letterNr = 1:P
        letter = T(:,P);
        for it = 1:Nit
            distImage = DistortImage(letter);
            % use the net to retrieve
            [Y,~,~] = net({1 timesteps},{},{distImage});
                        
            % clip
            Y{end} = sign(Y{end});
            % check if it's the correct one
            sum(abs(Y{end} - letter));
            Nwrong(P) = Nwrong(P) + sum(abs(Y{end} - letter));
        end
    end
    Nwrong(P) = Nwrong(P) / (Nit*P*size(allLetters,1)); % normalize over number of states that we generated and nr of pixels
end

% estimate also with Hebb rule
figure;
sigmas = sqrt((1:size(allLetters,2))/size(allLetters,1));
mus = zeros(1,size(allLetters,2));
Perr = ones(1,size(allLetters,2))-normcdf(ones(1,size(allLetters,2)),mus,sigmas);
%plot it
plot(1:size(allLetters,2),Nwrong,'r-','LineWidth',2); hold on;
plot(1:size(allLetters,2),Perr,'b-','LineWidth',2);
legend('Pixel error','P_{error}')
ylabel('Error');
xlabel('Number of patterns stored');
savefig('Eerror_ifo_P.fig');