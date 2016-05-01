function distImage = DistortImage(image)

% take three random numbers between 1 and 35
indices = randsample(35,3); % unique, no replacement

% now get those pixels and switch them
distImage = image;
distImage(indices) = -distImage(indices);
