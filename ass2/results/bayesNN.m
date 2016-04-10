clear all
close all

% ABOUT THE NETWORK:
% the w=(w1,w2) define first of all a direction in w-space. Perpedicular is
% the boundary line. The probability in w-space is non-symmetric under
% parity operations, due to the transfer function. It is 1 (or a + marker)
% in the direction of the w-vector. It is zero in the opposite direction
% (corresponding to a o marker). The transfer is at |w|=0, since there is
% no bias here.

%Also, the size of w matters. The smaller it it, the
% smaller the boundary slope (w = 0 corresponds to a constant y=1 value,
% w=\infty defines a heaviside function.).


a=1;                                % absolute value of lower boundary of the weights
b=1;                                % upper boundary of the weights
s=0.1;                              % stepsize for the grid plot
w1=(-a:s:b)';                       % define the grid for the plot over w-space
w2=(-a:s:b)';                       % "

X2=[-5 -5; 5 5];                    % define the first two points
X4=[-5 -5; 5 5; 0 1; -1 0];         % define all four points
T2=[0 ; 1];                         % target for first two points
T4=[0 ; 1; 0; 1];                   % target for all four

%**********************************
% using just the first 2 data points

figure
subplot(3,1,1);
% make prior
for i=1:length(w1)
    for j=1:length(w2)
        w=[w1(i) w2(j)];
        prior(i,j)=(1/(2*pi))*exp(-norm(w)^2)/2;    % calculate the predefined gaussian prior
    end
end
surf(w1,w2,prior)
grid on
box on
title('Prior')

% make posteriors
n=size(X2,1); % number of rows = number of data points
posterior = prior; % mostly for the correct size (initialization)
for k=1:n
    x=X2(k,:); % these are the data points in the case at hand
    for i=1:length(w1)
        for j=1:length(w2)
            w=[w1(i) w2(j)];
            y=1/(1+exp(-w*x')); % our little network: y=sigmoid(w*x), sharp linear boundary line
            likelihood=y^T2(k)*(1-y)^(1-T2(k)); % probability of getting 0 or 1, given our real value y
            posterior(i,j)=likelihood*posterior(i,j); % update the posterior with the data
        end
    end
end
subplot(3,1,2);
surf(w1,w2,posterior)
grid on
box on
title('Posterior after 2 points')

%************************
% using all 4 data points
% make prior again
for i=1:length(w1)
    for j=1:length(w2)
        w=[w1(i) w2(j)];
        prior(i,j)=(1/(2*pi))*exp(-norm(w)^2)/2; % this can be avoided
    end
end

% make posteriors
n=size(X4,1);
posterior = prior;
for k=1:n
    x=X4(k,:);
    for i=1:length(w1)
        for j=1:length(w2)
            w=[w1(i) w2(j)];
            y=1/(1+exp(-w*x'));
            likelihood=y^T4(k)*(1-y)^(1-T4(k));
            posterior(i,j)=likelihood*posterior(i,j);
        end
    end
end
subplot(3,1,3);
surf(w1,w2,posterior)
grid on
box on
title('Posterior after all points')