%% Bayesian Learning in neural networks
% 
% This demo demonstrates Bayesian learning of network weights
%
% Rob Heylen, feb 2007
clear all
close all
%% 
% Generate a single neuron perceptron with zero bias and two arbitrary weights
% plot the targets and decision boundary
numinput=50;
net=newp([-1 1; -1 1], 1);
net.IW{1,1}=rands(1,2);
P=rands(2,numinput)
T=sim(net,P)

%subplot(1,2,1);
plotpv(P,T);
hold on;
plotpc(net.IW{1,1},0);



%% 
% Generate a prior distribution for the weights and plot it

w1=(-1:0.05:1)';
w2=(-1:0.05:1)';
for i=1:length(w1)
    for j=1:length(w2)
        w=[w1(i) w2(j)];
        prior(i,j)=(1/(2*pi))*exp(-norm(w)^2)/2;
    end
end
%subplot(1,2,2);
%surf(w1,w2,prior);
contour(w1, w2, prior, 'LineWidth',2);
xlabel('w_1');
ylabel('w_2');
title('prior');
axis equal
savefig('prior_contour.fig');

%%
% Create posteriors by presenting all targets one by one.
% Plot updated distribution after each update.
for k=1:numinput
    x=P(:,k);
    for i=1:length(w1)
        for j=1:length(w2)
            w=[w1(i) w2(j)];
            y=1/(1+exp(-w*x));
            likelihood=y^T(k)*(1-y)^(1-T(k));
            prior(i,j)=likelihood*prior(i,j);
        end
    end
    n=sum(sum(prior)); % This loop is for normalization of the distribution
    for i=1:length(w1)
        for j=1:length(w2)
            prior(i,j)=prior(i,j)/n;
        end
    end
    surf(w1,w2,prior);
    pause(0.1);
end

%%
% Find the maximum of the distribution and plot the corresponding decision
% boundary. This is close to the decision boundary of the perceptron, and
% not many points are misclassified.
prob=0;
for i=1:length(w1)
    for j=1:length(w2)
        if (prior(i,j)>prob)
            prob=prior(i,j);
            maxind=[i,j];
        end
    end
end
%subplot(1,2,1);
plotpv(P,T);
hold on
allah1 = plotpc([w1(maxind(1)), w2(maxind(2))], 0);
set(allah1, 'Color', 'r');
hold on;
allah2 = plotpc(net.IW{1,1},0);
set(allah2, 'Color', 'b');
xlabel('x_1');
ylabel('x_2');
title('classification');
axis equal
savefig('separation_line.fig');

%%
% On the contour plot on the right we plot the posterior weight
% distribution and the actual perceptron weight with a star. The
% correlation is obvious.
%subplot(1,2,2);
%hold off;
contour(w1, w2, prior, 'LineWidth',2);
xlabel('w_1');
ylabel('w_2');
title('posterior');
hold on;
x=net.IW{1,1};
plot(x(2), x(1), 'b*');
plot(w1(maxind(2)), w2(maxind(1)), 'r*');
axis equal
savefig('posterior_contour.fig');