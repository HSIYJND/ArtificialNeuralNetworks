%In this script an elman network is trained and tested in order to
%model a so called hammerstein model. The system is described like this:

% x(t+1) = 0.6x(t-1) + sin(u(t))
%y(t) = x(t);

%Elman network should be able to understand the relation between output
%y(t) and input u(t). x(t) is a latent variable representing the internal
%state of the system/

clc;
clear;
close all;

n_tr=300; %number of training points
n_te=200; %number of test points

n_neurons=20;

n=1000; %total number of samples

u(1)=randn; %random number drawn from a standard gaussian distribution
x(1)=rand+sin(u(1));
y(1)=.6*x(1);

for i=2:n

    u(i)=randn;
    x(i)=.6*x(i-1)+sin(u(i));
    y(i)=x(i);

end

plot(y);
xlabel('time');
ylabel('y');

X=u(1:n_tr); %training set
T=y(1:n_tr);

T_test=y(end-n_te:end); %test set
X_test=u(end-n_te:end);

net = newelm(X,T,n_neurons); %create network
net = train(net,X,T); %train network

T_test_sim = sim(net,X_test); %test network

figure;

%Plot results and calculate correlation coefficient between target and
%output

plot(1:size(X_test,2),T_test,'r',1:size(X_test,2),T_test_sim,'b');
xlabel('time');
ylabel('y');
legend('target','prediction',-1);
R = corrcoef(T_test,T_test_sim);
R = R(1,2);

%------------------------------------------------------------------------
