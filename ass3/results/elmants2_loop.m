

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

n_neurons=100;

n=1000; %total number of samples

u(1)=randn; %random number drawn from a standard gaussian distribution
x(1)=rand+sin(u(1));
y(1)=.6*x(1);

for i=2:n
    u(i)=randn;
    x(i)=.6*x(i-1)+sin(u(i));
    y(i)=x(i);
end
figure;

plot(y);
xlabel('time');
ylabel('y');

X=u(1:n_tr); %training set
T=y(1:n_tr);

T_test=y(end-n_te:end); %test set
X_test=u(end-n_te:end);

n_neurons_max =100;
correlations = zeros(n_neurons_max,1);
error = zeros(n_neurons_max,1);
parfor n_neurons=1:n_neurons_max
    disp(['Number of neurons: ',num2str(n_neurons)])
    net = newelm(X,T,n_neurons,{'tansig','purelin'}); %create network
    net.trainParam.epochs = 1000;
    net.divideFcn = 'dividetrain';
    net.trainParam.showWindow = false;
    [net,tr] = train(net,X,T); %train network
    T_test_sim = sim(net,X_test); %test network
    R = corrcoef(T_test,T_test_sim);
    R = R(1,2);
    correlations(n_neurons) = R;
    mse = sum((T_test - T_test_sim).^2)/size(T_test,2);
    error(n_neurons) = mse;
end


close all;
figure;

%Plot results and calculate correlation coefficient between target and
%output
plot(1:n_neurons_max, correlations, 'LineWidth',2 );
xlabel('Number of neurons');
ylabel('Correlation');
savefig('correlation_map.fig')

close all;
figure;
plot(1:n_neurons_max, error,'LineWidth',2 );
xlabel('Number of neurons');
ylabel('MSE');
savefig('mse_map.fig')
%------------------------------------------------------------------------