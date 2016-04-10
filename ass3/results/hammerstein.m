% generate a hammerstein time series
n = 100; % number of sequences
u = zeros(1,n);
x = zeros(1,n);
y = zeros(1,n);
% fill it up
u(1)=randn; %random number drawn from a standard gaussian distribution
x(1)=rand+sin(u(1));
y(1)=0.6*x(1);
for i=2:n
    u(i)=randn;
    x(i)=0.6*x(i-1)+sin(u(i));
    y(i)=x(i);
end
plot(1:n,y)