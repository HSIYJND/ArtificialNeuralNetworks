function bayesian_training

x=-1:.01:1;
t=sin(2*pi*x)+.1*randn(size(x));
net=newff(x,t,[20],{},'trainbr');
net=train(net,x,t); 

%% test the performance and compare with the generating function

x=-1:.012:1;
t=sin(2*pi*x);
y=sim(net,x);
figure;
plot(x,t,'x');hold on;plot(x,y,'r-');
mse(y-t)
 