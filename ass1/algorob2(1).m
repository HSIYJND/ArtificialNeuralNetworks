%%%%%%%%%%%%%%
%algorob.m
%A script comparing performance of on-line and batch steepest descent algorithms.
%smaller number of figures
%%%%%%%

%generation of examples
x=0:0.2:3*pi; y=sin(x);
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

%creation of networks
neto=newff([0 3*pi],[5 1],{'tansig','purelin'},'traingd','learngd');
netb=newff([0 3*pi],[5 1],{'tansig','purelin'},'traingd','learngd');
netb.iw{1,1}=neto.iw{1,1}; % set the same weights for the networks
netb.lw{2,1}=neto.lw{2,1};
netb.b{1}=neto.b{1}; %set the same bias
netb.b{2}=neto.b{2};

%training and simulation
neto.adaptParam.passes=10; %set the number of passes in the function adapt (online training) 
netb.trainParam.epochs=10; %set the number of epochs in the function train (batch training)
neto=adapt(neto,p,t); % online training
netb=train(netb,p,t);  % batch training
ao1=sim(neto,p); ab1=sim(netb,p); % simulate the networks with the input vector

neto.adaptParam.passes=100; 
netb.trainParam.epochs=100; 
neto=adapt(neto,p,t);
netb=train(netb,p,t);
ao2=sim(neto,p); ab2=sim(netb,p);

neto.adaptParam.passes=400;
netb.trainParam.epochs=400;
neto=adapt(neto,p,t);
netb=train(netb,p,t);
ao3=sim(neto,p); ab3=sim(netb,p);

%plots
figure
subplot(3,1,1);
plot(x,y,'bx',x,cell2mat(ao1),'r',x,cell2mat(ab1),'g'); %plot the sine function and the output of the networks
title('10 epochs');
legend('target','on-line learning','batch learning',4);
%
subplot(3,1,2);
plot(x,y,'bx',x,cell2mat(ao2),'r',x,cell2mat(ab2),'g');
title('100 epochs');
legend('target','on-line learning','batch learning',4);
%
subplot(3,1,3);
plot(x,y,'bx',x,cell2mat(ao3),'r',x,cell2mat(ab3),'g');
title('400 epochs');
legend('target','on-line learning','batch learning',4);
