%%%%%%%%%%%%%%
%algorob.m
%A script comparing performance of on-line and batch steepest descent algorithms.
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
neto.adaptParam.passes=1; %set the number of passes in the function adapt (online training)
netb.trainParam.epochs=48; %set the number of epochs in the function train (batch training)
neto=adapt(neto,p,t); % online training
netb=train(netb,p,t); % batch training
ao1=sim(neto,p); ab1=sim(netb,p); % simulate the networks with the input vector

neto.adaptParam.passes=2;
netb.trainParam.epochs=96;
neto=adapt(neto,p,t);
netb=train(netb,p,t);
ao2=sim(neto,p); ab2=sim(netb,p);

neto.adaptParam.passes=10;
netb.trainParam.epochs=480;
neto=adapt(neto,p,t);
netb=train(netb,p,t);
ao3=sim(neto,p); ab3=sim(netb,p);

%plots
figure
subplot(3,3,1);
plot(x,y,'bx',x,cell2mat(ao1),'r',x,cell2mat(ab1),'g'); %plot the sine function and the output of the networks
title('48 epochs');
legend('target','on-line learning','batch learning',4);
subplot(3,3,2);
postregm(cell2mat(ao1),y); % perform a linear regression analysis
subplot(3,3,3);
postregm(cell2mat(ab1),y);
%
subplot(3,3,4);
plot(x,y,'bx',x,cell2mat(ao2),'r',x,cell2mat(ab2),'g');
title('144 epochs');
legend('target','on-line learning','batch learning',4);
subplot(3,3,5);
postregm(cell2mat(ao2),y);
subplot(3,3,6);
postregm(cell2mat(ab2),y);
%
subplot(3,3,7);
plot(x,y,'bx',x,cell2mat(ao3),'r',x,cell2mat(ab3),'g');
title('624 epochs');
legend('target','on-line learning','batch learning',4);
subplot(3,3,8);
postregm(cell2mat(ao3),y);
subplot(3,3,9);
postregm(cell2mat(ab3),y);
