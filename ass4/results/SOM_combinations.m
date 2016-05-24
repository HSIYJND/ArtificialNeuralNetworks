%% perform unsupervised learning with SOM  
close all;
clear all;
clc;

topologies = char('gridtop','hextop','randtop');
measures = char('dist','boxdist','linkdist','mandist');
clusterperformance=zeros(size(topologies,1),size(measures,1));
Niter=100;
Ncheck=100;

for itop=1:size(topologies,1)
    for imeas=1:size(measures,1)
        rng(1);
        
        top=topologies(itop);
        meas=measures(imeas);

        parfor it=1:Niter
            % first we generate data uniformely distributed within two
            % concentric cylinders

            X=2*(rand(5000,3)-.5);
            indx=(X(:,1).^2+X(:,2).^2<.6)&(X(:,1).^2+X(:,2).^2>.1);
            X=X(indx,:)';

            % we then initialize the SOM with hextop as topology function
            % and linkdist as distance function
            net = newsom(X,[5 5 5],top,meas); 

            % finally we train the network and see how their position changes
            net.trainParam.epochs = 1000;
            net = train(net,X);
            
            % verify 
            for icheck=1:Ncheck
                
                
            end
        end
    end
end

