%% define the attractor points
T = [1 1; -1 -1; 1 -1]'

%% create a hopfield network
net = newhop(T);

% visualize
hold on
plot(T(1,:),T(2,:),'r*')

% now generate random samples and evolve
for pointNr = 1:50
    starter = {rands(2,1)};
    [y Pf final] = net({20}, {}, starter);
    cell2mat(final)
    %here, y are all the next few steps, Af is the final one
    % append all the points
    sequence=[cell2mat(starter) cell2mat(y)];
    
    % plot that
    starter=cell2mat(starter);
    plot(starter(1,1),starter(2,1), 'kx', sequence(1,:),sequence(2,:), 'LineWidth',2)
end


%% now calculate average step length
