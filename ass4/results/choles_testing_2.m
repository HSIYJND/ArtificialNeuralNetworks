load choles_all

x = p; % put it in my format
p = size(x,1)
N = size(x,2)

xplot = zeros(p,1);
yplot = zeros(p,1);

% make x zero mean and calculate the covariance matrix
V = cov(transpose(x - repmat(mean(x,2), 1, N)));
[v,d] = eig(V);
fulleigvals = diag(d);

% do PCA
for q = 1:p
    [E,z, d] = PCAJannes(x, q);
    
    % reconstruct x
    errorx = x - (E*z + repmat(mean(x, 2),1,N));
    
    % calculate the distance
    mse = sum(sum(errorx.^2));
    
    % save it
    xplot(q) = q;
    yplot(q) = mse;
    %yplot(q) = sum(fulleigvals) - sum(d);
end

semilogy(xplot, yplot)
hold on

% now with the standard methods (only mean=0 and std=1)
x = mapstd(x);

% do PCA
for q = 1:p
    [E,z,d] = PCAJannes(x, q);
    
    % reconstruct x
    errorx = x - (E*z + repmat(mean(x, 2),1,N));
    
    % calculate the distance
    mse = sum(sum(errorx.^2));
    
    % save it
    xplot(q) = q;
    yplot(q) = mse;
    %yplot(q) = sum(fulleigvals) - sum(d);
end

semilogy(xplot, yplot, 'r--')
hold on

% do PCA
z = processpca('apply',x)

% reconstruct x
errorx = x - processpca('reverse',z, 0.02);

% calculate the distance
mse = sum(sum(errorx.^2));

% save it
hline = refline([0 mse]);