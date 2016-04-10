% generate 50x500 matrix of normally distributed values
x = randn(50,500);
x = x + 1; % shift it

xplot = zeros(50,1);
yplot = zeros(50,1);

% make x zero mean and calculate the covariance matrix
V = cov(transpose(x - repmat(mean(x,2), 1, 500)));
[v,d] = eig(V);
fulleigvals = diag(d);

% do PCA
for q = 1:50
    [E,z, d] = PCAJannes(x, q);
    
    % reconstruct x
    errorx = x - (E*z + repmat(mean(x, 2),1,500));
    
    % calculate the distance
    mse = sum(sum(errorx.^2));
    
    % save it
    xplot(q) = q;
    yplot(q) = sum(fulleigvals) - sum(d);
end

semilogy(xplot, yplot)