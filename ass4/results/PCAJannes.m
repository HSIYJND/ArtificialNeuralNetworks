function [E,z,d] = PCAJannes(x, q)
% PCAJANNES  Do a Principle Component Analysis on a data set x
%   [E,z] = PCAJANNES(x,q) with x a dataset with each column a data entry, 
%   performs PCA with result z and matrix E.
%   q is the reduced dimension (dimension of z)
%
%   See also PCA.

% get the dimension p of x and the number of datapoints N
p = size(x,1);
N = size(x,2);

% calculate mean and stddev
meanX = mean(x,2);
%stddevX = std(x,0,2);

% subtract the mean vector from x
for i = 1:N
    x(:,i) = x(:,i) - meanX;
    %x(:,i) = x(:,i) ./ stddevX;
end

% make x zero mean and calculate the covariance matrix
V = cov(x');

% calculate the eigenvectors and eigenvalues
% E is the matrix with columns the eigenvectors
[E,d] = eigs(V,q);
d=diag(d);
% reduce the data set my multiplying with this matrix
z=transpose(E)*x;
