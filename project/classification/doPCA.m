function [E,z,d,meanVec,stddevVec,stdX] = doPCA(x, q)
% DOPCA  Do a Principle Component Analysis on a data set x
%   [E,z,mean,stddev] = DOPCA(x,q) with x a dataset with each row a data entry, 
%   performs PCA with result z and matrix E.
%   q is the reduced dimension (dimension of z)
%
%   See also PCA.

% get the dimension p of x and the number of datapoints N
p = size(x,2);
assert( q <= p );
N = size(x,1);

% calculate the mean for all p data properties
meanVec = mean(x);
stddevVec = std(x);

% rescale and shift with these vectors
stdX = zeros(N, 11);
for i = 1:N
    stdX(i,:) = x(i,:) - meanVec;       % shift
    stdX(i,:) = stdX(i,:) ./ stddevVec;  % rescale
end

% calculate the covariance matrix of the standardized data set
V = cov(stdX);

% calculate the eigenvectors and eigenvalues
% E is the matrix with columns the eigenvectors
[E,d] = eig(V); % calculate q largest eigenvalues and eigenvecs
d = diag(d);

% now sort them in descending order and take only q of them
[d, indices] = sort(d, 'descend');
d = d(1:q);
E = E(:,indices(1:q));

% reduce the data set my multiplying with this matrix
z=stdX*E;


