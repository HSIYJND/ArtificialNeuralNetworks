function mse = CalcPCAReconError(x, q)
% CalcPCAReconError  Do a Principle Component Analysis on a data set x
%   [E,z] = CalcPCAReconError(x,q) with x a dataset with each column a data entry, 
%   performs PCA with result z and matrix E.
%   q is the reduced dimension (dimension of z)
%
%   See also PCA.

% get the number of datapoints N
N = size(x,2);

% project on the PCA basis
[E,z,~]=PCAJannes(x, q);

% now reconstruct and calculate the error
% reconstruct x
errorx = x; % for the shape
meanx = mean(x, 2);
stdx = std(x,0,2); % N-1
for i = 1:N
    errorx(:,i) = x(:,i) - ((E*z(:,i)).*stdx  + meanx);
end

% calculate the distance
mse = sum(sum(errorx.^2))/N; % normalize on the total nr of data points