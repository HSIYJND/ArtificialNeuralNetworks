% load the threes.mat file
load threes.mat -ascii
% to be compatible with our code from before, transpose it
threes = transpose(threes);

% get some of the dimensions
N = size(threes,2);
p = size(threes,1);

% compute and show the mean 3
meanthree = mean(threes, 2);
colormap(gray);
imagesc(reshape(meanthree,16,16),[0,1])

% zero mean the data
size(meanthree)
size(threes)
size(repmat(meanthree,1,N))
zmthrees = threes - repmat(meanthree,1,N);

% compute the covariance matrix
covmatrix = cov(transpose(zmthrees));

% calculate the eigenvalues and eigenvectors
[Eall,Dall] = eig(covmatrix);
Dall = diag(Dall);
figure
semilogy(Dall)
hold off;
close all;

% display six largest eigenvalues
[DallSorted,DallSortedIndices] = sort(Dall,'descend');

figure
for i = 1:6
    subplot(2,3,i);
    colormap(gray);
    imagesc(reshape(zmthrees(:,i),16,16),[0,1]);
    axis off;
end
savefig('basis.fig')
hold off;
close all;


%% Choose an image to do analysis on
TryImageIndex = 20;
qmax = 6
figure
for q=1:qmax
    % now compress
    [E,z,d] = PCAJannes(zmthrees, q);
    %reconstruct
    size(meanthree)
    TryImageReconstructed = E*z(:,TryImageIndex) + meanthree;
    subplot(1,qmax+1,q);
    colormap(gray);
    imagesc(reshape(TryImageReconstructed,16,16),[0,1]);
    axis off;
end
subplot(1,qmax+1,qmax+1);
imagesc(reshape(threes(:,TryImageIndex),16,16),[0,1])
axis off;
savefig('reconstruction_steps.fig')
hold off;
close all;


%% Now make the mse-k plot
kmax = 50
xplot = zeros(kmax,1);
yplot = zeros(kmax,1);

for k = 1:kmax
    xplot(k) = k;
    yplot(k) = CalcPCAReconError(zmthrees, k);
end

%% plot the eigenvalues via a cumsum
yplotcumsum = sum(DallSorted) - cumsum(DallSorted(1:k));
close all;
figure
plot(xplot, yplot, 'r-', xplot, yplotcumsum, 'b--', 'LineWidth', 2)
legend('Total reconstruction error','(total sum) - (cumsum \lambda_k)')
xlabel('k')
savefig('recon_and_cumsum.fig')

CalcPCAReconError(zmthrees, 256)