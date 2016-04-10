%% Lossy compression of an hyperspectral  image 
% 
% Marco Signoretto, March 2011



close all;
clear all;
clc;



disp('In this demo we use PCA to compress an hyperspectral  image.');disp([' ']); 

%% let us fix the number of principal components to retain
n_factors=input('# factors to retain? (press enter for default=2) \n');
if isempty(n_factors);
   n_factors=2; 
end

str_url = 'http://personalpages.manchester.ac.uk/staff/david.foster/Hyperspectral_images_of_natural_scenes_04_files/scenes/';

name_file=input('file name? (press enter for default=scene7.zip) \n','s');
if isempty(name_file);
   name_file='scene7.zip';
end



disp([' ']);disp('downloading and unzipping files from repository...this might take a while!');

urlwrite([str_url,name_file], 'zipped.zip');
filename=unzip('zipped.zip');


for i=1:numel(filename); 
    try 
        load(filename{i}); 
    end
end
save reflectances reflectances;

sizes=size(reflectances);

X_orig=double(reshape(reflectances,sizes(1)*sizes(2),sizes(3)));
disp('finished!');

disp('let us check how the first image of the  ensamble looks like.');
figure;
imagesc(reshape(X_orig(:,1),sizes(1),sizes(2)));
colormap(gray);
drawnow;
disp('press a key to continue.');
pause;


mean_=mean(X_orig,2);
mean_mat=double(repmat(mean_,1,sizes(3)));

X=X_orig-mean_mat;

disp('compute PCA of hyperspectral image...');

cov_X=cov(X);
[E s]=eig(cov_X);

disp('done');

[s indx]=sort(diag(s),'descend');
E=E(:,indx);

proj_mat=E(:,1:n_factors)';

Z=proj_mat*X';

%% save all you need for the (lossy) reconstruction of the hyperspectral image
disp('save all you need for the (lossy) reconstruction of the hyperspectral image...');
save  reflectances-compress Z proj_mat mean_ sizes;
save ffnames filename;
disp('done');

%% start from scratch and reconstruct the hyperspectral image from what you have just saved
clear all;
disp('let us now clear the workspace and reconstruct the hyperspectral image from scratch...');

load reflectances-compress;
load ffnames filename;

X_=proj_mat'*Z;
 
mean_mat=double(repmat(mean_,1,sizes(3)));
X_=X_'+mean_mat; 

%% compute reconstruction error

load reflectances;
disp('done. computing reconstruction error...');
NRMSE=sqrt(mean((reflectances(:)-X_(:)).^2))/(max(reflectances(:))-min(reflectances(:)));
disp([' ']);disp(['normalized root mean square error = ',num2str(NRMSE)]);
s=dir('reflectances.mat');
sc=dir('reflectances-compress.mat');        
const=9.53674316e-7;
disp([' ']);disp(['The size of the original reflectances file is: ',num2str(s.bytes*const),' MB.']);    
disp(['The size of the "compressed" file is: ',num2str(sc.bytes*const),' MB.']);

%% check how the first reconstructed matrix looks like

disp([' ']);disp('finally let us check how the first image of the reconstructed ensemble looks like.');
disp('press a key to continue.');
pause;
figure;
imagesc(reshape(X_(:,1),sizes(1),sizes(2)));
colormap(gray);
drawnow;



% finally delete the files that were created
for i=1:numel(filename);delete(filename{i}); end
delete 'reflectances.mat' ;
delete 'reflectances-compress.mat';
delete 'ffnames.mat';