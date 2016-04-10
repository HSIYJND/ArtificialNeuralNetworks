clear all
close all

% load the data
load 'data/Data_Problem1_regression.mat'

% generate my data
d1 = 9;
d2 = 8;
d3 = 7;
d4 = 6;
d5 = 3;
Tnew = zeros(size(T1));
for i = 1:size(Tnew,1)
    Tnew(i) = (d1*T1(i) + d2*T2(i) + d3*T3(i) + d4*T4(i) + d5*T5(i))/(d1 + d2 + d3 + d4 + d5);
end

data.X1 = X1;
data.X2 = X2;
data.T = Tnew;

save('data/function_data.mat','data')