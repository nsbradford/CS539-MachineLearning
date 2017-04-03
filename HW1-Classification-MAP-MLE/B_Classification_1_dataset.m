function [ output ] = B_Classification_1( )
%B_CLASSIFICATION_1 Summary of this function goes here
%   Detailed explanation goes here
% Sample 1: number of instances: 500, mean=60 and standard deviation=8.
% Sample 2: number of instances: 300, mean=30 and standard deviation=12.
% Sample 3: number of instances: 200, mean=80 and standard deviation=4.
%

size1 = 500;
size2 = 300;
size3 = 200;

s1 = randn(size1, 1) .* 8 + 60.0;
s2 = randn(size2, 1) .* 12 + 30.0;
s3 = randn(size3, 1) .* 4 + 80.0;

r1 = ones(size1, 1) * 1;
r2 = ones(size2, 1) * 2;
r3 = ones(size3, 1) * 3;

data1 = horzcat(s1, r1);
data2 = horzcat(s2, r2);
data3 = horzcat(s3, r3);

output = vertcat(data1, data2, data3);

end

