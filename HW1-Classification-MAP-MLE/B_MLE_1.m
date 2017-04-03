function [ output_mean, output_std ] = B_MLE_1( input )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

input_sum = sum(input);
N = length(input);
output_mean = input_sum / N;

output_var = sum((input - output_mean).^2) / N;
output_std = sqrt(output_var);

end

