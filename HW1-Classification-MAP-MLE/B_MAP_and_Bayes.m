function [ expected_value ] = B_MAP_and_Bayes( input_args )
%B_ Summary of this function goes here
%   Detailed explanation goes here

m = mean(input_args);
sigma = 8;
mu0 = 60;
sigma0 = 3;
N = length(input_args);
var = sigma^2;
var0 = sigma0^2;

prior_weight = (N / var) / (N / var + 1 / var0) * m;
evidence_weight = (1 / var0) / ( N / var + 1 / var0) * mu0;
expected_value = prior_weight + evidence_weight;

end

