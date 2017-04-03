function [ output ] = B_data_generation( )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

my_stdev = 8;
my_mean = 60;
output = my_stdev .* randn(1000,1) + my_mean;

end



