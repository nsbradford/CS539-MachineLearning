function [ output_args ] = B_Classification_2_discriminant( mu, sigma, x, p_Ci )
%B_CLASSIFICATION_2 Summary of this function goes here
%   Detailed explanation goes here

% term1 = -0.5 * log(2 * pi) - log(sigma);
% term3 =  - ((x - mu) .^2 ./ (2 * sigma^2));
% term4 = log(p_Ci);
% output_args = term1 + term3 + term4;

output_args = -0.5 * log(2 * pi) - log(sigma) - ((x - mu) .^2 ./ (2 * sigma^2)) + log(p_Ci);

end
