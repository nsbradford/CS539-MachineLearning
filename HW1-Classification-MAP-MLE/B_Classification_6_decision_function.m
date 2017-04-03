function [ output_args ] = B_Classification_2_decision_function( data, x )
%B_CLASSIFICATION_2_ Summary of this function goes here
%   Detailed explanation goes here

ind1 = data(:,2) == 1;
ind2 = data(:,2) == 2;
ind3 = data(:,2) == 3;

data1 = data(ind1, 1);
data2 = data(ind2, 1);
data3 = data(ind3, 1);

mu1 = mean(data1); % sum(data1) / length(data1);
mu2 = mean(data2);
mu3 = mean(data3);
sigma1 = std(data1);
sigma2 = std(data2);
sigma3 = std(data3);

results1 = B_Classification_2_discriminant(mu1, sigma1, x, 0.5);
results2 = B_Classification_2_discriminant(mu2, sigma2, x, 0.3);
results3 = B_Classification_2_discriminant(mu3, sigma3, x, 0.2);

results = horzcat(results1, results2, results3);
[max_vals, max_inds] = max(results,[],2);

output_args = max_inds;

% plot likelihoods: p(x|Ci)
figure;
subplot(1,2,1)  
hold on;
range = [0:.5:100];
plot(range, normpdf(range, mu1, sigma1));
plot(range, normpdf(range, mu2, sigma2));
plot(range, normpdf(range, mu3, sigma3));
hold off;

% plot Posteriors: p(Ci|x) = p(x|Ci)p(Ci)
subplot(1,2,2)  
hold on;
% plot(range, B_Classification_2_discriminant(mu1, sigma1, range, 0.5) .* 0.5);
% plot(range, B_Classification_2_discriminant(mu2, sigma2, range, 0.3) .* 0.5);
% plot(range, B_Classification_2_discriminant(mu3, sigma3, range, 0.2) .* 0.3);
denom1 = normpdf(range, mu1, sigma1) .* 0.5;
denom2 = normpdf(range, mu2, sigma2) .* 0.3;
denom3 = normpdf(range, mu3, sigma3) .* 0.2;
allofem = denom1 + denom2 + denom3;
plot(range, denom1 ./ allofem);
plot(range, denom2 ./ allofem);
plot(range, denom3 ./ allofem);
hold off

end

