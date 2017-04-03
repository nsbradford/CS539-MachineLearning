function [ confusion ] = B_Classification_2_test( )
%B_CLASSIFICATION_2_TEST Summary of this function goes here
%   Detailed explanation goes here

data = B_Classification_1_dataset();
% x = [10, 30, 50, 70, 90]';
N = length(data);
trainRatio = 0.6;
trainInds = randperm(N, trainRatio * N);
testInds = setdiff(linspace(1, N, N), trainInds);

train_data = data(trainInds, :);
test_data = data(testInds, :);
actual = test_data(:, 2);
results = B_Classification_6_decision_function(train_data, test_data(:, 1));

accuracy = NaN; %classperf(actual, results);
confusion = confusionmat(actual, results);

end

