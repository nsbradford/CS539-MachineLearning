function [ c, performance, net ] = HW3_B_NeuralNet( trainingData, testData )
%HW3_B_NEURALNET Summary of this function goes here
%   Detailed explanation goes here

rng(1) % For reproducibility

trainX = trainingData(1:64, :);
trainY = ind2vec(trainingData(65, :) + 1, 10);
testX = testData(1:64, :);
testY = ind2vec(testData(65, :) + 1, 10);

% Create a Pattern Recognition Network
hiddenLayerSize = [200, 40];
net = patternnet(hiddenLayerSize);
% net.trainFcn = 'traingd'; % 'trainscg' is default

% net.trainParam.epochs	1000	
% Maximum number of epochs to train
% net.trainParam.goal	0	
% Performance goal
% net.trainParam.showCommandLine	false	
% Generate command-line output
% net.trainParam.showWindow	true	
% Show training GUI
% net.trainParam.lr	0.01	
% Learning rate
% net.trainParam.max_fail	6	
% Maximum validation failures
% net.trainParam.min_grad	1e-5	
% Minimum performance gradient
% net.trainParam.show	25	
% Epochs between displays (NaN for no displays)
% net.trainParam.time	inf	

%net.trainParam.epochs = 400;
%net.trainParam.lr = 0.01;

% net.performParam.regularization = 1e-5;
% net.performParam.normalization;

% Train the Network
[net,tr] = train(net, trainX, trainY);

% Test the Network
outputs = net(testX);
errors = gsubtract(testY, outputs);
performance = perform(net, testY, outputs);
[c,cm,ind,per] = confusion(testY, outputs);
% c=% misclassified, cm=matrix,

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
figure, plotconfusion(testY,outputs)
%figure, ploterrhist(errors)

end

