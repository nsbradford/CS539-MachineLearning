%% CNN
% =========================================================================
rng(1);

%% Initialize the imdb structure (image database)
% =========================================================================
filename = '2016-01-30--11-24-51.h5';
N = 5000; % Max # of images in file 52722
offset = 4500; % Start of highway driving
label_key = 'steering_angle';
label_path = './data/log/';
image_path = './data/camera/';
imdb = cnnImdb(N, offset, filename, label_key, label_path, image_path);

train_indices = find(imdb.images.set == 1); % 1 = 75% train
val_indices = find(imdb.images.set == 2); % 2 = 25% val
[train_image, train_label] = getBatch(imdb, train_indices);
[val_image, val_label] = getBatch(imdb, val_indices);

%% Train CNN
% =========================================================================
layers = [ ...
    imageInputLayer([160 320 3])
    convolution2dLayer([8 8], 16, 'Stride', [4 4], 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    convolution2dLayer([5 5], 32, 'Stride', [2 2], 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    convolution2dLayer([5 5], 64, 'Stride', [2 2], 'Name', 'conv3')
    dropoutLayer(0.2, 'Name', 'dropout1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.5, 'Name', 'dropout2')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 15, ...
    'CheckpointPath', './checkpoints');

net = trainNetwork(train_image, train_label, layers, options);

%% Test CNN
% =========================================================================
predicted_labels = predict(net, val_image);
prediction_error = val_label - predicted_labels;

thr = 10;
num_correct = sum(abs(prediction_error) < thr);
num_test_images = size(val_image, 4);

accuracy = num_correct / num_test_images;

squares = prediction_error.^2;
rmse = sqrt(mean(squares));

disp(accuracy);
disp(rmse);

%% Functions
% =========================================================================
function [image, label] = getBatch(imdb, batch)
    %GETBATCH  Get a batch of training data
    %   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
    %   and labels LABEL from IMDB according to the list of images
    %   BATCH.

    image = imdb.images.data(:,:,:,batch);
    label = imdb.images.label(batch);
end
