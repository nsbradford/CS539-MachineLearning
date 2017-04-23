
%% Run CNN
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% Initialize the imdb structure (image database).
% =========================================================================
N = 1000; % Max # of images in file: 52722
filename = '2016-01-30--11-24-51.h5';
label_key = 'steering_angle';
label_path = './data/log/';
image_path = './data/camera/';
imdb = cnnImdb(1000, filename, label_key, label_path, image_path);

%% Train model
% =========================================================================
% [net, info] = cnnTrain(imdb);

%% Test model
% =========================================================================

%% Make prediction for single image
% =========================================================================
% image_path = './data/images/4470.png';
% imshow(image_path);
% img = imread(image_path);
% label = cnnPredict(cnnModel, img);
% title(label, 'FontSize', 20);