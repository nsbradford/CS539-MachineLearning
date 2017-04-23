
%% Set up MatConvNet
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cd matconvnet-1.0-beta24
run matlab/vl_setupnn;
cd ..

%% Initialize the imdb structure (image database)
% =========================================================================
N = 200; % Max # of images in file: 52722
offset = 4469; % Start of highway driving
filename = '2016-01-30--11-24-51.h5';
label_key = 'steering_angle';
label_path = './data/log/';
image_path = './data/camera/';
imdb = cnnImdb(N, offset, filename, label_key, label_path, image_path);

%% Initialize the net
% =========================================================================
net = cnnInit();
vl_simplenn_display(net);
res = vl_simplenn(net, imdb.images.data(:,:,:,1));

figure(32); clf; colormap gray;
set(gcf,'name', 'Part 3.2: network input');
subplot(1,2,1);
imagesc(res(1).x); axis image off;
title('CNN input');

set(gcf,'name', 'Part 3.2: network output');
subplot(1,2,2);
imagesc(res(end).x); axis image off;
title('CNN output (not trained yet)');

%% Learn the model
% =========================================================================
trainOpts.expDir = 'data/2016-01-30--11-24-51';
trainOpts.gpus = [];
trainOpts.batchSize = 10;
trainOpts.learningRate = 0.02;
trainOpts.plotDiagnostics = false;
trainOpts.numEpochs = 5;
trainOpts.errorFunction = 'none';

net = cnn_train(net, imdb, @getBatch, trainOpts);

%% Evaluate the model
% =========================================================================
train = find(imdb.images.set == 1);
val = find(imdb.images.set == 2);

[res_train, preds_train] = cnnPredict(net, imdb, train(1:30:151)); % Training results
[res_test, preds_test] = cnnPredict(net, imdb, val(1:30:151)); % Validation results

%% Functions
% =========================================================================
function [im, label] = getBatch(imdb, batch)
    %GETBATCH  Get a batch of training data
    %   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
    %   and labels LABEL from IMDB according to the list of images
    %   BATCH.

    im = imdb.images.data(:,:,:,batch);
    label = imdb.images.label(batch);
end




















