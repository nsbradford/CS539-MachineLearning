%% Meta Learning
% =========================================================================
rng(1);

%% Initialize the imdb structure for train and test (image database)
% =========================================================================
train_filename = '2016-01-30--11-24-51.h5'; % Max # of images in file 52722
test_filename = '2016-02-08--14-56-28.h5'; % Max # of images in file 25865 
N = 100;
label_key = 'steering_angle';
label_path = './data/log/';
image_path = './data/camera/';

offset = 4500;
train_imdb = cnnImdb(...
    N, offset, train_filename, label_key, label_path, image_path, false);
test_imdb = cnnImdb(...
    N, offset, test_filename, label_key, label_path, image_path, false);

%% AdaBoost
% =========================================================================

%% Random Forest
% =========================================================================
% https://www.mathworks.com/help/stats/regression-tree-ensembles.html
rensemble = fitrensemble(train_imdb.images.data, train_imdb.images.label);

time_start = tic;
Y_predicted = predict(rensemble, test_imdb.images.data);
time_elapsed = toc(time_start);

display(time_elapsed);
table(test_imdb.images.label(1:10), Y_predicted, 'VariableNames',...
    {'ObservedValue','PredictedValue'})



