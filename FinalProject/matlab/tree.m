%% Decision Trees
% =========================================================================
rng(1);

%% Initialize the imdb structure for train and test (image database)
% =========================================================================
train_filename = '2016-01-30--11-24-51.h5'; % Max # of images in file 52722
test_filename = '2016-02-08--14-56-28.h5'; % Max # of images in file 25865 
N = 1000;
label_key = 'steering_angle';
label_path = './data/log/';
image_path = './data/camera/';

offset = 4500;
train_imdb = cnnImdb(...
    N, offset, train_filename, label_key, label_path, image_path, false, true);
test_imdb = cnnImdb(...
    N, offset, test_filename, label_key, label_path, image_path, false, true);

%% Fit Tree
% =========================================================================
rtree = fitrtree(train_imdb.images.data, train_imdb.images.label);
cv_rtree = crossval(rtree, 'Kfold', 4);

time_start = tic;
y_fit = kfoldPredict(cv_rtree);
time_elapsed = toc(time_start);

mse_train = mean((y_fit - cv_rtree.Y).^2);

y_pred = predict(rtree, test_imdb.images.data);
mse_test = mean((y_pred - test_imdb.images.label).^2);

display(time_elapsed);
display(mse_train);
display(mse_test);

%% Fit Pruned Tree
% =========================================================================
rtree = fitrtree(train_imdb.images.data, train_imdb.images.label);
cv_rtree = crossval(rtree, 'Kfold', 4, 'Prune', 'on');

time_start = tic;
y_fit = kfoldPredict(cv_rtree);
time_elapsed = toc(time_start);

mse_train = mean((y_fit - cv_rtree.Y).^2);

y_pred = predict(rtree, test_imdb.images.data);
mse_test = mean((y_pred - test_imdb.images.label).^2);

display(time_elapsed);
display(mse_train);
display(mse_test);
