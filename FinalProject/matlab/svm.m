%% SVM
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

%% Fit Linear Kernel SVM
% =========================================================================
rsvm_linear = fitrsvm(train_imdb.images.data, train_imdb.images.label, ...
    'KernelFunction', 'linear');
cv_rsvm_linear = crossval(rsvm_linear, 'Kfold', 4);

time_start_linear = tic;
y_fit_linear = kfoldPredict(cv_rsvm_linear);
time_elapsed_linear = toc(time_start_linear);

mse_train_linear = mean((y_fit_linear - cv_rsvm_linear.Y).^2);

y_pred_linear = predict(rsvm_linear, test_imdb.images.data);
mse_test_linear = mean((y_pred_linear - test_imdb.images.label).^2);

display(time_elapsed_linear);
display(mse_train_linear);
display(mse_test_linear);

% Show first 5 results
table(test_imdb.images.label(1:5), y_pred_linear(1:5), 'VariableNames',...
    {'ObservedValue','PredictedValue'})

%% Fit Gaussian Kernel SVM
% =========================================================================
rsvm_gaussian = fitrsvm(train_imdb.images.data, train_imdb.images.label, ...
    'KernelFunction', 'gaussian');

cv_rsvm_gaussian = crossval(rsvm_gaussian, 'Kfold', 4);

time_start_gaussian = tic;
y_fit_gaussian = kfoldPredict(cv_rsvm_gaussian);
time_elapsed_gaussian = toc(time_start_gaussian);

mse_train_gaussian = mean((y_fit_gaussian - cv_rsvm_gaussian.Y).^2);

y_pred_gaussian = predict(rsvm_gaussian, test_imdb.images.data);
mse_test_gaussian = mean((y_pred_gaussian - test_imdb.images.label).^2);

display(time_elapsed_gaussian);
display(mse_train_gaussian);
display(mse_test_gaussian);

% Show first 5 results
table(test_imdb.images.label(1:5), y_pred_gaussian(1:5), 'VariableNames',...
    {'ObservedValue','PredictedValue'})
