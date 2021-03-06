%% Meta Learning
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

%% Random Forest Bagging
% =========================================================================
% https://www.mathworks.com/help/stats/regression-tree-ensembles.html
rensemble_bagging = fitrensemble(...
    train_imdb.images.data, train_imdb.images.label, 'Bag');
cv_rensemble_bagging = crossval(rensemble_bagging, 'Kfold', 4);

time_start_bagging = tic;
y_fit_bagging = kfoldPredict(cv_rensemble_bagging);
time_elapsed_bagging = toc(time_start_bagging);

predicted_labels_bagging = predict(y_fit_bagging, test_imdb.images.data);
prediction_error_bagging = test_imdb.images.label - predicted_labels_bagging;

squares_bagging = prediction_error_bagging.^2;
mse_bagging = mean(squares_bagging);

disp(mse_bagging);

%% LSBoost
% =========================================================================
rensemble_boosting = fitrensemble(...
    train_imdb.images.data, train_imdb.images.label, 'LSBoost');
cv_rensemble_boosting = crossval(rensemble_bagging, 'Kfold', 4);

time_start_boosting = tic;
y_fit_boosting = kfoldPredict(cv_rensemble_boosting);
time_elapsed_boosting = toc(time_start_bagging);

predicted_labels_boosting = predict(y_fit_boosting, test_imdb.images.data);
prediction_error_boosting = test_imdb.images.label - predicted_labels_boosting;

squares_boosting = prediction_error_boosting.^2;
mse_boosting = mean(squares_boosting);

disp(mse_boosting);
