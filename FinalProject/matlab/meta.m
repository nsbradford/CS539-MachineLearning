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

