%% read data
% =========================================================================
filename = '2016-01-30--11-24-51.h5';

%% load log data in one matrix
% =========================================================================
%{
log_path = './data/log/';
log_filename = strcat(log_path, filename);
log_HDF5_info = h5info(log_filename);
log_HDF5_keys = struct2cell(log_HDF5_info.Datasets);
log_data = [];
for n = log_HDF5_keys
    key = n{1};
    disp(strcat("Loading ", key, " into matrix..."));
    data = h5read(log_filename, strcat('/', key));
end
%}

%% load camera data
% =========================================================================
camera_path = './data/camera/';
camera_filename = strcat(camera_path, filename);
camera_info = h5info(camera_filename);
camera_data = camera_info.Datasets(1);

for n = 1:5500
    % http://stackoverflow.com/q/42137631/3208877
    image_data = h5read(camera_filename, '/X', [1 1 1 n], [320 160 3 1]);
    rotated_image = imrotate(image_data, -90);
    imshow(rotated_image);
end





