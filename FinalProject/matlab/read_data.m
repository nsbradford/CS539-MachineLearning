%% read data
% =========================================================================
filename = '2016-01-30--11-24-51.h5';

%% load log data
% =========================================================================
log_path = './data/log/';
log_filename = strcat(log_path, filename);
%{
log_HDF5_info = h5info(log_filename);
log_HDF5_keys = struct2cell(log_HDF5_info.Datasets);
log_data = [];
for n = log_HDF5_keys
    key = n{1};
    disp(strcat("Loading ", key, " into matrix..."));
    data = h5read(log_filename, strcat('/', key));
end
%}

steering_angle_data = h5read(log_filename, '/steering_angle');

%% load camera data
% =========================================================================
camera_path = './data/camera/';
camera_filename = strcat(camera_path, filename);
camera_info = h5info(camera_filename);
camera_data = camera_info.Datasets(1);

%% save images to file
% =========================================================================
%%{
start = 4470;
stop = 4480;
min = 0;
max= 52722;
for n = start:stop
    % http://stackoverflow.com/q/42137631/3208877
    image_data = h5read(camera_filename, '/X', [1 1 1 n], [320 160 3 1]);
    rotated_image = imrotate(image_data, -90); 
    imshow(rotated_image);
    imwrite(rotated_image, strcat('./data/images/', num2str(n), '.png'));
    
    [st, en] = get_log_numbers(n);
    average = get_average_log_numbers(st, en, steering_angle_data);
    output = strcat(...
        "#", num2str(n), " - ",...
        "start: ", num2str(steering_angle_data(st)), ", ",...
        "end: ", num2str(steering_angle_data(en)), ", ",...
        "average: ", num2str(average));
    disp(output);
end
%%}

function [st, en] = get_log_numbers(image_number)
    st = ((image_number - 1) * 5) + 1;
    en = st + 4;
end 

function average = get_average_log_numbers(start, stop, log_data)
    sum = 0;
    for n = start:stop
        sum = sum + log_data(n);
    end
    average = sum / 5;
end 
