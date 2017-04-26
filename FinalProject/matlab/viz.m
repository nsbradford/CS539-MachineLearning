%% RGB Color Distribution
% =========================================================================
train_filename = './data/camera/2016-01-30--11-24-51.h5';
test_filename = './data/camera/2016-02-08--14-56-28.h5';
offset = 4700;
N = 5000;
%{
red = zeros(160, 320, 1, N, 'single');
green = zeros(160, 320, 1, N, 'single');
blue = zeros(160, 320, 1, N, 'single');

disp("Creating color channel matrices...");
for i = 1:N
    disp(i);
    image_data = h5read(test_filename, '/X', [1 1 1 offset + i], [320 160 3 1]);
    rotated_image = imrotate(image_data, -90); 
    single_image = im2single(rotated_image);
    red(:,:,:,i) = single_image(:,:,1,:);
    green(:,:,:,i) = single_image(:,:,2,:);
    blue(:,:,:,i) = single_image(:,:,3,:);
end
%}
disp("Creating red channel histogram...");
[y_red, x] = imhist(red);
disp("Creating green channel histogram...");
[y_green, x] = imhist(green);
disp("Creating blue channel histogram...");
[y_blue, x] = imhist(blue);

disp("Plotting...");
plot(x, y_red, 'Red', x, y_green, 'Green', x, y_blue, 'Blue', 'LineWidth', 2);
set(gca,'FontSize', 15, 'FontWeight', 'bold');
hold on;
title('RGB Channel Distribution (Test)')
xlabel('Color'); ylabel('Count')
hold off;

%% Steering Angle Distribution
% =========================================================================
train_filename = '2016-01-30--11-24-51.h5'; % 52717 log records
test_filename = '2016-02-08--14-56-28.h5'; % 25869 log records
N = 25869;
label_key = '/steering_angle';
label_path = './data/log/';
log_filename = strcat(label_path, test_filename);
label_data = h5read(log_filename, label_key);

labels = zeros(N, 1);
for i = 1:N
      [start, ~] = get_label_indices(i);
      label = get_label_start(start, label_data);
      labels(i) = label;
end

norm_labels = labels - min(labels(:));
norm_labels = norm_labels ./ max(norm_labels(:));
x = 1; y = -1;
norm_range = y - x;
norm_labels = (norm_labels * norm_range) + x;

nbins = 10;
figure
hist(norm_labels, nbins);
h = findobj(gca, 'Type', 'patch');
set(h, 'FaceColor', [0 0.5 0.5]);
set(gca,'FontSize', 15, 'FontWeight', 'bold');
hold on;
title('Steering Angle Distribution (Test)')
xlabel('Steering Angle'); ylabel('Count')
hold off;

function [start, stop] = get_label_indices(image_number)
    sampling_difference = 5;
    start = ((image_number - 1) * sampling_difference) + 1;
    stop = start + (sampling_difference - 1);
end

function start = get_label_start(n, log_data)
    start = log_data(n);
end