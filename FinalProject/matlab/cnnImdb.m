function imdb = cnnImdb(N, filename, label, label_path, image_path)
    % =====================================================================
    % Initialize the imdb structure (image database).
    % Note the fields are arbitrary: only your getBatch needs to understand it.
    % The field imdb.set is used to distinguish between the training and
    % validation sets, and is only used in the above call to cnn_train.
    % Based on http://www.robots.ox.ac.uk/~joao/cnn_toy_data.m

    rng('default');
    rng(0);

    log_filename = strcat(label_path, filename);
    camera_filename = strcat(image_path, filename);
    
    label_key = strcat('/', label);
    label_data = h5read(log_filename, label_key);
    
    % Preallocate memory
    images = zeros(160, 320, 3, N, 'single');
    labels = zeros(N, 1);
    set = ones(N, 1);

    % Generate N sample images.
    % Note that here one could load the images from files, and do any kind of
    % necessary pre-processing.
    for i = 1:N
      [start, stop] = get_label_indices(i);
      average = get_label_average(start, stop, label_data);
      labels(i) = average;

      % Read single image from .h5
      % http://stackoverflow.com/q/42137631/3208877
      image_data = h5read(camera_filename, '/X', [1 1 1 N], [320 160 3 1]);
      rotated_image = imrotate(image_data, -90); 
      single_image = im2single(rotated_image);
      images(:,:,:,i) = single_image;

      % Mark last 10% of samples as part of the validation set
      if i > 0.9 * N
        set(i) = 2;
      end
    end

    % Show some example images
    figure(2); montage(images(:,:,:,1:100)); title('Example images');

    % Remove mean over whole dataset
    images = bsxfun(@minus, images, mean(images, 4));

    % Store results in the imdb struct
    imdb.images = images;
    imdb.labels = labels;
    imdb.set = set;
    
    function [start, stop] = get_label_indices(image_number)
        % =====================================================================
        % Find log file indices associated with image
        % Note sampling difference is equal to 100 HZ sampling rate divided by
        % 20 HZ image sampling rate (5 log files per image)
        sampling_difference = 5;
        start = ((image_number - 1) * sampling_difference) + 1;
        stop = start + (sampling_difference - 1);
    end

    function average = get_label_average(start, stop, log_data)
        % =====================================================================
        % Get average of log files for start and stop indices
        sum = 0;
        for n = start:stop
            sum = sum + log_data(n);
        end
        average = sum / 5;
    end
end