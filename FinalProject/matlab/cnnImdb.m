function imdb = cnnImdb(...
    N, offset, filename, label, label_path, image_path, splitData, flatten)
    % =====================================================================
    % Initialize the imdb structure (image database).
    % Note the fields are arbitrary: only your getBatch needs to understand it.
    % The field imdb.set is used to distinguish between the training and
    % validation sets, and is only used in the above call to cnn_train.
    % Based on http://www.robots.ox.ac.uk/~joao/cnn_toy_data.m

    rng(1);

    log_filename = strcat(label_path, filename);
    camera_filename = strcat(image_path, filename);
    
    label_key = strcat('/', label);
    label_data = h5read(log_filename, label_key);
    
    % =====================================================================
    % Preallocate memory
    images = zeros(160, 320, 1, N, 'single');
    labels = zeros(N, 1);
    set = ones(N, 1);

    % =====================================================================
    % Generate N sample images.
    % Note that here one could load the images from files, and do any kind of
    % necessary pre-processing.
    for i = 1:N
      [start, ~] = get_label_indices(i);
      label = get_label_start(start, label_data);
      labels(i) = label;

      % ===================================================================
      % Read single image from .h5
      % http://stackoverflow.com/q/42137631/3208877
      image_data = h5read(camera_filename, '/X', [1 1 1 offset + i], [320 160 3 1]);
      rotated_image = imrotate(image_data, -90); 
      single_image = im2single(rotated_image);
      if flatten
          images(:,:,:,i) = rgb2gray(single_image);
      else
          images(:,:,:,i) = single_image;
      end

      % ===================================================================
      % Mark last 25% of samples as part of the validation set
      if splitData
          if i > 0.75 * N
            set(i) = 2;
          end
      end
    end

    % =====================================================================
    % Show some example images
    figure(2); montage(images(:,:,:,(N-5):N)); title('Example Images');

    % =====================================================================
    % Store results in the imdb struct
    if flatten
        imdb.images.data = reshape(squeeze(images), [N, 51200]);
    else
        imdb.images.data = images    
    end
    imdb.images.label = labels;
    imdb.images.set = set;

    function [start, stop] = get_label_indices(image_number)
        % =================================================================
        % Find log file indices associated with image
        % Note sampling difference is equal to 100 HZ sampling rate divided by
        % 20 HZ image sampling rate (5 log files per image)
        sampling_difference = 5;
        start = ((image_number - 1) * sampling_difference) + 1;
        stop = start + (sampling_difference - 1);
    end

    function start = get_label_start(n, log_data)
        start = log_data(n);
    end

    function average = get_label_average(start, stop, log_data)
        % =================================================================
        % Get average of log files for start and stop indices
        sum = 0;
        for n = start:stop
            sum = sum + log_data(n);
        end
        average = sum / 5;
    end
end