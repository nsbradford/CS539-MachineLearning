function net = cnnInit()
    net.meta.inputSize = [160 320 3 1];

    net.layers = { };

    net.layers{end+1} = struct(...
      'name', 'conv1',...
      'type', 'conv',...
      'weights', {xavier(3,3,3,100)},...
      'pad', 0,...
      'stride', 1,...
      'learningRate', [1 1],...
      'weightDecay', [1 0]);

    net.layers{end+1} = struct(...
      'name', 'relu1',...
      'type', 'relu');

    net.layers{end+1} = struct(...
      'name', 'conv2',...
      'type', 'conv',...
      'weights', {xavier(3,3,100,100)},...
      'pad', 0,...
      'stride', 1,...
      'learningRate', [1 1],...
      'weightDecay', [1 0]);

    net.layers{end+1} = struct(...
      'name', 'relu2',...
      'type', 'relu');

    net.layers{end+1} = struct(...
      'name', 'prediction',...
      'type', 'conv',...
      'weights', {xavier(3,3,100,3)},...
      'pad', 0,...
      'stride', 1,...
      'learningRate', [1 1],...
      'weightDecay', [1 0]);

    % Consolidate the network, fixing any missing option
    % in the specification above

    net = vl_simplenn_tidy(net);
    
    function weights = xavier(varargin)
        % XAVIER  Xavier filter initialization.
        % WEIGHTS = XAVIER(H, W, C, N) initializes N filters of support H x
        % W and C channels using Xavier method. WEIGHTS = {FILTERS,BIASES}is
        % a cell array containing both filters and biases.
        %
        % See also:
        % Glorot, Xavier, and Yoshua Bengio.
        % "Understanding the difficulty of training deep feedforward neural networks."
        % International conference on artificial intelligence and statistics. 2010.

        filterSize = [varargin{:}];
        scale = sqrt(2/prod(filterSize(1:3)));
        filters = randn(filterSize, 'single') * scale;
        biases = zeros(filterSize(4), 1, 'single');
        weights = {filters, biases};
    end
end