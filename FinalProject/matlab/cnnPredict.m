function [res, preds] = cnnPredict(net, imdb, subset)
    res = vl_simplenn(net, imdb.images.data(:,:,:,subset)) ;
    preds = res(end).x ;
end