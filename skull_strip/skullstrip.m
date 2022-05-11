function [result, result_mask] = skullstrip(img_slice, window_size, global_threshold, eps)
    % mixed thresholding
    img_thresh = mixed_threshold(img_slice, window_size, global_threshold, eps, 'mean');

    % morphological operations
    structdisk = strel('disk', 2);
    img_erode = imerode(img_thresh, structdisk);

    structsq = strel('square', 2);
    img_close = imclose(img_erode, structsq);

    % get rid of largest component
    CC = bwconncomp(img_close);
    
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    
    result_mask = zeros(size(img_close));
    if isempty(CC.PixelIdxList)
        result = result_mask;
        return;
    end
    
    result_mask(CC.PixelIdxList{idx}) = 1;
    result = img_slice; 
    result(~result_mask) = 0; 
    % montage({img_slice, img_thresh, img_erode, img_close, result_mask, result}, 'Size', [1 6])

end