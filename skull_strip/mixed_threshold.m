function output = mixed_threshold(img, window_size, glob_thresh, eval, mode)
    
    output = zeros(size(img));
    
    % loop through the img
    for r = 1:size(img, 1)
        for c = 1:size(img, 2)
            % get the window around the pixel
            window = get_window(img, r, c, window_size);
            
            % get the local thresholding value
            if mode == 'mean'
                local_thresh = mean(window, 'all');
            else 
                local_thresh = mode(window, 'all');
            end 
             
            % global threshold the image
            if abs(local_thresh - glob_thresh) > eval
                for i = r - window_size:r+window_size
                    for j = c - window_size:c+window_size
                        if i < 1 || j < 1 || i > size(img, 1) || j > size(img, 2)
                            continue;
                        end
                        
                        if img(i,j) < glob_thresh
                            output(i,j) = 0; 
                        else
                            output(i,j) = 1;
                        end
                    end
                end
                continue;
            end
            
            % local threshold the image
            for i = r - window_size:r+window_size
                for j = c - window_size:c+window_size
                    if i < 1 || j < 1 || i > size(img, 1) || j > size(img, 2)
                        continue
                    end
                    
                    if img(i,j) < local_thresh
                        output(i,j) = 0;
                    else
                        output(i,j) = 1;
                    end
                end
            end
            
            
        end
    end
end