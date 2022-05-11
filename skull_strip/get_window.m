% img is original image
% r, c are the coordinates of the center pixel from img
% window size is an integer to specify the box dimensions around r,c

function output = get_window(img, r, c, window_size)
    output = zeros(2 * window_size + 1, 2 * window_size + 1);
    row = 1;
    col = 1; 
    valid_i = 1;
    valid_j= 1; 
    % loop through the window area of the image
    for i = r-window_size:r+window_size
        for j = c-window_size:c+window_size
            if i < 1 || j < 1 || i > size(img, 1) || j > size(img, 2)
                output(row, col) = img(valid_i, valid_j);
            else 
                output(row,col) = img(i,j);
                valid_i = i;
                valid_j = j;
            end
            col = col + 1;
        end
        row = row + 1;
        col = 1;
    end
end