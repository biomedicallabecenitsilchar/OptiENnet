
% Read an RGB image
rgbImage = imread('image-square (35).jpg');

% Apply CLAHE to each color channel separately
claheImage = rgbImage;
for i = 1:3 % Loop through each color channel
    channel = rgbImage(:,:,i);
    claheImage(:,:,i) = adapthisteq(channel, 'NumTiles', [8, 8], 'ClipLimit', 0.0019);
end

% Calling func_denoise_wp2d
[XCMP,wptCMP] = func_denoise_wp2d(claheImage)

% Display the original, CLAHE-enhancedand and denoised images side by side
figure;
subplot(1, 3, 1);
imshow(rgbImage);
title('Original RGB Image');

subplot(1, 3, 2);
imshow(claheImage);
title('CLAHE-Enhanced RGB Image');

subplot(1, 3, 3);
imshow(XCMP);
title('Denoised RGB Image');

