% SC4061 Lab 1

% NOTE:
%
% - I have commented out all the `saveas` lines as it was only
%       used at the start to export the figures used for the report.
% - If you are running this file and would like to save the figures,
%       create a `results/` directory in the root directory to save.


% EXPERIMENT 2.1 - Contrast Stretching

% 2.1 (a) - Input image

Pc = imread("mrt-train.jpg"); 
whos Pc;

% Convert image from RGB to Greyscale
P = rgb2gray(Pc);


% 2.1 (b) - View image
figure;
imshow(P);
title('Original mrt-train Image');


% 2.1 (c) - Check original max and min attribute
min_original = min(P(:))
max_originial = max(P(:))

fprintf('Minimum intensity: %d\n', min_original);
fprintf('Maximum intensity: %d\n', max_originial);


% 2.1 (d) - Stretch contrast
P2 = imsubtract(P, double(min_original));                
P2 = immultiply(P2, 255 / double(max_originial - min_original));  
min_stretched = min(P2(:));
max_stretched = max(P2(:));

fprintf('New minimum intensity: %d\n', min_stretched);
fprintf('New maximum intensity: %d\n', max_stretched);

% 2.1 (e) - Display the stretched image
figure;
imshow((P2));
title('Stretched mrt-train Image');

%% 


% EXPERIMENT 2.2 - Histogram Equalization

% 2.2 (a) - Display image intensity histogram

% 10 Bins
figure;
imhist(P, 10);
title("Histogram (10 Bins)");
% saveas(gcf, 'p_histogram_10bins.png');

% 256 Bins
figure;
imhist(P, 256);
title('Histogram (256 Bins)');
% saveas(gcf, 'p_histogram_256bins.png');


% 2.2 (b) - Histogram Equalization
P3 = histeq(P,255);

% 10 Bins
figure;
imhist(P3,10);
title('P3 Histogram (10 Bins)');
% saveas(gcf, 'p3_histogram_10bins.png');

% 256 Bins
figure;
imhist(P3,256);
title('P3 Histogram (256 Bins)');
% saveas(gcf, 'p3_histogram_256bins.png');


% 2.2 (b) - Re-apply Histogram Equalization on P3 
P4 = histeq(P3,255);

% 10 Bins
figure
imhist(P4,10);
title('P4 Histogram (10 Bins)');
% saveas(gcf, 'p4_histogram_10bins.png');

% 256 Bins
figure
imhist(P4,256);
title('P4 Histogram (256 Bins)');
% saveas(gcf, 'p4_histogram_256bins.png');

%% 


% EXPERIMENT 2.3 - Linear Spatial Filtering

% 2.3 (a[i]) - Y and X-dimensions are 5 and σ = 1.0
h1 = fspecial('gaussian', 5, 1);

% 2.3 (a[ii]) - Y and X-dimensions are 5 and σ = 2.0
h2 = fspecial('gaussian', 5, 2);

% Normalize filters
h1 = h1 / sum(h1, 'all');
h2 = h2 / sum(h2, 'all');

% Check if h1 and h2 are normalized
fprintf('Sum of h1 elements: %f\n', sum(h1, 'all'));
fprintf('Sum of h2 elements: %f\n', sum(h2, 'all'));

% 3d figure for h1
figure;
mesh(h1);
% saveas(gcf, 'results/gaussian_filter_1.png');

% 3d figure for h2
figure;
mesh(h2);
% saveas(gcf, 'results/gaussian_filter_2.png');

% 2.3 (b) - Download `lib-gn.jpg` image
Pgn = imread('lib-gn.jpg');
figure
imshow(Pgn);
% saveas(gcf, 'results/lib_gn.png');


% 2.3 (c) - Linear filtering
Pgn1 = uint8(conv2(Pgn,h1));
figure
imshow(Pgn1);
% saveas(gcf, 'results/pgn1.png');


Pgn2 = uint8(conv2(Pgn,h2));
figure
imshow(Pgn2);
% saveas(gcf, 'results/pgn2.png');

% 2.3 (d) - Download `lib-sp.jpg` image
Psp = imread('lib-sp.jpg');
figure
imshow(Psp);
% saveas(gcf, 'results/lib-sp.png');

% 2.3 (e) - Repeat step (c) above
Psp1 = uint8(conv2(Psp,h1));
figure
imshow(Psp1);
% saveas(gcf, 'results/psp1.jpg');


Psp2 = uint8(conv2(Psp,h2));
figure
imshow(Psp2);
% saveas(gcf, 'results/psp2.jpg');


%% 

% EXPERIMENT 2.4 - Median Filtering


% Pgn - Median filter with 3x3 neighborhood
Pgn_med3 = medfilt2(Pgn, [3 3]);
figure;
imshow(Pgn_med3);
% saveas(gcf, 'results/gaussian_median_3x3.jpg');

% Pgn - Median filter with 5x5 neighborhood
Pgn_med5 = medfilt2(Pgn, [5 5]);
figure;
imshow(Pgn_med5);
% saveas(gcf, 'results/gaussian_median_5x5.jpg');


% Psp - Median filter with 3x3 neighborhood
Psp_med3 = medfilt2(Psp, [3 3]);
figure;
imshow(Psp_med3);
% saveas(gcf, 'results/speckle_median_3x3.jpg');

% Psp - Median filter with 5x5 neighborhood
Psp_med5 = medfilt2(Psp, [5 5]);
figure;
imshow(Psp_med5);
% saveas(gcf, 'results/speckle_median_5x5.jpg');


%% 

% EXPERIMENT 2.5 - Suppressing Noise Interference Patterns

% 2.5 (a) -  Download `pck-int.jpg` image
Ppck = imread('pck-int.jpg');
figure
imshow(Ppck);

% 2.5 (b) -  Obtain the Fourier transform F of the image
F = fft2(Ppck);
S = abs(F).^2;

figure
imagesc(fftshift(S.^0.1));
colormap('default')
% saveas(gcf, 'results/power_spectrum.png');


% 2.5 (c) -  Redisplay the power spectrum without fftshift
figure
imagesc(S.^0.1);
[x, y] = ginput(2)
% saveas(gcf, 'results/power_spectrum_2.png');

x1 = round(x(1));
y1 = round(y(1));
x2 = round(x(2));
y2 = round(y(2));

fprintf('Peak 1: (%d, %d)\n', x1, y1);
fprintf('Peak 2: (%d, %d)\n', x2, y2);


% 2.5 (d) - Set to zero the 5x5 neighbourhood elements at locations 
% corresponding to the above peaks

F(y1-2:y1+2, x1-2:x1+2) = 0;
F(y2-2:y2+2, x2-2:x2+2) = 0;

S = abs(F).^2;
figure
imagesc(fftshift(S.^0.1));
colormap('default');
% saveas(gcf, 'results/power_spectrum_adjusted.png');


% 2.5 (e) - Compute the inverse Fourier transform using ifft2

F_inv = real(ifft2(F));
F_inv = uint8(F_inv);

figure;
imshow(F_inv);
% saveas(gcf, 'results/f_inverse_1.jpg');


% Zero out entire rows and columns at peak locations
F(y1, :) = 0;
F(:, x1) = 0;
F(y2, :) = 0;
F(:, x2) = 0;

FI = uint8(real(ifft2(F)));

figure;
imshow(FI);
% saveas(gcf, 'results/f_inverse_2.png');

% 2.5 (f) - "Free" the primate by filtering out the fence

Pcage = imread('primate-caged.jpg');
Pcage_gray = rgb2gray(Pcage);
figure
imshow(Pcage_gray);

% Compute fourier transform
F_cage = fft2(double(Pcage_gray));

% Compute power spectrum
S_cage = abs(F_cage).^2;

% Display power spectrum with fftshift
figure;
imagesc(fftshift(S_cage.^0.1));
colormap('default');

% Display power spectrum without fftshift
figure;
imagesc(S_cage.^0.1);
colormap('default');

% Select intensity threshold
[col, row] = ginput(1);
intensity_threshold = S_cage(round(row), round(col));

fprintf('Selected intensity threshold: %.2f\n', intensity_threshold);
% saveas(gcf, 'results/primate_power_spectrum_click.png');

% Define rectangular filtering region
[rows, cols] = size(S_cage);

x1 = 2;
y1 = 2;
x2 = cols - 1;
y2 = rows - 1;

fprintf('Filtering region: rows %d to %d, cols %d to %d\n', y1, y2, x1, x2);

% Filter out high-intensity frequencies based on my earlier selection
for y = y1:y2
    for x = x1:x2
        if S_cage(y, x) > intensity_threshold
            F_cage(y, x) = 0;
        end
    end
end

% Visualize the modified power spectrum
S_cage_modified = abs(F_cage).^2;

figure;
imagesc(fftshift(S_cage_modified.^0.1));
colormap('default');
% saveas(gcf, 'results/primate_power_spectrum_modified.png');

% Inverse Fourier Transform
Pcage_freed = uint8(real(ifft2(F_cage)));

figure;
imshow(Pcage_freed);
% saveas(gcf, 'results/primate_freed.jpg');

%%

% EXPERIMENT 2.6 - Undoing Perspective Distortion of Planar Surface

% 2.6 (a) - Download `book.jpg’ and display the image

Pbook = imread('book.jpg');
figure;
imshow(Pbook);


% 2.6 (b) - Find out the location of 4 corners of the book

% NOTE: After getting the 4 corners with ginput, I have commented out
%   and replaced with the actual values I got from the initial run.

% [X, Y] = ginput(4);
X = [4.71; 144.26; 306.67; 254.94];
Y = [161.40; 29.07; 48.32; 219.15];

% Extract coords
fprintf('Corner 1: (%.2f, %.2f)\n', X(1), Y(1));
fprintf('Corner 2: (%.2f, %.2f)\n', X(2), Y(2));
fprintf('Corner 3: (%.2f, %.2f)\n', X(3), Y(3));
fprintf('Corner 4: (%.2f, %.2f)\n', X(4), Y(4));

% Coords of desired image
desired_X = [0 0 210 210]; 
desired_Y = [297 0 0 297];



% 2.6 (c) - Set up the matrices required to estimate the projective transformation

A = [
 [X(1),Y(1),1,0,0,0, -desired_X(1)*X(1),-desired_X(1)*Y(1)];
 [0,0,0,X(1),Y(1),1, -desired_Y(1)*X(1),-desired_Y(1)*Y(1)];
 [X(2),Y(2),1,0,0,0, -desired_X(2)*X(2),-desired_X(2)*Y(2)];
 [0,0,0,X(2),Y(2),1, -desired_Y(2)*X(2),-desired_Y(2)*Y(2)];
 [X(3),Y(3),1,0,0,0, -desired_X(3)*X(3),-desired_X(3)*Y(3)];
 [0,0,0,X(3),Y(3),1, -desired_Y(3)*X(3),-desired_Y(3)*Y(3)];
 [X(4),Y(4),1,0,0,0, -desired_X(4)*X(4),-desired_X(4)*Y(4)];
 [0,0,0,X(4),Y(4),1, -desired_Y(4)*X(4),-desired_Y(4)*Y(4)];
];

v = [desired_X(1); desired_Y(1); desired_X(2); desired_Y(2); desired_X(3); desired_Y(3); desired_X(4); desired_Y(4)];
u = A \ v;
U = reshape([u;1], 3, 3)';

w = U*[X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:))


% 2.6 (d) - Warp the Image

T = maketform('projective', U');
Pwarped = imtransform(Pbook, T, 'XData', [0 210], 'YData', [0 297]);


% 2.6 (e) - Display the image

figure
imshow(Pwarped);
% saveas(gcf, 'results/book_warped.jpg');


% 2.6 (f) - Identify the big rectangular pink area

figure;
imshow(Pwarped);

com_screen = imcrop;

% Cropped 
figure;
imshow(com_screen);
% saveas(gcf, 'results/com_screen.jpg');


%%


% EXPERIMENT 2.7 - Code Two Perceptrons


% Algorithm 1

x1 = [3; 3; 1];
x2 = [1; 1; 1];

% training params

w = [0; 0; 0];
alpha = 1;
k = 1;
correct_count = 0;

while true
    % alternating between our only 2 samples
    if mod(k,2) == 1
        x = x1
        result = w' * x
        if result > 0
            correct_count = correct_count + 1

        else
            w = w + alpha * x
            correct_count = 0
        end

    else
        x = x2;
        result = w' * x;
        if result < 0
            correct_count = correct_count + 1;

        else
            w = w - alpha * x;
            correct_count = 0;
        end
        
    end
    
    % Check breaking condition
    if correct_count == 2
        break;
    end
    
    k = k + 1;
end

fprintf('Total iterations: %d\n', k);
fprintf('Final weights: w = [%.1f %.1f %.1f]\n', w(1), w(2), w(3));

% Plot results for algo 1
figure;
hold on;
grid on;

% Decision boundary
x_vals = 0:0.1:4;
y_vals = -(w(1) * x_vals + w(3)) / w(2);
plot(x_vals, y_vals, 'k-', 'LineWidth', 2);

% Training points
plot(x1(1), x1(2), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(x2(1), x2(2), 'rd', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Format plot
xlabel('x_1');
ylabel('x_2');
legend('Decision Boundary', 'x_1 (Class c_1)', 'x_2 (Class c_2)');
axis([0 4 0 4]);

hold off;


%%


% Algorithm 2

x1 = [3; 3; 1];
x2 = [1; 1; 1];

w = [0; 0; 0];
alpha = 0.2;
k=1;
max_iterations = 100;
tolerance = 0.01;

% Normalize the vectors

x1_normalized = x1 / norm(x1);
x2_normalized = x2 / norm(x2);

cost_history = zeros(1, max_iterations); 


while k <= max_iterations
    if mod(k, 2) == 1
        x = x1_normalized;
        r = 1;
    else
        x = x2_normalized;
        r = -1;
    end
    
    prediction = dot(w, x);
    error = r - prediction;
    cost = 0.5 * error^2;
    
    % Store cost for plotting later below
    cost_history(k) = cost;
    
    % Update weights
    w = w + alpha * error * x;
    
    k = k + 1;
end


fprintf('Total iterations: %d\n', max_iterations);
fprintf('Final cost: %.6f\n', cost_history(end));
fprintf('Final weights: w = [%.4f %.4f %.4f]\n', w(1), w(2), w(3));


% Plot Cost funciton
figure;
plot(1:max_iterations, cost_history, 'b-', 'LineWidth', 2);
xlabel('Iterations');
ylabel('Cost');
title('Algorithm 2: Cost vs Iterations');
grid on;


% Plot results for algo 2
figure;
hold on;
grid on;

% Decision boundary
x_vals = 0:0.1:4;
y_vals = -(w(1) * x_vals + w(3)) / w(2);
plot(x_vals, y_vals, 'k-', 'LineWidth', 2);

% Training points
plot(x1(1), x1(2), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(x2(1), x2(2), 'rd', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Format plot
xlabel('x_1');
ylabel('x_2');
title('Algorithm 2 - Decision Boundary');
legend('Decision Boundary', 'x_1 (Class c_1)', 'x_2 (Class c_2)');
axis([0 4 0 4]);

hold off;      