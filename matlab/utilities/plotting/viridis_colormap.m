function cmap = viridis_colormap()
% VIRIDIS_COLORMAP Viridis colormap approximation
%
%   cmap = VIRIDIS_COLORMAP() returns a 256x3 colormap matrix that
%   approximates the matplotlib viridis colormap. This is a perceptually
%   uniform colormap suitable for sequential data visualization.
%
%   Output:
%       cmap - 256x3 matrix of RGB values in range [0, 1]
%
%   Example:
%       imagesc(data);
%       colormap(viridis_colormap());
%       colorbar;

    n = 256;

    % Key points from viridis colormap (RGB values)
    viridis_data = [
        0.267004, 0.004874, 0.329415;
        0.282623, 0.140926, 0.457517;
        0.253935, 0.265254, 0.529983;
        0.206756, 0.371758, 0.553117;
        0.163625, 0.471133, 0.558148;
        0.127568, 0.566949, 0.550556;
        0.134692, 0.658636, 0.517649;
        0.266941, 0.748751, 0.440573;
        0.477504, 0.821444, 0.318195;
        0.741388, 0.873449, 0.149561;
        0.993248, 0.906157, 0.143936
    ];

    % Interpolate to get n colors
    x_orig = linspace(0, 1, size(viridis_data, 1));
    x_new = linspace(0, 1, n);

    cmap = zeros(n, 3);
    cmap(:, 1) = interp1(x_orig, viridis_data(:, 1), x_new);
    cmap(:, 2) = interp1(x_orig, viridis_data(:, 2), x_new);
    cmap(:, 3) = interp1(x_orig, viridis_data(:, 3), x_new);
end
