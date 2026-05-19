function cmap = redblue()
% REDBLUE Red-Blue diverging colormap
%
%   cmap = REDBLUE() returns a 256x3 colormap matrix that transitions
%   from blue through white to red. Useful for visualizing symmetric
%   data centered at zero (e.g., correlation matrices, residuals).
%
%   Output:
%       cmap - 256x3 matrix of RGB values in range [0, 1]
%
%   Example:
%       imagesc(data);
%       colormap(redblue());
%       colorbar;

    n = 256;
    cmap = zeros(n, 3);
    mid = ceil(n/2);

    % Blue to white
    cmap(1:mid, 1) = linspace(0, 1, mid);
    cmap(1:mid, 2) = linspace(0, 1, mid);
    cmap(1:mid, 3) = ones(mid, 1);

    % White to red
    cmap(mid+1:n, 1) = ones(n-mid, 1);
    cmap(mid+1:n, 2) = linspace(1, 0, n-mid);
    cmap(mid+1:n, 3) = linspace(1, 0, n-mid);
end
