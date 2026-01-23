function f = cmapsign4(n)
% CMAPSIGN4 Return a cyan-blue-black-red-yellow colormap.
%
% f = cmapsign4(n) returns an n x 3 colormap array.
%
% This colormap is symmetric around black (center), going from
% cyan-white through cyan and blue to black, then from black
% through red and yellow to yellow-white.
%
% This is useful for visualizing data that has both positive and
% negative values, with zero mapped to black.

if nargin < 1
    n = 256;  % default number of colors
end

colors = [
    0.8, 1, 1    % cyan-white
    0, 1, 1      % cyan
    0, 0, 1      % blue
    0, 0, 0      % black (center)
    1, 0, 0      % red
    1, 1, 0      % yellow
    1, 1, 0.8    % yellow-white
];

f = zeros(n, 3);
for p = 1:size(colors, 2)
    f(:, p) = interp1(linspace(0, 1, size(colors, 1)), colors(:, p)', linspace(0, 1, n), 'linear');
end

end
