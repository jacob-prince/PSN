function f = cmapsign4(n)
% CMAPSIGN4  Cyan-blue-black-red-yellow diverging colormap (symmetric about black).
%
%   f = cmapsign4(n) returns a colormap symmetric around black (zero), useful
%   for signed data: cyan-white -> cyan -> blue -> black -> red -> yellow ->
%   yellow-white.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <n> (optional) - number of colors. Default: 256.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <f> - [n x 3] RGB colormap, values in [0, 1].

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
