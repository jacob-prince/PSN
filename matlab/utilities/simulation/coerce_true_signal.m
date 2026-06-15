function arr = coerce_true_signal(true_signal)
% COERCE_TRUE_SIGNAL  Normalize a user-supplied true_signal into a 2D (ncond, nvox) array.
%
%   arr = coerce_true_signal(true_signal) accepts a 2D numeric matrix, an
%   image-shaped numeric array, or an image filepath/name, and returns a 2D
%   [ncond x nvox] matrix suitable for use as a ground-truth signal. This is
%   what lets natural images stand in as the ground-truth signal so denoising
%   can be eyeballed on a recognizable picture.
%
%   Accepts:
%     - a char/string filepath to an image file (loaded via imread and
%       normalized to [0, 1]); a bare name like 'pliny' resolves to a bundled
%       image in matlab/utilities/simulation/images/ (e.g. images/pliny.jpg)
%     - a numeric array that is already 2D (ncond, nvox), returned as-is
%     - an image-shaped numeric array (H, W) or (H, W, C); spatial dimensions
%       are flattened to 2D
%
%   For image-derived inputs the SMALLER flattened dimension is placed on the
%   units axis (nvox) and the larger on conditions, purely so the O(nvox^3)
%   covariance work stays cheap at runtime. Explicit 2D (ncond, nvox) numeric
%   arrays are left in the caller's orientation.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <true_signal> - char/string image path or name, OR a numeric array that is
%   2D [ncond x nvox], or image-shaped [H x W] / [H x W x C].
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <arr> - [ncond x nvox] double matrix. Image-derived inputs are flattened,
%   normalized to [0,1], and oriented so the smaller dim is nvox.

    from_image = false;

    if ischar(true_signal) || isstring(true_signal)
        true_signal = load_image_as_array(char(true_signal));
        from_image = true;
    end

    arr = double(true_signal);

    if ndims(arr) == 3
        % (H, W, C) -> (H, W*C); flatten width x channels into one axis.
        % Use reshape with explicit ordering that matches numpy's
        % arr.reshape(h, w*c): element (h, w, c) maps to column (w-1)*C + c.
        [h, w, c] = size(arr);
        arr = reshape(permute(arr, [3, 2, 1]), [w * c, h])';
        from_image = true;
    elseif ~ismatrix(arr)
        error(['true_signal must be a filepath, a 2D (ncond, nvox) array, or an ' ...
               'image-shaped 2D/3D array; got an array with %d dims.'], ndims(arr));
    end

    % Smaller dim -> units (nvox), larger -> conditions. Image-derived only.
    if from_image && size(arr, 2) > size(arr, 1)
        arr = arr';
    end
end


function arr = load_image_as_array(pth)
% Load an image file into a double array normalized to [0, 1].
% Returns an [H x W] array for grayscale images or [H x W x C] for color.

    pth = resolve_image_path(pth);
    arr = imread(pth);
    arr = double(arr);
    % Normalize integer pixel ranges (e.g. uint8 0-255) to [0, 1]; leave
    % already-normalized float images untouched.
    if ~isempty(arr) && max(arr(:)) > 1.0
        arr = arr / 255.0;
    end
end


function pth = resolve_image_path(pth)
% Resolve a true_signal image string to an actual file on disk.
% If the string is already a real path, use it as-is. Otherwise treat it as a
% name and look it up in the bundled images/ directory next to this function,
% so e.g. true_signal='pliny' (or 'pliny.jpg') resolves to images/pliny.jpg.

    image_exts = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff', 'webp'};

    if exist(pth, 'file') == 2
        return;
    end

    this_dir = fileparts(mfilename('fullpath'));
    images_dir = fullfile(this_dir, 'images');
    [~, name, ext] = fileparts(pth);
    base = [name, ext];  % basename, possibly with its own extension

    % 1) exact stem with a known image extension: images/<name>.<ext>
    % 2) substring match: images/*<name>*.<ext>
    patterns = {[base, '.*'], ['*', base, '*']};
    for pi = 1:numel(patterns)
        listing = dir(fullfile(images_dir, patterns{pi}));
        names = sort({listing.name});
        for ni = 1:numel(names)
            cand = names{ni};
            [~, ~, cext] = fileparts(cand);
            cext = lower(strrep(cext, '.', ''));
            if any(strcmp(cext, image_exts))
                pth = fullfile(images_dir, cand);
                return;
            end
        end
    end

    error(['true_signal ''%s'' is not an existing path, and no image matching ' ...
           '''*%s*'' was found in %s'], pth, base, images_dir);
end
