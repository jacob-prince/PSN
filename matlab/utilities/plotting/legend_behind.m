function lgd = legend_behind(varargin)
% LEGEND_BEHIND  legend(...) with a transparent fill so it does not block data.
%
%   lgd = legend_behind(...) takes the same arguments as legend() and returns the
%   Legend handle, but sets its box fill to transparent. MATLAB legends always
%   composite ABOVE the axes, so they cannot be placed behind data in the z-order
%   (the matplotlib approach); a transparent fill is the closest equivalent - the
%   plotted curves/markers show THROUGH the legend instead of being hidden by an
%   opaque box. The box outline is kept so the legend stays delineated.
%
%   This is the default for every legend in the diagnostic figure.

    lgd = legend(varargin{:});
    if ~isempty(lgd) && isgraphics(lgd)
        lgd.Color = 'none';
    end
end
