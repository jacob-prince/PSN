function visualize_results(results, opt)
% VISUALIZE_RESULTS  Create diagnostic figures
%
%   visualize_results(results, opt) generates diagnostic visualizations of
%   PSN denoising results by calling the external visualization.m script.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <results> - struct containing PSN results (output from psn main function)
%
% <opt> - struct with PSN options. Relevant field:
%   .wantverbose - if 1, print messages about visualization status
%
% -------------------------------------------------------------------------
% Behavior:
% -------------------------------------------------------------------------
%
% If visualization.m exists in the MATLAB path, calls it with (input_data, results).
% Otherwise, prints a message (if wantverbose=1) and skips visualization

    if exist('visualization', 'file')
        visualization(results.input_data, results);
    else
        % Basic placeholder visualization
        if opt.wantverbose
            fprintf('  (visualization.m not found - skipping figures)\n');
        end
    end
end
