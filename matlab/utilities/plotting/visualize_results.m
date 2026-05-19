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
% <opt> - struct with PSN options. Relevant fields:
%   .wantverbose - if 1, print messages about visualization status
%   .figurepath  - if non-empty, save figure to this path and close it
%
% -------------------------------------------------------------------------
% Behavior:
% -------------------------------------------------------------------------
%
% If visualization.m exists in the MATLAB path, calls it with (input_data, results).
% Otherwise, prints a message (if wantverbose=1) and skips visualization.
% If figurepath is specified, saves the figure and closes it.

    if exist('visualization', 'file')
        % Check if we should save (and not display)
        if isfield(opt, 'figurepath') && ~isempty(opt.figurepath)
            % Create figure invisible, save, and close
            fig = visualization(results.input_data, results, 'Visible', 'off');

            % Ensure directory exists
            [filepath_dir, ~, ~] = fileparts(opt.figurepath);
            if ~isempty(filepath_dir) && ~exist(filepath_dir, 'dir')
                mkdir(filepath_dir);
            end

            % Save with high resolution (check if figure is still valid)
            if isvalid(fig)
                saveas(fig, opt.figurepath);
            end

            if opt.wantverbose
                fprintf('PSN: Diagnostic figure saved to: %s\n', opt.figurepath);
            end

            % Close figure after saving (check if still valid)
            if isvalid(fig)
                close(fig);
            end
        else
            % Display figure normally
            fig = visualization(results.input_data, results);
        end
    else
        % Basic placeholder visualization
        if opt.wantverbose
            fprintf('  (visualization.m not found - skipping figures)\n');
        end
    end
end
