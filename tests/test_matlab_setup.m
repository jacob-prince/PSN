% Test script to verify GSN dependency setup for PSN
fprintf('Testing GSN dependency setup...\n');

% Add GSN path (same logic as in gsndenoise.m)
gsn_path = fullfile(fileparts(mfilename('fullpath')), 'external', 'gsn', 'matlab');
fprintf('Looking for GSN at: %s\n', gsn_path);

if exist(gsn_path, 'dir')
    fprintf('GSN directory found, adding to path...\n');
    addpath(gsn_path);
    
    % Test if performgsn is now available
    if exist('performgsn', 'file') == 2
        fprintf('SUCCESS: performgsn function found!\n');
        
        % Try to get help for the function to verify it loads
        try
            help_text = help('performgsn');
            fprintf('Function help available (first 100 chars):\n');
            fprintf('%.100s...\n', help_text);
        catch ME
            fprintf('WARNING: Could not get help for performgsn: %s\n', ME.message);
        end
    else
        fprintf('ERROR: performgsn function not found after adding path\n');
        fprintf('Contents of GSN matlab directory:\n');
        dir(gsn_path)
    end
else
    fprintf('ERROR: GSN directory not found at expected location\n');
    fprintf('Make sure you have run: git submodule update --init --recursive\n');
end

fprintf('Test complete.\n');
