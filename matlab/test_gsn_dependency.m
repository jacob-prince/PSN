% Test script to verify GSN dependency is properly set up

% Setup GSN dependency path (same as in psn.m)
gsn_matlab_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'external', 'gsn', 'matlab');
fprintf('Looking for GSN at: %s\n', gsn_matlab_path);

if exist(gsn_matlab_path, 'dir')
    addpath(gsn_matlab_path);
    fprintf('✓ GSN path added successfully\n');
    
    % Test if performgsn is available
    if exist('performgsn', 'file')
        fprintf('✓ performgsn function found\n');
        
        % Get function info
        help_text = help('performgsn');
        first_line = strsplit(help_text, '\n');
        fprintf('performgsn description: %s\n', first_line{1});
    else
        fprintf('✗ performgsn function not found\n');
    end
else
    fprintf('✗ GSN directory not found at: %s\n', gsn_matlab_path);
    fprintf('Please run: git submodule update --init --recursive\n');
end
