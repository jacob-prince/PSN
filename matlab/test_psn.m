% Test script for PSN MATLAB implementation
% Generate test data
rng(42);
nunits = 10;
nconds = 20;
ntrials = 5;
data = randn(nunits, nconds, ntrials);

% Test basic PSN call
results = psn(data, 0, struct(), false);  % Don't show figure for now

% Check that we get the expected outputs
fprintf('Test Results:\n');
fprintf('Denoiser shape: [%d x %d]\n', size(results.denoiser, 1), size(results.denoiser, 2));
fprintf('Denoised data shape: [%d x %d]\n', size(results.denoiseddata, 1), size(results.denoiseddata, 2));
fprintf('Full basis shape: [%d x %d]\n', size(results.fullbasis, 1), size(results.fullbasis, 2));
fprintf('Best threshold: %s\n', mat2str(results.best_threshold));
fprintf('Unit means shape: [%d x %d]\n', size(results.unit_means, 1), size(results.unit_means, 2));

% Test magnitude thresholding
opt_mag = struct();
opt_mag.cv_mode = -1;
opt_mag.mag_frac = 0.9;
results_mag = psn(data, 0, opt_mag, false);

fprintf('\nMagnitude thresholding results:\n');
fprintf('Dimensions retained: %d\n', results_mag.dimsretained);
fprintf('Best threshold indices: %s\n', mat2str(results_mag.best_threshold));

fprintf('\nTest completed successfully!\n');
