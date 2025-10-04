% Test script for PSN MATLAB implementation
% Generate test data
rng(42);
nunits = 10;
nconds = 20;
ntrials = 5;
data = randn(nunits, nconds, ntrials) * 2 + randn(nunits, nconds, ntrials) - 0.5;

% Test basic PSN call
results = psn(data, 0, struct(), true);  

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

% Test threshold=0 functionality (cv_thresholds can now include 0)
fprintf('\n=== Testing threshold=0 functionality ===\n');
opt_zero = struct();
opt_zero.cv_mode = 0;
opt_zero.cv_threshold_per = 'population';
opt_zero.cv_thresholds = 0:nunits;  % Include 0 as a baseline
opt_zero.denoisingtype = 0;
results_zero = psn(data, 0, opt_zero, false);

fprintf('Cross-validation with threshold range 0:%d\n', nunits);
fprintf('CV scores shape: [%d x %d x %d]\n', size(results_zero.cv_scores, 1), size(results_zero.cv_scores, 2), size(results_zero.cv_scores, 3));
fprintf('Expected cv_scores rows: %d (includes threshold=0)\n', nunits + 1);
fprintf('Best threshold selected: %d\n', results_zero.best_threshold);

% Test that threshold=0 produces zero denoiser
opt_force_zero = struct();
opt_force_zero.cv_mode = 0;
opt_force_zero.cv_threshold_per = 'population';
opt_force_zero.cv_thresholds = 0;  % Force threshold=0 only
opt_force_zero.denoisingtype = 0;
results_force_zero = psn(data, 0, opt_force_zero, false);

fprintf('\nForced threshold=0 test:\n');
fprintf('Best threshold: %d\n', results_force_zero.best_threshold);
fprintf('Denoiser is all zeros: %d\n', all(results_force_zero.denoiser(:) == 0));
% When threshold=0, denoised data should equal unit means (broadcasted across conditions)
expected_zero_denoised = repmat(results_force_zero.unit_means, 1, nconds);
fprintf('Denoised data equals unit means: %d\n', all(abs(results_force_zero.denoiseddata(:) - expected_zero_denoised(:)) < 1e-10));

% Test unit-wise CV with threshold=0 included
opt_unit_zero = struct();
opt_unit_zero.cv_mode = 0;
opt_unit_zero.cv_threshold_per = 'unit';
opt_unit_zero.cv_thresholds = 0:nunits;
opt_unit_zero.denoisingtype = 0;
results_unit_zero = psn(data, 0, opt_unit_zero, false);

fprintf('\nUnit-wise CV with threshold range 0:%d\n', nunits);
fprintf('Best thresholds per unit: %s\n', mat2str(results_unit_zero.best_threshold));
fprintf('Min threshold selected: %d\n', min(results_unit_zero.best_threshold));
fprintf('Max threshold selected: %d\n', max(results_unit_zero.best_threshold));

% Validate that cv_scores has correct shape
assert(size(results_zero.cv_scores, 1) == nunits + 1, 'cv_scores should have nunits+1 rows');
assert(size(results_unit_zero.cv_scores, 1) == nunits + 1, 'cv_scores should have nunits+1 rows');
assert(results_force_zero.best_threshold == 0, 'Forced threshold should be 0');
assert(all(results_force_zero.denoiser(:) == 0), 'Denoiser should be all zeros when threshold=0');
expected_zero_denoised = repmat(results_force_zero.unit_means, 1, nconds);
assert(all(abs(results_force_zero.denoiseddata(:) - expected_zero_denoised(:)) < 1e-10), 'Denoised data should equal unit means when threshold=0');

fprintf('\n=== All threshold=0 tests passed! ===\n');
fprintf('\nTest completed successfully!\n');
