% TEST_PSN_ALL_COMBINATIONS Comprehensive test of all PSN option combinations
%
% This script systematically tests every combination of PSN input options
% to validate that there are no bugs and all combinations work correctly.

function test_psn_all_combinations()
    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('TESTING ALL PSN OPTION COMBINATIONS\n');
    fprintf('%s\n\n', repmat('=', 1, 80));

    %% PHASE 1: Test invalid inputs (should fail)
    fprintf('\n%s\n', repmat('-', 1, 80));
    fprintf('PHASE 1: Testing invalid inputs (expecting failures)\n');
    fprintf('%s\n\n', repmat('-', 1, 80));

    test_invalid_inputs();

    %% PHASE 2: Test valid combinations across different dataset shapes
    fprintf('\n%s\n', repmat('-', 1, 80));
    fprintf('PHASE 2: Testing valid combinations on different dataset shapes\n');
    fprintf('%s\n\n', repmat('-', 1, 80));

    test_different_shapes();

    %% PHASE 3: Test all valid option combinations on standard dataset
    fprintf('\n%s\n', repmat('-', 1, 80));
    fprintf('PHASE 3: Testing all valid option combinations\n');
    fprintf('%s\n\n', repmat('-', 1, 80));

    % Generate test data
    fprintf('Generating test data...\n');
    [train_data, test_data, ground_truth] = simulate('generate_data', ...
        'nvox', 30, ...
        'ncond', 50, ...
        'ntrial', 3, ...
        'signal_decay', 2.0, ...
        'noise_decay', 1.25, ...
        'random_seed', 42, ...
        'verbose', false, ...
        'want_fig', false);

    fprintf('  Data shape: [%d, %d, %d] (nvox, ncond, ntrial)\n\n', size(train_data));

    % Define all option combinations to test
    test_combinations = generate_test_combinations();

    % Track results
    n_total = length(test_combinations);
    n_passed = 0;
    n_failed = 0;
    failed_tests = {};

    % Run all tests
    fprintf('Running %d test combinations...\n\n', n_total);

    for i = 1:n_total
        test_case = test_combinations{i};

        try
            % Run PSN with this combination
            results = psn(train_data, test_case.opts);

            % Validate outputs
            validate_outputs(results, test_case.opts, train_data);

            n_passed = n_passed + 1;
            fprintf('[%3d/%3d] ✓ PASS: %s\n', i, n_total, test_case.description);

        catch ME
            n_failed = n_failed + 1;
            failed_tests{end+1} = struct('case', test_case, 'error', ME);
            fprintf('[%3d/%3d] ✗ FAIL: %s\n', i, n_total, test_case.description);
            fprintf('         Error: %s\n', ME.message);
        end
    end

    %% PHASE 4: Test NaN handling
    fprintf('\n%s\n', repmat('-', 1, 80));
    fprintf('PHASE 4: Testing NaN handling (uneven trials)\n');
    fprintf('%s\n\n', repmat('-', 1, 80));

    test_nan_handling();

    % Summary
    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('OVERALL TEST SUMMARY\n');
    fprintf('%s\n', repmat('=', 1, 80));
    fprintf('Total tests:  %d\n', n_total);
    fprintf('Passed:       %d (%.1f%%)\n', n_passed, 100*n_passed/n_total);
    fprintf('Failed:       %d (%.1f%%)\n', n_failed, 100*n_failed/n_total);
    fprintf('%s\n\n', repmat('=', 1, 80));

    % Report failed tests in detail
    if n_failed > 0
        fprintf('FAILED TESTS DETAIL:\n');
        fprintf('%s\n', repmat('-', 1, 80));
        for i = 1:length(failed_tests)
            ft = failed_tests{i};
            fprintf('\nTest %d: %s\n', i, ft.case.description);
            fprintf('Options: %s\n', struct2str(ft.case.opts));
            fprintf('Error: %s\n', ft.error.message);
            fprintf('Stack:\n');
            for j = 1:min(3, length(ft.error.stack))
                fprintf('  %s (line %d)\n', ft.error.stack(j).name, ft.error.stack(j).line);
            end
        end
        fprintf('%s\n\n', repmat('=', 1, 80));
        error('Some tests failed! See details above.');
    else
        fprintf('✓ ALL TESTS PASSED!\n\n');
    end
end


function test_combinations = generate_test_combinations()
% GENERATE_TEST_COMBINATIONS Create all combinations of PSN options to test

    test_combinations = {};

    % Define option values to test
    basis_values = {'pca', 'signal', 'noise', 'difference', 'random',};
    threshold_methods = {'global', 'hybrid', 'unit'};
    criteria = {'prediction', 'variance'};

    % Test all combinations of basis, threshold_method, and criterion
    for b = 1:length(basis_values)
        for t = 1:length(threshold_methods)
            for c = 1:length(criteria)
                basis = basis_values{b};
                method = threshold_methods{t};
                criterion = criteria{c};

                % Skip invalid combinations
                if strcmp(criterion, 'variance_eigenvalues') && ~strcmp(method, 'global')
                    continue;  % variance_eigenvalues only works with global
                end

                opts = struct();
                opts.basis = basis;
                opts.threshold_method = method;
                opts.criterion = criterion;
                opts.wantfig = false;
                opts.wantverbose = false;

                % Set criterion-specific parameters
                if strcmp(criterion, 'variance')
                    opts.variance_threshold = 0.95;
                end

                desc = sprintf('basis=%s, method=%s, criterion=%s', basis, method, criterion);
                test_combinations{end+1} = struct('opts', opts, 'description', desc);
            end
        end
    end

    % Test variance_eigenvalues criterion (only with global method and compatible basis types)
    % Note: variance_eigenvalues is not compatible with 'random' or custom matrix basis
    for b = 1:length(basis_values)
        basis = basis_values{b};

        % Skip 'random' (not compatible with variance_eigenvalues)
        if strcmp(basis, 'random')
            continue;
        end

        opts = struct();
        opts.basis = basis;
        opts.threshold_method = 'global';
        opts.criterion = 'variance_eigenvalues';
        opts.variance_threshold = 0.95;
        opts.wantfig = false;
        opts.wantverbose = false;

        desc = sprintf('basis=%s, method=global, criterion=variance_eigenvalues', basis);
        test_combinations{end+1} = struct('opts', opts, 'description', desc);
    end

    % Test with custom basis matrix
    fprintf('  Adding custom basis matrix tests...\n');
    custom_basis = randn(30, 30);
    [custom_basis, ~] = qr(custom_basis, 0);  % Orthonormalize

    for t = 1:length(threshold_methods)
        for c = 1:length(criteria)
            method = threshold_methods{t};
            criterion = criteria{c};

            if strcmp(criterion, 'variance_eigenvalues')
                continue;  % Skip for custom basis
            end

            opts = struct();
            opts.basis = custom_basis;
            opts.threshold_method = method;
            opts.criterion = criterion;
            opts.wantfig = false;
            opts.wantverbose = false;

            if strcmp(criterion, 'variance')
                opts.variance_threshold = 0.95;
            end

            desc = sprintf('basis=custom, method=%s, criterion=%s', method, criterion);
            test_combinations{end+1} = struct('opts', opts, 'description', desc);
        end
    end

    % Test different variance thresholds
    fprintf('  Adding variance threshold tests...\n');
    variance_thresholds = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0];
    for vt = variance_thresholds
        opts = struct();
        opts.basis = 'signal';
        opts.threshold_method = 'global';
        opts.criterion = 'variance';
        opts.variance_threshold = vt;
        opts.wantfig = false;
        opts.wantverbose = false;

        desc = sprintf('variance_threshold=%.2f', vt);
        test_combinations{end+1} = struct('opts', opts, 'description', desc);
    end

    % Test allowable_thresholds
    fprintf('  Adding allowable_thresholds tests...\n');
    allowable_sets = {[5, 10, 15], [1, 2, 3, 4, 5], [10, 20, 30]};
    for a = 1:length(allowable_sets)
        opts = struct();
        opts.basis = 'signal';
        opts.threshold_method = 'global';
        opts.criterion = 'prediction';
        opts.allowable_thresholds = allowable_sets{a};
        opts.wantfig = false;
        opts.wantverbose = false;

        desc = sprintf('allowable_thresholds=[%s]', num2str(allowable_sets{a}));
        test_combinations{end+1} = struct('opts', opts, 'description', desc);
    end

    % Test unit_groups with unit and hybrid methods
    fprintf('  Adding unit_groups tests...\n');
    unit_groups = [ones(10,1); 2*ones(10,1); 3*ones(10,1)];
    for method = {'hybrid', 'unit'}
        for criterion = {'prediction', 'variance'}
            opts = struct();
            opts.basis = 'signal';
            opts.threshold_method = method{1};
            opts.criterion = criterion{1};
            opts.unit_groups = unit_groups;
            opts.wantfig = false;
            opts.wantverbose = false;

            if strcmp(criterion{1}, 'variance')
                opts.variance_threshold = 0.95;
            end

            desc = sprintf('unit_groups (3 groups), method=%s, criterion=%s', method{1}, criterion{1});
            test_combinations{end+1} = struct('opts', opts, 'description', desc);
        end
    end

    % Test edge cases
    fprintf('  Adding edge case tests...\n');

    % Zero variance threshold (should keep 0 dimensions)
    opts = struct();
    opts.basis = 'signal';
    opts.threshold_method = 'global';
    opts.criterion = 'variance';
    opts.variance_threshold = 0.0;
    opts.wantfig = false;
    opts.wantverbose = false;
    desc = 'variance_threshold=0 (edge case)';
    test_combinations{end+1} = struct('opts', opts, 'description', desc);

    % Allowable thresholds with only [0]
    opts = struct();
    opts.basis = 'signal';
    opts.threshold_method = 'global';
    opts.criterion = 'prediction';
    opts.allowable_thresholds = [0];
    opts.wantfig = false;
    opts.wantverbose = false;
    desc = 'allowable_thresholds=[0] (edge case)';
    test_combinations{end+1} = struct('opts', opts, 'description', desc);

    % Single unit group (all units in one group)
    opts = struct();
    opts.basis = 'signal';
    opts.threshold_method = 'unit';
    opts.criterion = 'prediction';
    opts.unit_groups = ones(30, 1);
    opts.wantfig = false;
    opts.wantverbose = false;
    desc = 'unit_groups (all units in 1 group)';
    test_combinations{end+1} = struct('opts', opts, 'description', desc);

    fprintf('  Generated %d test combinations\n', length(test_combinations));
end


function validate_outputs(results, opts, train_data)
% VALIDATE_OUTPUTS Check that PSN outputs are valid

    [nunits, nconds, ntrials] = size(train_data);

    % Check required fields exist
    required_fields = {'denoiseddata', 'denoiser', 'best_threshold', 'unit_means'};
    for i = 1:length(required_fields)
        if ~isfield(results, required_fields{i})
            error('Missing required field: %s', required_fields{i});
        end
    end

    % Check denoiseddata shape
    if ~isequal(size(results.denoiseddata), [nunits, nconds])
        error('denoiseddata has wrong shape: expected [%d, %d], got [%s]', ...
              nunits, nconds, num2str(size(results.denoiseddata)));
    end

    % Check denoiser shape
    if ~isequal(size(results.denoiser), [nunits, nunits])
        error('denoiser has wrong shape: expected [%d, %d], got [%s]', ...
              nunits, nunits, num2str(size(results.denoiser)));
    end

    % Check unit_means shape
    if ~isequal(size(results.unit_means), [nunits, 1])
        error('unit_means has wrong shape: expected [%d, 1], got [%s]', ...
              nunits, num2str(size(results.unit_means)));
    end

    % Check best_threshold validity
    if strcmp(opts.threshold_method, 'global')
        if ~isscalar(results.best_threshold)
            error('best_threshold should be scalar for global method');
        end
        if results.best_threshold < 0 || results.best_threshold > nunits
            error('best_threshold out of valid range: %d', results.best_threshold);
        end
    else
        if ~isequal(size(results.best_threshold), [nunits, 1])
            error('best_threshold should be [%d, 1] for %s method', ...
                  nunits, opts.threshold_method);
        end
        if any(results.best_threshold < 0) || any(results.best_threshold > nunits)
            error('Some best_threshold values out of valid range');
        end
    end

    % Check that denoised data doesn't have NaN or Inf
    if any(isnan(results.denoiseddata(:))) || any(isinf(results.denoiseddata(:)))
        error('denoiseddata contains NaN or Inf values');
    end

    % Check that denoiser doesn't have NaN or Inf
    if any(isnan(results.denoiser(:))) || any(isinf(results.denoiser(:)))
        error('denoiser contains NaN or Inf values');
    end

    % Check variance diagnostics if present
    if isfield(results, 'svnv_before') && isfield(results, 'svnv_after')
        if ~isequal(size(results.svnv_before), [nunits, 2])
            error('svnv_before has wrong shape');
        end
        if ~isequal(size(results.svnv_after), [nunits, 2])
            error('svnv_after has wrong shape');
        end

        % Signal variance should not increase after denoising
        if any(results.svnv_after(:, 1) > results.svnv_before(:, 1) + 1e-6)
            warning('Signal variance increased for some units after denoising');
        end

        % Noise variance should decrease after denoising
        if any(results.svnv_after(:, 2) > results.svnv_before(:, 2) + 1e-6)
            warning('Noise variance increased for some units after denoising');
        end
    end

    % Check for degenerate solutions
    if isscalar(results.best_threshold)
        if results.best_threshold == 0
            % Denoiser should be all zeros
            if any(results.denoiser(:) ~= 0)
                error('best_threshold=0 but denoiser is not all zeros');
            end
        end
    else
        if all(results.best_threshold == 0)
            % Denoiser should be all zeros
            if any(results.denoiser(:) ~= 0)
                error('best_threshold=0 but denoiser is not all zeros');
            end
        end
    end

    % Check symmetry of denoiser for global method
    if strcmp(opts.threshold_method, 'global')
        if norm(results.denoiser - results.denoiser', 'fro') > 1e-10
            error('Denoiser should be symmetric for global method');
        end
    end
end


function test_invalid_inputs()
% TEST_INVALID_INPUTS Test that invalid inputs properly fail

    % Generate a small dataset for testing
    [train_data, ~, ~] = simulate('generate_data', ...
        'nvox', 20, 'ncond', 30, 'ntrial', 3, ...
        'random_seed', 42, 'verbose', false, 'want_fig', false);

    invalid_cases = {};

    % Invalid basis type
    invalid_cases{end+1} = struct(...
        'opts', struct('basis', 'invalid_basis', 'wantfig', false, 'wantverbose', false), ...
        'description', 'Invalid basis type');

    % Invalid threshold_method
    invalid_cases{end+1} = struct(...
        'opts', struct('threshold_method', 'invalid_method', 'wantfig', false, 'wantverbose', false), ...
        'description', 'Invalid threshold_method');

    % Invalid criterion
    invalid_cases{end+1} = struct(...
        'opts', struct('criterion', 'invalid_criterion', 'wantfig', false, 'wantverbose', false), ...
        'description', 'Invalid criterion');

    % Negative variance_threshold
    invalid_cases{end+1} = struct(...
        'opts', struct('criterion', 'variance', 'variance_threshold', -0.5, 'wantfig', false, 'wantverbose', false), ...
        'description', 'Negative variance_threshold');

    % variance_threshold > 1
    invalid_cases{end+1} = struct(...
        'opts', struct('criterion', 'variance', 'variance_threshold', 1.5, 'wantfig', false, 'wantverbose', false), ...
        'description', 'variance_threshold > 1');

    % Custom basis with wrong dimensions
    wrong_basis = randn(10, 10);  % Should be 20x20
    invalid_cases{end+1} = struct(...
        'opts', struct('basis', wrong_basis, 'wantfig', false, 'wantverbose', false), ...
        'description', 'Custom basis with wrong dimensions');

    % unit_groups with wrong size
    wrong_groups = ones(10, 1);  % Should be 20x1
    invalid_cases{end+1} = struct(...
        'opts', struct('threshold_method', 'unit', 'unit_groups', wrong_groups, 'wantfig', false, 'wantverbose', false), ...
        'description', 'unit_groups with wrong size');

    % Negative values in allowable_thresholds
    invalid_cases{end+1} = struct(...
        'opts', struct('allowable_thresholds', [-1, 5, 10], 'wantfig', false, 'wantverbose', false), ...
        'description', 'Negative allowable_thresholds');

    % allowable_thresholds exceeding nvox
    invalid_cases{end+1} = struct(...
        'opts', struct('allowable_thresholds', [100, 200], 'wantfig', false, 'wantverbose', false), ...
        'description', 'allowable_thresholds > nvox');

    % allowable_thresholds exceeding ndims for custom basis (smaller than full rank)
    small_custom_basis = randn(20, 5);
    [small_custom_basis, ~] = qr(small_custom_basis, 0);
    invalid_cases{end+1} = struct(...
        'opts', struct('basis', small_custom_basis, 'allowable_thresholds', [3, 10, 15], 'wantfig', false, 'wantverbose', false), ...
        'description', 'allowable_thresholds > ndims (custom basis with 5 dims)');

    % Non-numeric basis
    invalid_cases{end+1} = struct(...
        'opts', struct('basis', 123, 'wantfig', false, 'wantverbose', false), ...
        'description', 'Non-string/matrix basis');

    % Single trial data (PSN requires >= 2 trials)
    [single_trial_data, ~, ~] = simulate('generate_data', ...
        'nvox', 20, 'ncond', 30, 'ntrial', 2, ...
        'random_seed', 42, 'verbose', false, 'want_fig', false);
    single_trial_data = single_trial_data(:, :, 1);  % Keep only first trial
    single_trial_data = reshape(single_trial_data, [size(single_trial_data, 1), size(single_trial_data, 2), 1]);  % Make 3D
    invalid_cases{end+1} = struct(...
        'data', single_trial_data, ...
        'opts', struct('wantfig', false, 'wantverbose', false), ...
        'description', 'Single trial data (< 2 trials required)');

    % Single condition data
    single_cond_data = train_data(:, 1:1, :);
    invalid_cases{end+1} = struct(...
        'data', single_cond_data, ...
        'opts', struct('wantfig', false, 'wantverbose', false), ...
        'description', 'Single condition data (< 2 conditions required)');

    % Condition with all NaNs (no valid trials)
    nan_condition_data = train_data;
    nan_condition_data(:, 1, :) = NaN;  % Condition 1 has all NaNs
    invalid_cases{end+1} = struct(...
        'data', nan_condition_data, ...
        'opts', struct('wantfig', false, 'wantverbose', false), ...
        'description', 'Condition with all NaNs (no valid trials)');

    % Run invalid input tests
    n_total = length(invalid_cases);
    n_passed = 0;
    n_failed = 0;

    fprintf('Testing %d invalid input cases...\n\n', n_total);

    for i = 1:n_total
        test_case = invalid_cases{i};

        try
            % Use custom data if provided, otherwise use default train_data
            if isfield(test_case, 'data')
                test_data_input = test_case.data;
            else
                test_data_input = train_data;
            end

            % This should fail
            results = psn(test_data_input, test_case.opts);

            % If we get here, the test failed (should have thrown error)
            n_failed = n_failed + 1;
            fprintf('[%2d/%2d] ✗ FAIL: %s (did not throw expected error)\n', ...
                i, n_total, test_case.description);

        catch ME
            % Expected to fail
            n_passed = n_passed + 1;
            fprintf('[%2d/%2d] ✓ PASS: %s (correctly rejected)\n', ...
                i, n_total, test_case.description);
        end
    end

    fprintf('\nInvalid input tests: %d/%d passed\n', n_passed, n_total);

    if n_failed > 0
        error('Some invalid inputs were not properly rejected!');
    end
end


function test_different_shapes()
% TEST_DIFFERENT_SHAPES Test PSN on different dataset shapes

    % Define different dataset shapes to test
    shapes = {
        struct('nvox', 10, 'ncond', 20, 'ntrial', 2, 'desc', 'Small dataset'),
        struct('nvox', 20, 'ncond', 40, 'ntrial', 3, 'desc', 'Medium dataset'),
        struct('nvox', 50, 'ncond', 80, 'ntrial', 4, 'desc', 'Large dataset'),
        struct('nvox', 15, 'ncond', 15, 'ntrial', 5, 'desc', 'Square dataset'),
        struct('nvox', 25, 'ncond', 200, 'ntrial', 10, 'desc', 'Many trials'),
        struct('nvox', 5, 'ncond', 10, 'ntrial', 3, 'desc', 'Very small dataset'),
    };

    % Define a subset of option combinations to test on each shape
    test_opts = {
        struct('basis', 'signal', 'threshold_method', 'global', 'criterion', 'prediction'),
        struct('basis', 'noise', 'threshold_method', 'hybrid', 'criterion', 'variance', 'variance_threshold', 0.9),
        struct('basis', 'pca', 'threshold_method', 'unit', 'criterion', 'prediction'),
    };

    n_total = length(shapes) * length(test_opts);
    n_passed = 0;
    n_failed = 0;
    failed_tests = {};

    fprintf('Testing %d combinations across %d dataset shapes...\n\n', ...
        n_total, length(shapes));

    test_idx = 0;
    for s = 1:length(shapes)
        shape = shapes{s};

        % Generate data with this shape
        [train_data, ~, ~] = simulate('generate_data', ...
            'nvox', shape.nvox, ...
            'ncond', shape.ncond, ...
            'ntrial', shape.ntrial, ...
            'random_seed', 42, ...
            'verbose', false, ...
            'want_fig', false);

        for o = 1:length(test_opts)
            test_idx = test_idx + 1;
            opts = test_opts{o};
            opts.wantfig = false;
            opts.wantverbose = false;

            desc = sprintf('%s [%dx%dx%d]: %s/%s/%s', ...
                shape.desc, shape.nvox, shape.ncond, shape.ntrial, ...
                opts.basis, opts.threshold_method, opts.criterion);

            try
                % Run PSN
                results = psn(train_data, opts);

                % Validate outputs
                validate_outputs(results, opts, train_data);

                n_passed = n_passed + 1;
                fprintf('[%2d/%2d] ✓ PASS: %s\n', test_idx, n_total, desc);

            catch ME
                n_failed = n_failed + 1;
                failed_tests{end+1} = struct('desc', desc, 'error', ME);
                fprintf('[%2d/%2d] ✗ FAIL: %s\n', test_idx, n_total, desc);
                fprintf('         Error: %s\n', ME.message);
            end
        end
    end

    fprintf('\nDataset shape tests: %d/%d passed\n', n_passed, n_total);

    if n_failed > 0
        fprintf('\nFailed tests:\n');
        for i = 1:length(failed_tests)
            ft = failed_tests{i};
            fprintf('  %d. %s\n', i, ft.desc);
            fprintf('     Error: %s\n', ft.error.message);
        end
        error('Some dataset shape tests failed!');
    end
end


function test_nan_handling()
% TEST_NAN_HANDLING Test PSN with NaN data (uneven trials across conditions)
%
% This tests that PSN correctly handles data with missing trials (NaNs),
% following GSN's approach for computing average number of trials.

    fprintf('Testing NaN handling...\n\n');

    % Generate base data
    nvox = 20;
    ncond = 30;
    ntrial = 5;

    [base_data, ~, ~] = simulate('generate_data', ...
        'nvox', nvox, 'ncond', ncond, 'ntrial', ntrial, ...
        'random_seed', 42, 'verbose', false, 'want_fig', false);

    % Test cases with different NaN patterns
    nan_test_cases = {};

    %% Test 1: Random missing trials (entire trials, not individual elements)
    data_random_nans = base_data;
    rng(42);
    % Randomly remove entire trials (20% of condition-trial combinations)
    % This ensures each condition still has at least 1 valid trial
    for c = 1:ncond
        % Randomly decide which trials to remove for this condition
        n_to_remove = min(ntrial - 1, floor(ntrial * 0.3)); % Remove 30% but keep at least 1
        if n_to_remove > 0
            trials_to_remove = randperm(ntrial, n_to_remove);
            data_random_nans(:, c, trials_to_remove) = NaN;
        end
    end
    nan_test_cases{end+1} = struct(...
        'data', data_random_nans, ...
        'desc', 'Random missing entire trials (30% removed, at least 1 valid per cond)');

    %% Test 2: Some conditions with fewer trials
    data_uneven = base_data;
    % Conditions 1-10: keep all 5 trials
    % Conditions 11-20: remove trial 5 (4 trials)
    % Conditions 21-30: remove trials 4-5 (3 trials)
    data_uneven(:, 11:20, 5) = NaN;
    data_uneven(:, 21:30, 4:5) = NaN;
    nan_test_cases{end+1} = struct(...
        'data', data_uneven, ...
        'desc', 'Uneven trials: 10 conds @ 5 trials, 10 @ 4, 10 @ 3');

    %% Test 3: Extreme case - most conditions have 2 trials, few have 5
    data_extreme = base_data;
    % Conditions 1-5: keep all 5 trials
    % Conditions 6-30: only keep 2 trials
    data_extreme(:, 6:30, 3:5) = NaN;
    nan_test_cases{end+1} = struct(...
        'data', data_extreme, ...
        'desc', 'Extreme uneven: 5 conds @ 5 trials, 25 conds @ 2 trials');

    %% Test 4: Each condition has different number of trials (graduated)
    data_graduated = base_data;
    for c = 1:ncond
        % Condition c keeps (c mod 5) + 1 trials
        nkeep = mod(c-1, 5) + 1;
        if nkeep < ntrial
            data_graduated(:, c, (nkeep+1):ntrial) = NaN;
        end
    end
    nan_test_cases{end+1} = struct(...
        'data', data_graduated, ...
        'desc', 'Graduated trials: cycling 1-5 trials per condition');

    %% Test 5: All NaNs in specific units for some trials (unit-specific missingness)
    data_unit_specific = base_data;
    % Units 1-5: missing trial 5
    % Units 6-10: missing trials 4-5
    % Units 11-20: all trials present
    data_unit_specific(1:5, :, 5) = NaN;
    data_unit_specific(6:10, :, 4:5) = NaN;
    nan_test_cases{end+1} = struct(...
        'data', data_unit_specific, ...
        'desc', 'Unit-specific missingness');

    % Define subset of PSN options to test with NaN data
    test_opts = {
        struct('basis', 'signal', 'threshold_method', 'global', 'criterion', 'prediction'),
        struct('basis', 'signal', 'threshold_method', 'hybrid', 'criterion', 'prediction'),
        struct('basis', 'signal', 'threshold_method', 'unit', 'criterion', 'prediction'),
        struct('basis', 'difference', 'threshold_method', 'global', 'criterion', 'prediction'),
        struct('basis', 'difference', 'threshold_method', 'unit', 'criterion', 'variance', 'variance_threshold', 0.9),
        struct('basis', 'pca', 'threshold_method', 'global', 'criterion', 'prediction'),
        struct('basis', 'noise', 'threshold_method', 'hybrid', 'criterion', 'variance', 'variance_threshold', 0.95),
    };

    n_total = length(nan_test_cases) * length(test_opts);
    n_passed = 0;
    n_failed = 0;
    failed_tests = {};

    fprintf('Running %d tests (%d NaN patterns x %d option combinations)...\n\n', ...
        n_total, length(nan_test_cases), length(test_opts));

    test_idx = 0;
    for nc = 1:length(nan_test_cases)
        nan_case = nan_test_cases{nc};
        data_with_nans = nan_case.data;

        % Compute expected ntrials_avg for validation
        validcnt = sum(~any(isnan(data_with_nans), 1), 3);
        expected_ntrials_avg = sum(validcnt(validcnt > 1)) / ncond;
        if expected_ntrials_avg < 1
            expected_ntrials_avg = 1;
        end

        for o = 1:length(test_opts)
            test_idx = test_idx + 1;
            opts = test_opts{o};
            opts.wantfig = false;
            opts.wantverbose = false;

            desc = sprintf('%s | %s/%s/%s', ...
                nan_case.desc, opts.basis, opts.threshold_method, opts.criterion);

            try
                % Run PSN with NaN data
                results = psn(data_with_nans, opts);

                % Validate outputs
                validate_outputs(results, opts, data_with_nans);

                % Additional NaN-specific validations
                validate_nan_outputs(results, data_with_nans, expected_ntrials_avg);

                n_passed = n_passed + 1;
                fprintf('[%2d/%2d] ✓ PASS: %s\n', test_idx, n_total, desc);

            catch ME
                n_failed = n_failed + 1;
                failed_tests{end+1} = struct('desc', desc, 'error', ME);
                fprintf('[%2d/%2d] ✗ FAIL: %s\n', test_idx, n_total, desc);
                fprintf('         Error: %s\n', ME.message);
            end
        end
    end

    fprintf('\nNaN handling tests: %d/%d passed\n', n_passed, n_total);

    if n_failed > 0
        fprintf('\nFailed NaN tests:\n');
        for i = 1:length(failed_tests)
            ft = failed_tests{i};
            fprintf('  %d. %s\n', i, ft.desc);
            fprintf('     Error: %s\n', ft.error.message);
        end
        error('Some NaN handling tests failed!');
    end

    fprintf('\n✓ All NaN handling tests passed!\n');
end


function validate_nan_outputs(results, data_with_nans, expected_ntrials_avg)
% VALIDATE_NAN_OUTPUTS Additional validation for NaN data results

    % Denoised data should not contain NaNs
    if any(isnan(results.denoiseddata(:)))
        error('Denoised data contains NaNs (should be filled in)');
    end

    % Denoiser should not contain NaNs
    if any(isnan(results.denoiser(:)))
        error('Denoiser contains NaNs');
    end

    % Residuals may contain NaNs where input data had NaNs (this is expected)
    % But residuals should have NaNs in exactly the same positions as input
    if isfield(results, 'residuals')
        input_nan_mask = isnan(data_with_nans);
        residual_nan_mask = isnan(results.residuals);

        % Check that NaN positions match
        if ~isequal(input_nan_mask, residual_nan_mask)
            warning('Residuals have NaNs in different positions than input data');
        end
    end

    % Unit means should not contain NaNs
    if any(isnan(results.unit_means))
        error('unit_means contains NaNs');
    end

    % Check that the computation used the correct average number of trials
    % This is implicit in the results being valid, but we can check indirectly
    % by ensuring the denoised data is reasonable
    if all(results.denoiseddata(:) == 0)
        error('Denoised data is all zeros (likely incorrect ntrials_avg)');
    end
end


function str = struct2str(s)
% STRUCT2STR Convert struct to string representation
    fields = fieldnames(s);
    parts = {};
    for i = 1:length(fields)
        f = fields{i};
        val = s.(f);
        if isnumeric(val)
            if isscalar(val)
                parts{end+1} = sprintf('%s=%.2f', f, val);
            else
                parts{end+1} = sprintf('%s=[%s]', f, num2str(val(:)', '%.2f '));
            end
        elseif ischar(val)
            parts{end+1} = sprintf('%s=%s', f, val);
        else
            parts{end+1} = sprintf('%s=<complex>', f);
        end
    end
    str = strjoin(parts, ', ');
end
