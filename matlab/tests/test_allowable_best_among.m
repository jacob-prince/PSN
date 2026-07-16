function test_allowable_best_among()
% TEST_ALLOWABLE_BEST_AMONG  best-among-allowable threshold selection (MATLAB).
%
%   When allowable_thresholds restricts the choice, PSN picks the BEST threshold
%   among the allowable values, rather than the unconstrained optimum snapped to the
%   nearest. Mirrors tests/test_allowable_best_among.py.

    here = fileparts(mfilename('fullpath'));
    base = fileparts(here);                       % matlab/
    addpath(genpath(base));                       % psn + all utilities subfolders

    % ---- helpers ----
    obj = [0;5;9;10;8;3;1];                        % unconstrained argmax = 3
    assert(argmax_allowable(obj,[1 5])==1, 'argmax_allowable basic');
    assert(constrain_to_allowable(3,[1 5])==5, 'snap rounds tie up -> differs');
    cumcurve = [0;2;4;6;8;10];
    assert(first_reach_allowable(cumcurve,3,[1 5])==5, 'first_reach reach');
    assert(first_reach_allowable(cumcurve,99,[1 2])==2, 'first_reach none->max');
    assert(argmax_allowable(obj,[4])==4 && first_reach_allowable(cumcurve,3,[4])==4, 'single forces');

    % ---- select_threshold_analytic: best-among != snap ----
    % prediction: diff=[5,4,1,-2,-5,-2] -> objective [0,5,9,10,8,3,1], peak at 3
    signal = [5;4;1;0;0;0]; noise = [0;0;0;2;5;2];
    opt = struct('criterion','prediction','basis','signal', ...
                 'variance_threshold',0.99,'alpha',[],'allowable_thresholds',[1 5]);
    k = select_threshold_analytic(signal, noise, [], 1, opt);
    assert(k==1, 'prediction best-among = 1');
    assert(constrain_to_allowable(3,[1 5])==5, 'prediction old snap would be 5');

    % variance: cumsum [0,2,4,6,8,10], target 0.3*10=3
    signal = [2;2;2;2;2];
    opt = struct('criterion','variance','basis','signal', ...
                 'variance_threshold',0.30,'alpha',[],'allowable_thresholds',[1 5]);
    k = select_threshold_analytic(signal, zeros(5,1), [], 1, opt);
    assert(k==5, 'variance best-among = 5');
    assert(constrain_to_allowable(2,[1 5])==1, 'variance old snap would be 1');

    % ---- end-to-end across criteria: membership + recomputed best-among ----
    rng(1);
    nunits = 20; nconds = 60; ntrials = 4;
    U = randn(nunits,6); sig = U*randn(6,nconds);
    data = repmat(sig,[1,1,ntrials]) + 0.7*randn(nunits,nconds,ntrials);
    allow = [2 5 9 14];

    r = psn(data, struct('basis','signal','threshold_method','global', ...
        'criterion','prediction','allowable_thresholds',allow,'wantfig',false,'wantverbose',false));
    k = r.best_threshold;
    assert(ismember(k, allow) && k==argmax_allowable(r.objective, allow), 'e2e prediction');

    r = psn(data, struct('basis','signal','threshold_method','global', ...
        'criterion','variance','variance_threshold',0.9,'allowable_thresholds',allow, ...
        'wantfig',false,'wantverbose',false));
    k = r.best_threshold; total = r.objective(end);
    assert(ismember(k, allow) && k==first_reach_allowable(r.objective, 0.9*total, allow), 'e2e variance');

    r = psn(data, struct('basis','signal','threshold_method','global', ...
        'criterion','max-tradeoff','allowable_thresholds',allow,'wantfig',false,'wantverbose',false));
    k = r.best_threshold;
    assert(ismember(k, allow) && k==max_tradeoff_threshold(r.signalvar, r.noisevar, ntrials, allow), 'e2e max-tradeoff');

    % single value forces
    r = psn(data, struct('basis','signal','threshold_method','global', ...
        'criterion','prediction','allowable_thresholds',7,'wantfig',false,'wantverbose',false));
    assert(r.best_threshold==7, 'e2e single forces');

    % hybrid membership
    r = psn(data, struct('basis','signal','threshold_method','hybrid', ...
        'criterion','max-tradeoff','allowable_thresholds',allow,'wantfig',false,'wantverbose',false));
    assert(all(ismember(r.best_threshold, allow)), 'e2e hybrid membership');

    fprintf('test_allowable_best_among: ALL PASS\n');
end
