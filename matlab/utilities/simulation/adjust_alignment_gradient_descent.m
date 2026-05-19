function U_noise_aligned = adjust_alignment_gradient_descent(U_signal, U_noise_init, alpha, k, verbose)
% ADJUST_ALIGNMENT_GRADIENT_DESCENT Gradient descent method to align U_noise to U_signal's top-k PCs
% with dot(U_noise(:, i), U_signal(:, i)) ≈ alpha.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <U_signal> - [nvox x nvox] orthonormal signal basis
%
% <U_noise_init> - [nvox x nvox] orthonormal noise basis (initial)
%
% <alpha> - scalar in [0, 1], target alignment strength
%   0 = orthogonal, 1 = perfectly aligned
%
% <k> - number of top PCs to align
%
% <verbose> (optional) - logical, whether to print convergence info. Default: true
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <U_noise_aligned> - [nvox x nvox] new orthonormal basis with first k columns
%   aligned to U_signal with strength alpha

    if nargin < 5
        verbose = true;
    end

    % Default parameters
    lr = 5e-1;
    lambda_orth = 1.0;
    num_steps = 10000;
    tol_align = 1e-6;
    tol_orth = 1e-6;

    % Handle edge cases
    if k == 0
        U_noise_aligned = U_noise_init;
        return;
    end

    % Use shortcut alignment for perfect alignment (alpha = 1.0)
    % This avoids convergence issues and ensures exact orthonormality
    if abs(alpha - 1.0) < 1e-10
        U_noise_aligned = shortcut_alignment(U_signal, U_noise_init, k);
        return;
    end

    U = U_noise_init;
    nvox = size(U, 1);
    I = eye(nvox);

    for step = 1:num_steps
        grad = zeros(size(U));
        align_vals = zeros(k, 1);
        for i = 1:k
            dot_val = U(:, i)' * U_signal(:, i);
            align_vals(i) = dot_val;
            grad(:, i) = (dot_val - alpha) * U_signal(:, i);
        end
        M = U' * U - I;
        grad = grad + lambda_orth * (U * M);
        U = U - lr * grad;

        max_align_err = max(abs(align_vals - alpha));
        orth_err = norm(U' * U - I, 'fro');

        if max_align_err < tol_align && orth_err < tol_orth
            if verbose
                fprintf('\t\tOptimization complete. Step %d/%d: align_err=%.2e, orth_err=%.2e\n', ...
                       step, num_steps, max_align_err, orth_err);
            end
            % Ensure perfect orthonormality before returning
            [U, ~] = qr(U, 0);
            U_noise_aligned = U;
            return;
        end
    end

    if verbose
        fprintf('Optimization did not converge. Step %d/%d: align_err=%.2e, orth_err=%.2e\n', ...
               num_steps, num_steps, max_align_err, orth_err);
    end

    % Ensure orthonormality even if didn't fully converge
    [U, ~] = qr(U, 0);
    U_noise_aligned = U;
end
