function U_noise = align_noise_basis_lowrank(U_signal, U_noise_init, alpha, k)
% ALIGN_NOISE_BASIS_LOWRANK  Low-rank alignment of top-k noise PCs to signal PCs.
%
%   U_noise = align_noise_basis_lowrank(U_signal, U_noise_init, alpha, k)
%   produces an orthonormal [nvox x rN] matrix whose first k columns satisfy
%   dot(U_noise(:,i), U_signal(:,i)) == alpha (approximately, but very close).
%
%   This is a scalable alternative to adjust_alignment_gradient_descent, which
%   requires full (nvox, nvox) matrices. Used by the fast low-rank simulation
%   path.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <U_signal> - [nvox x rS] orthonormal signal basis (columns are PCs).
%
% <U_noise_init> - [nvox x rN] orthonormal initial noise basis.
%
% <alpha> - scalar in [0, 1]. Target alignment strength of the top-k columns
%   (0 = orthogonal, 1 = perfectly aligned). Clamped to [0,1] with a warning.
%
% <k> - integer. Number of top columns to align; <= 0 returns U_noise_init.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <U_noise> - [nvox x rN] orthonormal noise basis whose first k columns are
%   aligned to U_signal with strength alpha.

    if k <= 0
        U_noise = U_noise_init;
        return;
    end

    if ~(alpha >= 0.0 && alpha <= 1.0)
        warning('align_alpha must be in [0,1]; will be clamped.');
        alpha = max(0.0, min(1.0, alpha));
    end

    nvox = size(U_signal, 1);
    rS = size(U_signal, 2);
    rN = size(U_noise_init, 2);
    k_eff = min([k, rS, rN, floor(nvox / 2)]);
    if k_eff <= 0
        U_noise = U_noise_init;
        return;
    end

    Usk = U_signal(:, 1:k_eff);

    % Construct V: k_eff orthonormal vectors orthogonal to span(Usk)
    A = randn(nvox, k_eff);
    A = A - Usk * (Usk' * A);
    [V, ~] = qr(A, 0);

    % Aligned block; columns remain orthonormal because Usk is orthogonal to V
    aligned = alpha * Usk + sqrt(max(0.0, 1.0 - alpha^2)) * V;

    % Fill remaining noise directions from the initial noise basis,
    % projected to the complement of [Usk, V]
    B = [Usk, V];  % (nvox, 2*k_eff)
    rest = U_noise_init(:, (k_eff + 1):end);
    if isempty(rest)
        U_noise = aligned;
        return;
    end

    rest = rest - B * (B' * rest);
    [rest, ~] = qr(rest, 0);

    U_noise = [aligned, rest];
end
