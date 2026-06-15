function Q = random_orthonormal_columns(nvox, r)
% RANDOM_ORTHONORMAL_COLUMNS  Random matrix with orthonormal columns.
%
%   Q = random_orthonormal_columns(nvox, r) returns an [nvox x r] matrix
%   whose columns are orthonormal, obtained from a reduced QR factorization
%   of a Gaussian random matrix. Used by the fast low-rank simulation path
%   to build signal/noise bases without forming full (nvox, nvox) matrices.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <nvox> - positive integer. Number of rows (units/voxels).
%
% <r> - positive integer. Number of orthonormal columns to return.
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <Q> - [nvox x r] matrix with orthonormal columns (Q' * Q == I).

    if r <= 0
        error('r must be positive');
    end
    A = randn(nvox, r);
    [Q, ~] = qr(A, 0);  % reduced (economy) QR
end
