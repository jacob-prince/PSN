function [evals_sorted, evecs_sorted] = eigh_descending_sym(matrix, do_symmetrize)
% EIGH_DESCENDING_SYM  Compute eigendecomposition with consistent sorting
%
%   [evals_sorted, evecs_sorted] = eigh_descending_sym(matrix) computes
%   the eigendecomposition of a symmetric matrix and returns eigenvalues
%   and eigenvectors sorted by descending eigenvalue magnitude, with
%   standardized eigenvector signs for reproducibility.
%
%   [evals_sorted, evecs_sorted] = eigh_descending_sym(matrix, do_symmetrize)
%   optionally forces the matrix to be symmetric before eigendecomposition.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <matrix> - [n x n] numeric matrix (should be symmetric or nearly symmetric)
%
% <do_symmetrize> (optional) - logical. If true, enforces symmetry via
%   (matrix + matrix')/2 before eigendecomposition. Default: false.
%   Note: GSN returns symmetric cSb/cNb, so symmetrization is typically
%   only needed for derived matrices like cSb - cNb/ntrials_avg
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <evals_sorted> - [n x 1] eigenvalues sorted in descending order
%
% <evecs_sorted> - [n x n] eigenvectors with columns sorted to match
%   <evals_sorted>. Signs standardized so that the largest-magnitude
%   element in each column is positive

    if nargin < 2
        do_symmetrize = false; %#ok<NASGU>  % retained for API compatibility
    end

    % Always symmetrize. This function is for symmetric matrices, and MATLAB's
    % general eig() returns COMPLEX eigenvectors when given a matrix with even
    % machine-epsilon asymmetry (which GSN covariances can have). Python's
    % np.linalg.eigh is always a symmetric solver, so symmetrizing here keeps the
    % results real and matches Python. For an exactly-symmetric matrix this is a
    % no-op, so it does not change any existing result.
    matrix = (matrix + matrix') / 2;

    % Compute eigendecomposition
    [evecs, evals] = eig(matrix, 'vector');

    % Sort by eigenvalue magnitude (descending)
    [evals_sorted, order] = sort(evals, 'descend');
    evecs_sorted = evecs(:, order);

    % Deterministic sign: make largest-magnitude element positive
    [~, piv] = max(abs(evecs_sorted), [], 1);
    idx = sub2ind(size(evecs_sorted), piv, 1:size(evecs_sorted, 2));
    sgn = sign(evecs_sorted(idx));
    sgn(sgn == 0) = 1;
    evecs_sorted = evecs_sorted .* sgn;
end
