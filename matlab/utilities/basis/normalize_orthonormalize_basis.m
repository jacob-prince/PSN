function basis = normalize_orthonormalize_basis(basis)
% NORMALIZE_ORTHONORMALIZE_BASIS  Ensure basis has orthonormal columns
%
%   basis = normalize_orthonormalize_basis(basis) takes a matrix and
%   ensures its columns are orthonormal (unit length and mutually orthogonal).
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <basis> - [n x k] numeric matrix with k basis vectors as columns
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <basis> - [n x k] matrix with orthonormal columns. First normalizes each
%   column to unit length, then checks orthogonality. If not orthogonal
%   (Gram matrix not identity within tolerance 1e-10), applies QR
%   decomposition to enforce orthonormality

    norms = sqrt(sum(basis.^2, 1));
    norms(norms == 0) = 1;
    basis = basis ./ norms;

    gram = basis' * basis;
    if ~all(all(abs(gram - eye(size(gram))) < 1e-10))
        [basis, ~] = qr(basis, 0);
    end
end
