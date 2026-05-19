function [sig, noi] = project_covs(cS, cN, B)
% PROJECT_COVS  Project covariances into basis
%
%   [sig, noi] = project_covs(cS, cN, B) computes the signal and noise
%   variance along each dimension of the basis B by projecting the
%   covariance matrices into the basis coordinate system.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <cS> - [nunits x nunits] symmetric signal covariance matrix from GSN
%
% <cN> - [nunits x nunits] symmetric noise covariance matrix from GSN
%
% <B> - [nunits x ndims] orthonormal basis matrix with basis vectors as columns
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <sig> - [ndims x 1] signal variance along each basis dimension.
%   Mathematically: sig(i) = B(:,i)' * cS * B(:,i), which is the i-th
%   diagonal element of B' * cS * B
%
% <noi> - [ndims x 1] noise variance along each basis dimension.
%   Mathematically: noi(i) = B(:,i)' * cN * B(:,i)
%
% -------------------------------------------------------------------------
% Implementation notes:
% -------------------------------------------------------------------------
%
% Uses efficient element-wise computation: diag(B' * C * B) = sum((C * B) .* B, 1)'
% This is O(N^2 * K) instead of O(N^3) for full matrix multiplication.
% Small negative values from numerical error are clamped to zero.

    % Efficient diagonal extraction (avoids full matrix multiplication)
    % diag(B' * C * B)[i] = B[:,i]' * C * B[:,i] = sum(B[:,i] .* (C * B[:,i]))
    sig = sum((cS * B) .* B, 1)';
    noi = sum((cN * B) .* B, 1)';

    % Clamp tiny negatives from numerical error
    sig = max(sig, 0);
    noi = max(noi, 0);
end
