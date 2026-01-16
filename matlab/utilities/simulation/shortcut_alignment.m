function U_noise_adj = shortcut_alignment(U_signal, U_noise, k)
% SHORTCUT_ALIGNMENT Directly align the first k noise PCs to the signal PCs without optimization.
%
% This method provides exact alignment (alpha=1.0) and maintains orthonormality
% through Gram-Schmidt orthogonalization.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <U_signal> - [nvox x nvox] orthonormal signal basis
%
% <U_noise> - [nvox x nvox] orthonormal noise basis
%
% <k> - int, number of top PCs to align
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <U_noise_adj> - [nvox x nvox] orthonormal basis with first k columns aligned to U_signal

    U_noise_adj = U_noise;
    nvox = size(U_signal, 1);

    % Set top k noise PCs equal to signal PCs
    U_noise_adj(:, 1:k) = U_signal(:, 1:k);

    % Orthonormalize remaining PCs via Gram-Schmidt
    for i = k+1:nvox
        v = U_noise_adj(:, i);

        % Orthogonalize against all previous columns (including the aligned ones)
        for j = 1:i-1
            v = v - (v' * U_noise_adj(:, j)) * U_noise_adj(:, j);
        end

        norm_v = norm(v);
        if norm_v < 1e-12
            % Choose a random orthogonal vector if degenerate
            attempts = 0;
            while norm_v < 1e-12 && attempts < nvox
                v = randn(nvox, 1);
                % Orthogonalize against all previous columns
                for j = 1:i-1
                    v = v - (v' * U_noise_adj(:, j)) * U_noise_adj(:, j);
                end
                norm_v = norm(v);
                attempts = attempts + 1;
            end

            if norm_v < 1e-12
                % Last resort: use standard basis vector
                for basis_idx = 1:nvox
                    v = zeros(nvox, 1);
                    v(basis_idx) = 1.0;
                    % Orthogonalize against all previous columns
                    for j = 1:i-1
                        v = v - (v' * U_noise_adj(:, j)) * U_noise_adj(:, j);
                    end
                    norm_v = norm(v);
                    if norm_v > 1e-12
                        break;
                    end
                end
            end
        end

        U_noise_adj(:, i) = v / norm_v;
    end
end
