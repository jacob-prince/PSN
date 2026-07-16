"""Diagnostic plotting for simulated data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def plot_data_diagnostic(data, ground_truth, params):
    """Generate a comprehensive diagnostic figure for simulated data.

    Creates a 12-subplot figure showing eigenvalue spectra, signal-to-noise
    ratios, covariance matrices, ground truth signal, example trial data,
    eigenvectors, alignment visualization, and trial-averaged data.

    Parameters
    ----------
    data : ndarray, shape (nvox, ncond, ntrial)
        The simulated data.
    ground_truth : dict
        Dictionary with ground truth information containing:
        - signal_cov : ndarray, shape (nvox, nvox) - signal covariance matrix
        - noise_cov : ndarray, shape (nvox, nvox) - noise covariance matrix
        - signal_eigs : ndarray, shape (nvox,) - signal eigenvalues
        - noise_eigs : ndarray, shape (nvox,) - noise eigenvalues
        - U_signal : ndarray, shape (nvox, nvox) - signal eigenvectors
        - U_noise : ndarray, shape (nvox, nvox) - noise eigenvectors
        - signal : ndarray, shape (ncond, nvox) - ground truth signal
    params : dict
        Dictionary with simulation parameters containing:
        - nvox : int - number of units
        - ncond : int - number of conditions
        - ntrial : int - number of trials
        - signal_decay : float - signal eigenvalue decay rate
        - noise_decay : float - noise eigenvalue decay rate
        - noise_multiplier : float - noise scaling factor
        - align_alpha : float - alignment strength (0=orthogonal, 1=aligned)
        - align_k : int - number of aligned dimensions
        - user_provided : dict (optional) - what was user-provided
        - clustered : bool (optional) - whether units were reordered

    Returns
    -------
    fig : matplotlib.figure.Figure
        The diagnostic figure.
    """
    # Extract important parameters and ground truth values
    nvox = params['nvox']
    ncond = params['ncond']
    ntrial = params['ntrial']
    signal_decay = params['signal_decay']
    noise_decay = params['noise_decay']
    noise_multiplier = params['noise_multiplier']
    align_alpha = params['align_alpha']
    align_k = params['align_k']
    user_provided = params.get('user_provided', {'signal_cov': False, 'true_signal': False})

    # Extract ground truth matrices
    signal_cov = ground_truth['signal_cov']
    noise_cov = ground_truth['noise_cov']
    signal_eigs = ground_truth['signal_eigs']
    noise_eigs = ground_truth['noise_eigs']
    U_signal = ground_truth['U_signal']
    U_noise = ground_truth['U_noise']
    true_signal = ground_truth['signal']

    # Create example trial data for visualization
    trial_avg = np.mean(data, axis=2)  # Average across trials
    example_trial = data[:, :, 0]      # First trial

    # Create figure with GridSpec for flexible subplot layout
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.8, 1, 1])

    # Add title with parameters
    title_text = (
        f"Simulated Data: {nvox} units × {ncond} conditions × {ntrial} trials\n"
    )

    # Add appropriate source info based on what was user-provided
    if user_provided.get('true_signal', False):
        title_text += "Using user-provided ground truth signal\n"
    elif user_provided.get('signal_cov', False):
        title_text += "Using user-provided signal covariance matrix\n"
    else:
        title_text += f"Signal decay={signal_decay:.2f}, Noise decay={noise_decay:.2f}, "
        title_text += f"Noise multiplier={noise_multiplier:.2f}\n"

    title_text += f"Alignment: alpha={align_alpha:.2f} (0=orthogonal, 1=aligned), k={align_k} top PCs"

    # Add note about clustering if units were reordered
    if params.get('clustered', False):
        title_text += "\nUnits reordered by hierarchical clustering"

    fig.suptitle(title_text, fontsize=14)

    # Split the first row into two parts for log and linear eigenvalue plots
    gs_eigen = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0:2])

    # Plot 1a: Eigenvalue spectra - Log scale
    ax1a = fig.add_subplot(gs_eigen[0, 0])
    ax1a.semilogy(np.arange(nvox), signal_eigs, 'b-', label='Signal eigenvalues')
    ax1a.semilogy(np.arange(nvox), noise_eigs, 'r-', label='Noise eigenvalues')
    if align_k > 0:
        ax1a.axvline(x=align_k-1, color='gray', linestyle='--',
                   label=f'Alignment cutoff (k={align_k})')
    ax1a.set_xlabel('Dimension')
    ax1a.set_ylabel('Eigenvalue (log scale)')
    ax1a.set_title('Eigenspectrum - Log Scale')
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)

    # Plot 1b: Eigenvalue spectra - Linear scale
    ax1b = fig.add_subplot(gs_eigen[0, 1])
    ax1b.plot(np.arange(nvox), signal_eigs, 'b-', label='Signal eigenvalues')
    ax1b.plot(np.arange(nvox), noise_eigs, 'r-', label='Noise eigenvalues')
    if align_k > 0:
        ax1b.axvline(x=align_k-1, color='gray', linestyle='--',
                   label=f'Alignment cutoff (k={align_k})')
    ax1b.set_xlabel('Dimension')
    ax1b.set_ylabel('Eigenvalue (linear scale)')
    ax1b.set_title('Eigenspectrum - Linear Scale')
    # Show top 25% of dimensions for better visualization in linear scale
    dims_to_show = max(int(nvox * 0.25), align_k + 5)
    ax1b.set_xlim(0, dims_to_show)
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    # Plot 2: Signal-to-noise ratio per dimension
    ax2 = fig.add_subplot(gs[0, 2:4])
    snr = signal_eigs / noise_eigs
    ax2.plot(np.arange(nvox), snr, 'g-')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('SNR')
    ax2.set_title('Signal-to-Noise Ratio per Dimension')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Signal covariance matrix
    ax3 = fig.add_subplot(gs[1, 0])
    # Calculate clim values similar to gsn_denoise.py
    cov_max = np.percentile(np.abs(signal_cov), 95)
    im3 = ax3.imshow(signal_cov, aspect='equal', cmap='RdBu_r', interpolation='none',
                   vmin=-cov_max, vmax=cov_max)
    plt.colorbar(im3, ax=ax3, label='Covariance')

    # Update title to be more informative about signal_cov source
    if user_provided.get('true_signal', False):
        title_suffix = "(Derived from user-provided GT signal)"
    elif user_provided.get('signal_cov', False):
        title_suffix = ""
    else:
        title_suffix = ""

    ax3.set_title(f'Signal Covariance Matrix\n{title_suffix}')
    ax3.set_xlabel('Unit')
    ax3.set_ylabel('Unit')

    # Plot 4: Noise covariance matrix
    ax4 = fig.add_subplot(gs[1, 1])
    # Use same colormap and scaling approach as signal covariance
    cov_max = np.percentile(np.abs(noise_cov), 95)
    im4 = ax4.imshow(noise_cov, aspect='equal', cmap='RdBu_r', interpolation='none',
                   vmin=-cov_max, vmax=cov_max)
    plt.colorbar(im4, ax=ax4, label='Covariance')
    ax4.set_title('Noise Covariance Matrix')
    ax4.set_xlabel('Unit')
    ax4.set_ylabel('Unit')

    # Calculate shared colorbar limits for signal and trial data
    # Combine the ground truth signal and example trial data to find common limits
    all_data = np.concatenate([true_signal.T.flatten(), example_trial.flatten()])
    data_min = np.percentile(all_data, 1)  # Use 1st percentile instead of min to avoid outliers
    data_max = np.percentile(all_data, 99)  # Use 99th percentile instead of max to avoid outliers
    data_abs_max = max(abs(data_min), abs(data_max))

    # Use symmetric limits for better visualization
    signal_clim = (-data_abs_max, data_abs_max)

    # Plot 5: Example of ground truth signal - show full matrix
    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(true_signal.T, aspect='auto', cmap='RdBu_r', interpolation='none',
                   vmin=signal_clim[0], vmax=signal_clim[1])
    plt.colorbar(im5, ax=ax5, label='Signal')
    title_prefix = "" if user_provided.get('true_signal', False) else ""
    ax5.set_title(f'{title_prefix}Ground Truth Signal')
    ax5.set_xlabel('Condition')
    ax5.set_ylabel('Unit')

    # Plot 6: Example trial data - show full matrix
    ax6 = fig.add_subplot(gs[1, 3])
    im6 = ax6.imshow(example_trial, aspect='auto', cmap='RdBu_r', interpolation='none',
                   vmin=signal_clim[0], vmax=signal_clim[1])
    plt.colorbar(im6, ax=ax6, label='Activity')
    ax6.set_title('Example Single Trial (with noise)')
    ax6.set_xlabel('Condition')
    ax6.set_ylabel('Unit')

    # Plot 7: Signal eigenvectors. Use a data-driven symmetric clim - a fixed
    # +/-0.3 washes out at large nvox, where eigenvector entries scale ~1/sqrt(nvox).
    sig_evec_max = np.percentile(np.abs(U_signal), 99) or 1.0
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(U_signal, aspect='auto', cmap='RdBu_r',
                   vmin=-sig_evec_max, vmax=sig_evec_max)
    plt.colorbar(im7, ax=ax7, label='Weight')
    ax7.set_title('Signal Eigenvectors')
    ax7.set_xlabel('Dimension')
    ax7.set_ylabel('Unit')

    # Plot 8: Noise eigenvectors (top k)
    noise_evec_max = np.percentile(np.abs(U_noise), 99) or 1.0
    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(U_noise, aspect='auto', cmap='RdBu_r',
                   vmin=-noise_evec_max, vmax=noise_evec_max)
    plt.colorbar(im8, ax=ax8, label='Weight')
    ax8.set_title('Noise Eigenvectors')
    ax8.set_xlabel('Dimension')
    ax8.set_ylabel('Unit')

    # Plot 9: Alignment visualization (dot products between signal and noise eigenvectors)
    ax9 = fig.add_subplot(gs[2, 2])
    # Calculate full alignment matrix for all dimensions
    # Use absolute values because eigenvectors are defined up to sign
    full_alignment_matrix = np.abs(U_signal.T @ U_noise)

    # Determine how many dimensions to display in the visualization
    # If nvox is large, subsample the matrix but ensure we include the aligned dimensions
    max_dims_to_show = min(25, nvox)  # Limit to 50x50 at most for readability

    if nvox <= max_dims_to_show:
        # If we have fewer dimensions than the limit, show all
        alignment_matrix = full_alignment_matrix
        dims_shown = nvox
    else:
        # Otherwise, show a subset with emphasis on aligned dimensions
        if align_k > 0:
            # Always include the aligned dimensions
            indices = list(range(align_k))

            # Add additional dimensions, evenly spaced
            remaining_spots = max_dims_to_show - align_k
            if remaining_spots > 0:
                # Determine spacing for remaining dimensions
                step = (nvox - align_k) / (remaining_spots + 1)
                for i in range(1, remaining_spots + 1):
                    idx = align_k + int(i * step)
                    indices.append(min(idx, nvox - 1))  # Ensure we don't exceed bounds

            # Sort indices to maintain proper order
            indices = sorted(indices)
        else:
            # No alignment, just evenly space the indices
            indices = np.linspace(0, nvox-1, max_dims_to_show, dtype=int)

        # Extract the submatrix
        alignment_matrix = full_alignment_matrix[indices, :][:, indices]
        dims_shown = len(indices)

    # Create the visualization
    im9 = ax9.imshow(alignment_matrix, aspect='equal', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im9, ax=ax9, label='|Dot product|')
    ax9.set_title('Eigenvector Alignment (|dot products|)')
    ax9.set_xlabel('Noise dimension')
    ax9.set_ylabel('Signal dimension')

    # Add grid lines to better separate the dimensions
    ax9.set_xticks(np.arange(-.5, dims_shown, 1), minor=True)
    ax9.set_yticks(np.arange(-.5, dims_shown, 1), minor=True)
    ax9.grid(which="minor", color="w", linestyle='-', linewidth=0.5, alpha=0.2)

    # Add diagonal values text to show exact alignment of corresponding eigenvectors
    if align_k > 0:
        for i in range(min(dims_shown, align_k)):
            ax9.text(i, i, f'{alignment_matrix[i, i]:.2f}',
                   ha='center', va='center', color='white', fontweight='bold')

    # Add some axis labels for clarity, using sparse labeling if there are many dimensions
    if dims_shown <= 10:
        ax9.set_xticks(np.arange(dims_shown))
        ax9.set_yticks(np.arange(dims_shown))
    else:
        # Show fewer ticks for readability
        step = max(1, dims_shown // 10)
        ax9.set_xticks(np.arange(0, dims_shown, step))
        ax9.set_yticks(np.arange(0, dims_shown, step))

    # Show the total number of dimensions in title if we're displaying a subset
    if dims_shown < nvox:
        ax9.set_title(f'Eigenvector Alignment (showing {dims_shown}/{nvox} dimensions)')

    # Plot 10: Trial-averaged data - also use the same colorbar limits
    ax10 = fig.add_subplot(gs[2, 3])
    im10 = ax10.imshow(trial_avg, aspect='auto', cmap='RdBu_r', interpolation='none',
                     vmin=signal_clim[0], vmax=signal_clim[1])
    plt.colorbar(im10, ax=ax10, label='Activity')
    ax10.set_title(f'Trial-averaged Data ({ntrial} trials)')
    ax10.set_xlabel('Condition')
    ax10.set_ylabel('Unit')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

    return fig
