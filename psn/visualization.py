import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.gridspec import GridSpec

from .utils import compute_noise_ceiling, compute_r2


def plot_diagnostic_figures(data, results, test_data=None):
    """
    Generate diagnostic figures for GSN denoising results.
    
    Parameters:
    -----------
    data : ndarray
        Training data used for denoising, shape (nunits, nconds, ntrials)
    results : dict
        Results dictionary from psn
    test_data : ndarray, optional
        Data to use for testing in the bottom row plots, shape (nunits, nconds, ntrials).
        If None, will use leave-one-out cross-validation on the training data.
    """

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a single large figure with proper spacing
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    # Extract data dimensions
    nunits, nconds, ntrials = data.shape

    # Add text at the top of the figure
    V_type = results.get('V')  # Get V directly from results
    if V_type is None:
        V_type = results.get('opt', {}).get('V', 0)  # Fallback to opt dict if not found
    if isinstance(V_type, np.ndarray):
        V_desc = f"user-supplied {V_type.shape}"
    else:
        V_desc = str(V_type)

    # Create title text with data shape and GSN application info
    title_text = f"Data shape: {nunits} units × {nconds} conditions × {ntrials} trials    |    V = {V_desc}\n"

    # Add cv_mode and magnitude thresholding info to title
    cv_mode = results.get('opt', {}).get('cv_mode', 0)
    truncate = results.get('opt', {}).get('truncate', 0)
    
    if cv_mode == -1:
        mag_type = results.get('opt', {}).get('mag_type', 0)
        mag_frac = results.get('opt', {}).get('mag_frac', 0.95)
        mag_frac_str = f"{mag_frac:.3f}".rstrip('0').rstrip('.')
        title_text = (f"Data shape: {nunits} units × {nconds} conditions × {ntrials} trials    |    "
                     f"V = {V_desc}    |    cv_mode = {cv_mode}    |    "
                     f"mag_type = {mag_type}, mag_frac = {mag_frac_str}")
        if truncate > 0:
            title_text += f"    |    truncate = {truncate}"
        title_text += "\n"
    else:
        threshold_per = results.get('opt', {}).get('cv_threshold_per', 'unit')
        title_text = (f"Data shape: {nunits} units × {nconds} conditions × {ntrials} trials    |    "
                     f"V = {V_desc}    |    cv_mode = {cv_mode}    |    thresh = {threshold_per}")
        if truncate > 0:
            title_text += f"    |    truncate = {truncate}"
        title_text += "\n"

    if test_data is None:
        title_text += f"psn applied to all {ntrials} trials"
    else:
        title_text += f"psn applied to {ntrials} trials, tested on 1 heldout trial"

    plt.figtext(0.5, 0.97, title_text,
                ha='center', va='top', fontsize=14)

    # Get raw and denoised data
    if results.get('opt', {}).get('denoisingtype', 0) == 0:
        raw_data = np.mean(data, axis=2)  # Average across trials for trial-averaged denoising
        denoised_data = results['denoiseddata']
    else:
        # For single-trial denoising, we'll plot the first trial
        raw_data = data[:, :, 0] if data.ndim == 3 else data
        denoised_data = results['denoiseddata'][:, :, 0] if results['denoiseddata'].ndim == 3 else results['denoiseddata']

    # Compute noise as difference
    noise = raw_data - denoised_data

    # Initialize lists for basis dimension analysis
    ncsnrs, sigvars, noisevars = [], [], []

    if 'fullbasis' in results and 'mags' in results:
        # Project data into basis
        data_reshaped = data.transpose(1, 2, 0)
        eigvecs = results['fullbasis']
        for i in range(eigvecs.shape[1]):
            this_eigv = eigvecs[:, i]
            proj_data = np.dot(data_reshaped, this_eigv)

            _, ncsnr, sigvar, noisevar = compute_noise_ceiling(proj_data[np.newaxis, ...])
            ncsnrs.append(float(ncsnr[0]))
            sigvars.append(float(sigvar[0]))
            noisevars.append(float(noisevar[0]))

        # Convert to numpy arrays
        sigvars = np.array(sigvars)
        ncsnrs = np.array(ncsnrs)
        noisevars = np.array(noisevars)
        S = results['mags']
        opt = results.get('opt', {})
        best_threshold = results.get('best_threshold', None)
        if best_threshold is None:
            best_threshold = results.get('dimsretained', [])

        # Plot 1: basis source matrix (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        V = results.get('V')

        if isinstance(V, (int, np.integer)):
            if V in [0, 1, 2, 3]:
                # Show the basis source matrix
                if 'basis_source' in results and results['basis_source'] is not None:
                    matrix_to_show = results['basis_source']
                    if V == 0:
                        title = 'GSN Signal Covariance (cSb)'
                    elif V == 1:
                        title = 'GSN Transformed Signal Cov\n(inv(cNb)*cSb)'
                    elif V == 2:
                        title = 'GSN Noise Covariance (cNb)'
                    else:  # V == 3
                        title = 'Naive Trial-avg Data\nCovariance'

                    vmin, vmax = np.percentile(matrix_to_show, [1, 99])

                    im1 = ax1.imshow(matrix_to_show, vmin=vmin, vmax=vmax,
                                   aspect='equal', interpolation='nearest', cmap='RdBu_r')
                    plt.colorbar(im1, ax=ax1, label='Covariance')
                    ax1.set_title(title, pad=10)
                    ax1.set_xlabel('Units')
                    ax1.set_ylabel('Units')
                else:
                    ax1.text(0.5, 0.5, f'Covariance Matrix\nNot Available for V={V}',
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('')
            elif V == 4:  # Random basis
                ax1.text(0.5, 0.5, 'Random Basis\n(No Matrix to Show)',
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('')
            elif V == 5:  # ICA basis
                # Show the ICA components (mixing matrix)
                if 'ica_mixing' in results and results['ica_mixing'] is not None:
                    matrix_to_show = results['ica_mixing']
                    vmin, vmax = np.percentile(matrix_to_show, [1, 99])
                    
                    im1 = ax1.imshow(matrix_to_show, vmin=vmin, vmax=vmax,
                                   aspect='auto', interpolation='nearest', cmap='RdBu_r')
                    plt.colorbar(im1, ax=ax1, label='Component Weight')
                    ax1.set_title('ICA Mixing Matrix\n(Components)', pad=10)
                    ax1.set_xlabel('Component')
                    ax1.set_ylabel('Units')
                else:
                    ax1.text(0.5, 0.5, 'ICA Components\nNot Available',
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('')
            else:
                ax1.text(0.5, 0.5, f'V={V}\n(No Matrix to Show)',
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('')
        elif isinstance(V, np.ndarray):
            # Handle case where V is a matrix
            ax1.text(0.5, 0.5, f'User-Supplied Basis\nShape: {V.shape}',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('User-Supplied Basis')
        else:
            # Handle any other case
            ax1.text(0.5, 0.5, 'No Basis Information Available',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('')

        # Plot 2: Full basis matrix (top middle-left)
        ax2 = fig.add_subplot(gs[0, 1])
        vmin, vmax = np.percentile(results['fullbasis'], [1, 99])
        im2 = ax2.imshow(results['fullbasis'], aspect='auto', interpolation='none',
                        vmin=vmin, vmax=vmax, cmap='RdBu_r')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Full Basis Matrix')
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Units')

        # Plot 3: Magnitude spectrum (top middle)
        # Determine labels based on ranking method
        ranking = results.get('opt', {}).get('ranking', 'signal_variance')

        # Define labels for different ranking methods
        ranking_labels = {
            'eigenvalue': ('Eigenvalues', 'Eigenvalue', 'Eigenspectrum (decreasing)'),
            'eigenvalue_asc': ('Eigenvalues', 'Eigenvalue', 'Eigenspectrum (increasing)'),
            'signal_variance': ('Signal Variance', 'Signal Variance', 'Signal Variance Spectrum'),
            'snr': ('Noise-Ceiling SNR', 'NCSNR', 'NCSNR Spectrum'),
            'signal_specificity': ('Signal% - Noise%', 'Signal% - Noise%', 'Signal Specificity Spectrum')
        }

        legend_label, ylabel, title = ranking_labels.get(ranking, ('Magnitude', 'Magnitude', 'Magnitude Spectrum'))
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(S, linewidth=1, color='blue', label=legend_label)  # Made line thinner

        # Show truncated dimensions if truncate > 0
        truncate = results.get('opt', {}).get('truncate', 0)
        if truncate > 0:
            # Highlight truncated dimensions in red
            truncate_range = min(truncate, len(S))
            ax3.plot(range(truncate_range), S[:truncate_range], 'rx', markersize=6, 
                    label=f'Truncated dims (first {truncate_range})')

        # Calculate and plot threshold indicators based on mode
        cv_mode = results.get('opt', {}).get('cv_mode', 0)
        cv_threshold_per = results.get('opt', {}).get('cv_threshold_per', 'unit')
        mag_type = results.get('opt', {}).get('mag_type', 0)

        if cv_mode >= 0:  # Cross-validation mode
            if cv_threshold_per == 'population':
                # Single line for population threshold
                if isinstance(best_threshold, (np.ndarray, list)):
                    best_threshold = int(best_threshold[0])  # Take first value if array
                ax3.axvline(x=float(best_threshold), color='r', linestyle='--', linewidth=1,
                          label=f'Population threshold: {best_threshold} dims')
            else:  # Unit mode
                # Mean line and asterisks for unit-specific thresholds
                if isinstance(best_threshold, (np.ndarray, list)):
                    mean_threshold = np.mean(best_threshold)
                    ax3.axvline(x=float(mean_threshold), color='r', linestyle='--', linewidth=1,
                              label=f'Mean threshold: {mean_threshold:.1f} dims')
                    # Add asterisks at the top for each unit's threshold
                    unique_thresholds = np.unique(best_threshold)
                    ylim = ax3.get_ylim()
                    for thresh in unique_thresholds:
                        ax3.plot(thresh, ylim[1], 'r*', markersize=5,
                               label="")
        else:  # Magnitude thresholding mode - show included dimensions
            if isinstance(best_threshold, (np.ndarray, list)) and len(best_threshold) > 0:
                # Add circles for included dimensions
                best_threshold_array = np.asarray(best_threshold, dtype=int)
                ax3.plot(best_threshold_array, S[best_threshold_array], 'ro', markersize=4,
                        label='Included dimensions')
                # Show vertical line for number of dimensions retained
                threshold_len = len(best_threshold)
                ax3.axvline(x=float(threshold_len), color='r', linestyle='--', linewidth=1,
                          label=f'Dims retained: {threshold_len}')

        ax3.set_xlabel('Dimension')
        ax3.set_ylabel(ylabel)
        ax3.set_title(f'Denoising Basis\n{title}')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Signal and noise variances with NCSNR (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(sigvars, linewidth=1, label='Sig. var')
        ax4.plot(noisevars, linewidth=1, label='Noise var')

        # Show truncated dimensions if truncate > 0
        truncate = results.get('opt', {}).get('truncate', 0)
        if truncate > 0:
            # Highlight truncated dimensions in red
            truncate_range = min(truncate, len(sigvars))
            ax4.plot(range(truncate_range), sigvars[:truncate_range], 'rx', markersize=6, 
                    label=f'Truncated dims (first {truncate_range})')

        # Handle thresholds based on mode
        cv_mode = results.get('opt', {}).get('cv_mode', 0)
        if cv_mode >= 0:  # Cross-validation mode
            if isinstance(best_threshold, (np.ndarray, list)):
                if len(best_threshold) > 0:
                    threshold_val = np.mean(best_threshold)
                    ax4.axvline(x=float(threshold_val), color='r', linestyle='--', linewidth=1,
                              label=f'Mean thresh: {threshold_val:.1f} dims')
            else:
                # Ensure scalar value for axvline
                ax4.axvline(x=float(best_threshold), color='r', linestyle='--', linewidth=1,
                           label=f'Thresh: {best_threshold} dims')
        else:  # Magnitude thresholding mode
            if isinstance(best_threshold, (np.ndarray, list)):
                threshold_len = len(best_threshold)
                ax4.axvline(x=float(threshold_len), color='r', linestyle='--', linewidth=1,
                           label=f'Dims retained: {threshold_len}')
                # Add circles for included dimensions
                if mag_type == 0 and len(best_threshold) > 0:
                    best_threshold_array = np.asarray(best_threshold, dtype=int)
                    ax4.plot(best_threshold_array, sigvars[best_threshold_array], 'ro', markersize=4,
                            label='Included dimensions')
            else:
                # Ensure scalar value for axvline
                ax4.axvline(x=float(best_threshold), color='r', linestyle='--', linewidth=1,
                           label=f'Dims retained: {best_threshold}')

        # Add NCSNR on secondary y-axis
        ax4_twin = ax4.twinx()
        ax4_twin.plot(ncsnrs, linewidth=1, color='magenta', label='NCSNR')
        ax4_twin.set_ylabel('NCSNR', color='magenta')
        ax4_twin.tick_params(axis='y', labelcolor='magenta')

        # Combine legends from both axes
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')

        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Variance')
        ax4.set_title('Signal and Noise Variance for \nData Projected into Basis')
        ax4.grid(True, alpha=0.3, which='both', axis='both')  # Enable grid for both axes
        ax4_twin.grid(False)  # Disable grid for twin axis to avoid double grid lines

       # Plot 5: Cross-validation results (first subplot in middle row)
        ax5 = fig.add_subplot(gs[1, 0])
        cv_mode = results.get('opt', {}).get('cv_mode', 0)
        if 'cv_scores' in results and cv_mode in [0, 1] and len(results['cv_scores']) > 0:
            #cv_data = stats.zscore(results['cv_scores'].mean(1),axis=1,ddof=1)
            cv_data = results['cv_scores'].mean(1)#,axis=0,ddof=1)

            # Get thresholds, handling both list and array types
            cv_thresholds = opt.get('cv_thresholds', np.arange(results['cv_scores'].shape[0]))
            if isinstance(cv_thresholds, list):
                thresholds = np.array(cv_thresholds)
            else:
                thresholds = cv_thresholds

            # Truncate thresholds that exceed data dimensionality
            max_dim = results['cv_scores'].shape[0]
            valid_mask = thresholds < max_dim
            thresholds = thresholds[valid_mask]
            cv_data = cv_data[valid_mask]

            cv_data = stats.zscore(cv_data,axis=0,ddof=1)
            vmin, vmax = np.percentile(cv_data, [0, 100])

            # Plot without extent - use default pixel coordinates
            im5 = ax5.imshow(cv_data.T, aspect='auto', interpolation='none',
                    clim=(vmin, vmax))
            plt.colorbar(im5, ax=ax5)

            # Update xlabel based on whether truncation is used
            truncate = results.get('opt', {}).get('truncate', 0)
            if truncate > 0:
                ax5.set_xlabel(f'PC threshold (starting from PC {truncate})')
            else:
                ax5.set_xlabel('PC exclusion threshold')
            ax5.set_ylabel('Units')
            ax5.set_title('Cross-validation scores (z)')

            # Set x-ticks to show actual threshold values at pixel centers
            step = max(len(thresholds) // 10, 1)  # Show ~10 ticks or less
            tick_positions = np.arange(0, len(thresholds), step)  # Pixel indices
            tick_labels = thresholds[::step]
            ax5.set_xticks(tick_positions)
            ax5.set_xticklabels(tick_labels)
            ax5.tick_params(axis='x', rotation=90)

            if results.get('opt', {}).get('cv_threshold_per') == 'unit':
                if isinstance(best_threshold, np.ndarray) and len(best_threshold) == nunits:
                    # For each unit, find the threshold index that gives maximum CV score
                    unit_indices = np.arange(nunits)  # Pixel row indices
                    threshold_positions = []

                    # Check if unit_groups are being used
                    unit_groups = results.get('opt', {}).get('unit_groups', np.arange(nunits))

                    if 'unit_groups' in results.get('opt', {}) and not np.array_equal(unit_groups, np.arange(nunits)):
                        # Unit groups are being used - show group-based thresholds
                        unique_groups = np.unique(unit_groups)

                        for unit_idx in range(nunits):
                            # Find which group this unit belongs to
                            unit_group = unit_groups[unit_idx]

                            # Get all units in this group
                            group_mask = unit_groups == unit_group

                            # Average CV scores across units in this group
                            group_cv_scores = np.mean(cv_data[:, group_mask], axis=1)  # Average across group units

                            # Find threshold index with maximum group score
                            max_thresh_idx = np.argmax(group_cv_scores)

                            # Position at pixel column index
                            threshold_positions.append(max_thresh_idx)
                    else:
                        # No unit grouping - use individual unit's maximum CV score
                        for unit_idx in range(nunits):
                            # Get CV scores for this unit across all thresholds
                            unit_cv_scores = cv_data[:, unit_idx]  # cv_data shape: (n_thresholds, n_units)

                            # Find threshold index with maximum score
                            max_thresh_idx = np.argmax(unit_cv_scores)

                            # Position at pixel column index
                            threshold_positions.append(max_thresh_idx)

                    ax5.plot(threshold_positions, unit_indices, 'r.', markersize=4)
        elif cv_mode == 2:
            # Analytic threshold selection - show signal/noise variance curves
            # Compute average signal and noise variance across units
            gsn_result = results.get('gsn_result', {})
            if gsn_result and 'cSb' in gsn_result and 'cNb' in gsn_result:
                basis = results['fullbasis']
                cSb = gsn_result['cSb']
                cNb = gsn_result['cNb']

                # Project covariances into basis space
                cSb_basis = basis.T @ cSb @ basis
                cNb_basis = basis.T @ cNb @ basis

                signal_proj = np.diag(cSb_basis)
                noise_proj = np.diag(cNb_basis)

                # Average across units (using squared loadings)
                avg_signal_var = np.zeros(basis.shape[1])
                avg_noise_var = np.zeros(basis.shape[1])

                for d in range(basis.shape[1]):
                    unit_loadings_sq = basis[:, d] ** 2
                    avg_signal_var[d] = np.mean(unit_loadings_sq * signal_proj[d])
                    avg_noise_var[d] = np.mean(unit_loadings_sq * noise_proj[d])

                # Rank by same method used in analytic threshold selection
                ranking = opt.get('ranking', 'snr')

                if ranking == 'snr':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        scores = np.divide(avg_signal_var, avg_noise_var,
                                      out=np.zeros_like(avg_signal_var),
                                      where=avg_noise_var > 0)
                    sort_idx = np.argsort(scores)[::-1]
                elif ranking == 'signal_variance':
                    scores = avg_signal_var
                    sort_idx = np.argsort(scores)[::-1]
                elif ranking == 'snd':
                    scores = avg_signal_var - avg_noise_var
                    sort_idx = np.argsort(scores)[::-1]
                elif ranking == 'eigenvalue':
                    # Keep original basis ordering
                    sort_idx = np.arange(len(avg_signal_var))
                else:
                    # Default to SNR
                    with np.errstate(divide='ignore', invalid='ignore'):
                        scores = np.divide(avg_signal_var, avg_noise_var,
                                      out=np.zeros_like(avg_signal_var),
                                      where=avg_noise_var > 0)
                    sort_idx = np.argsort(scores)[::-1]

                avg_signal_var_sorted = avg_signal_var[sort_idx]
                avg_noise_var_sorted = avg_noise_var[sort_idx]

                # Scale noise by ntrials
                avg_noise_var_scaled = avg_noise_var_sorted / ntrials
                sig_noise_diff = avg_signal_var_sorted - avg_noise_var_scaled

                # Plot
                dims = np.arange(len(avg_signal_var_sorted))
                ax5.plot(dims, avg_signal_var_sorted, 'b-', linewidth=2, label='Signal Var')
                ax5.plot(dims, avg_noise_var_sorted, 'r-', linewidth=2, label='Noise Var')
                ax5.plot(dims, avg_noise_var_scaled, 'r--', linewidth=2, label=f'Noise Var / {ntrials}')
                ax5.plot(dims, sig_noise_diff, 'g-', linewidth=2, label='Signal - Noise/n')
                ax5.axhline(0, color='k', linestyle=':', linewidth=1)

                # Show average threshold
                if isinstance(best_threshold, np.ndarray):
                    avg_thresh = best_threshold.mean()
                else:
                    avg_thresh = best_threshold

                ax5.axvline(avg_thresh, color='orange', linestyle='--', linewidth=2,
                           label=f'Threshold (mean={avg_thresh:.1f})')

                # Set xlabel based on ranking method
                ranking_labels = {
                    'snr': 'SNR-sorted',
                    'signal_variance': 'Signal Var-sorted',
                    'snd': 'SND-sorted',
                    'eigenvalue': 'Eigenvalue order'
                }
                ranking_label = ranking_labels.get(ranking, 'sorted')
                ax5.set_xlabel(f'Basis Dimension ({ranking_label})')
                ax5.set_ylabel('Variance')
                ax5.set_title('Analytic Threshold Selection')
                ax5.legend(fontsize=8)
                ax5.grid(True, alpha=0.3)
                ax5.set_xlim([0, min(100, len(dims))])  # Show first 100 dims

                # Set y-limits to focus on region around threshold crossing
                # Use percentiles to be robust to outliers
                xlim_max = min(100, len(dims))
                visible_diff = sig_noise_diff[:xlim_max]
                if len(visible_diff) > 0:
                    # Focus on the difference curve since that's what matters
                    ymin = np.percentile(visible_diff, 5)
                    ymax = np.percentile(visible_diff, 95)
                    # Ensure zero is visible and add some padding
                    ymin = min(ymin, 0)
                    ymax = max(ymax, 0)
                    y_range = ymax - ymin
                    ax5.set_ylim([ymin - 0.1 * y_range, ymax + 0.1 * y_range])
            else:
                # Fallback if GSN results not available
                if isinstance(best_threshold, np.ndarray):
                    ax5.hist(best_threshold, bins=20, alpha=0.7, color='C2', edgecolor='black')
                    ax5.axvline(best_threshold.mean(), color='red', linestyle='--',
                               label=f'Mean: {best_threshold.mean():.1f}')
                    ax5.set_xlabel('Threshold (dimensions retained)')
                    ax5.set_ylabel('Number of units')
                    ax5.set_title('Analytic Threshold Selection')
                    ax5.legend()
                    ax5.grid(True, alpha=0.3)
                else:
                    ax5.text(0.5, 0.5, f'Analytic Threshold: {best_threshold}',
                            ha='center', va='center', transform=ax5.transAxes, fontsize=14)
                    ax5.set_title('Analytic Threshold Selection')
        else:
            ax5.text(0.5, 0.5, 'No Cross-validation\nScores Available',
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Cross-validation scores')

        # Plot 6-8: Raw data, denoised data, noise (rest of middle row)
        all_data = np.concatenate([raw_data.flatten(), denoised_data.flatten(), noise.flatten()])
        vmin, vmax = np.percentile(all_data, [1, 99])
        data_clim = (vmin, vmax)

        # Raw data
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = plt.imshow(raw_data, aspect='auto', interpolation='none', clim=data_clim, cmap='RdBu_r')
        plt.colorbar(im6)
        plt.title('Input Data (trial-averaged)')
        plt.xlabel('Conditions')
        plt.ylabel('Units')

        # Denoised data
        ax7 = fig.add_subplot(gs[1, 2])
        im7 = plt.imshow(denoised_data, aspect='auto', interpolation='none', clim=data_clim, cmap='RdBu_r')
        plt.colorbar(im7)
        plt.title('Data projected into basis')
        plt.xlabel('Conditions')
        plt.ylabel('Units')

        # Noise
        ax8 = fig.add_subplot(gs[1, 3])
        im8 = plt.imshow(noise, aspect='auto', interpolation='none', clim=data_clim, cmap='RdBu_r')
        plt.colorbar(im8)
        plt.title('Residual')
        plt.xlabel('Conditions')
        plt.ylabel('Units')

        # Plot denoising matrix (first subplot in bottom row)
        ax9 = fig.add_subplot(gs[2, 0])
        vmin, vmax = np.percentile(results['denoiser'], [1, 99])
        denoiser_clim = (vmin, vmax)
        im9 = plt.imshow(results['denoiser'], aspect='auto', interpolation='none', clim=denoiser_clim, cmap='RdBu_r')
        plt.colorbar(im9)
        plt.title('Optimal Basis Matrix')
        plt.xlabel('Units')
        plt.ylabel('Units')

        # Compute R2 and correlations for bottom row
        if test_data is None:
            # Use leave-one-out cross-validation on training data
            raw_r2_per_unit = np.zeros((ntrials, nunits))
            denoised_r2_per_unit = np.zeros((ntrials, nunits))
            raw_corr_per_unit = np.zeros((ntrials, nunits))
            denoised_corr_per_unit = np.zeros((ntrials, nunits))

            for tr in range(ntrials):
                train_trials = np.setdiff1d(np.arange(ntrials), tr)
                train_avg = np.mean(data[:, :, train_trials], axis=2)
                test_trial = data[:, :, tr]

                for v in range(nunits):
                    raw_r2_per_unit[tr, v] = compute_r2(test_trial[v], train_avg[v])
                    raw_corr_per_unit[tr, v] = np.corrcoef(test_trial[v], train_avg[v])[0, 1]

                # Demean before denoising for consistent handling
                train_avg_demeaned = (train_avg.T - results['unit_means']).T
                test_trial_demeaned = (test_trial.T - results['unit_means']).T
                train_avg_denoised = (train_avg_demeaned.T @ results['denoiser']).T + results['unit_means'][:, np.newaxis]
                test_trial_denoised = (test_trial_demeaned.T @ results['denoiser']).T + results['unit_means'][:, np.newaxis]

                for v in range(nunits):
                    denoised_r2_per_unit[tr, v] = compute_r2(test_trial[v], train_avg_denoised[v])
                    denoised_corr_per_unit[tr, v] = np.corrcoef(test_trial[v], train_avg_denoised[v])[0, 1]
        else:
            # Use provided test data
            if np.ndim(test_data) > 2:
                test_avg = np.mean(test_data, axis=2)
            else:
                test_avg = test_data
            train_avg = np.mean(data, axis=2)

            raw_r2_per_unit = np.zeros((1, nunits))
            denoised_r2_per_unit = np.zeros((1, nunits))
            raw_corr_per_unit = np.zeros((1, nunits))
            denoised_corr_per_unit = np.zeros((1, nunits))

            for v in range(nunits):
                raw_r2_per_unit[0, v] = compute_r2(test_avg[v], train_avg[v])
                
                # Compute correlation, handling zero-variance case
                if np.std(test_avg[v]) == 0 or np.std(train_avg[v]) == 0:
                    raw_corr_per_unit[0, v] = np.nan
                else:
                    raw_corr_per_unit[0, v] = np.corrcoef(test_avg[v], train_avg[v])[0, 1]

            # Demean before denoising for consistent handling
            train_avg_demeaned = (train_avg.T - results['unit_means']).T
            test_avg_demeaned = (test_avg.T - results['unit_means']).T
            train_avg_denoised = (train_avg_demeaned.T @ results['denoiser']).T + results['unit_means'][:, np.newaxis]
            test_avg_denoised = (test_avg_demeaned.T @ results['denoiser']).T + results['unit_means'][:, np.newaxis]

            for v in range(nunits):
                denoised_r2_per_unit[0, v] = compute_r2(test_avg[v], train_avg_denoised[v])
                
                # Compute correlation, handling zero-variance case
                if np.std(test_avg[v]) == 0 or np.std(train_avg_denoised[v]) == 0:
                    denoised_corr_per_unit[0, v] = np.nan
                else:
                    denoised_corr_per_unit[0, v] = np.corrcoef(test_avg[v], train_avg_denoised[v])[0, 1]

        # Compute mean and SEM
        raw_r2_mean = np.mean(raw_r2_per_unit, axis=0)
        raw_r2_sem = stats.sem(raw_r2_per_unit, axis=0)
        denoised_r2_mean = np.mean(denoised_r2_per_unit, axis=0)
        denoised_r2_sem = stats.sem(denoised_r2_per_unit, axis=0)

        raw_corr_mean = np.mean(raw_corr_per_unit, axis=0)
        raw_corr_sem = stats.sem(raw_corr_per_unit, axis=0)
        denoised_corr_mean = np.mean(denoised_corr_per_unit, axis=0)
        denoised_corr_sem = stats.sem(denoised_corr_per_unit, axis=0)

        # Plot trial-averaged and denoised traces similar to EEG notebook
        # Get trial-averaged data
        trial_avg_full = np.mean(data, axis=2)  # (nunits, nconds)
        denoised_full = results['denoiseddata']  # Already trial-averaged if denoisingtype=0

        # If denoised data is 3D, average it
        if denoised_full.ndim == 3:
            denoised_full = np.mean(denoised_full, axis=2)

        # Calculate mean response across conditions for rainbow coloring
        cond_means = np.mean(trial_avg_full, axis=0)  # Mean across units for each condition
        sorted_cond_indices = np.argsort(cond_means)
        colors = plt.cm.rainbow(np.linspace(0, 1, nconds))

        # Create color array where each condition gets its color based on rank
        trace_colors = np.zeros((nconds, 4))  # RGBA
        for rank, cond_idx in enumerate(sorted_cond_indices):
            trace_colors[cond_idx] = colors[rank]

        # Plot trial-averaged traces
        ax_tavg = fig.add_subplot(gs[2, 1])
        for cond_idx in range(nconds):
            ax_tavg.plot(trial_avg_full[:, cond_idx], color=trace_colors[cond_idx],
                        linewidth=0.5, alpha=0.7)

        ax_tavg.set_xlabel('Units')
        ax_tavg.set_ylabel('Activity')
        ax_tavg.set_title('Trial-Averaged Traces\n(rainbow: conditions by mean response)')
        ax_tavg.grid(True, alpha=0.3)

        # Plot denoised traces
        ax_dn = fig.add_subplot(gs[2, 2])
        for cond_idx in range(nconds):
            ax_dn.plot(denoised_full[:, cond_idx], color=trace_colors[cond_idx],
                      linewidth=0.5, alpha=0.7)

        ax_dn.set_xlabel('Units')
        ax_dn.set_ylabel('Activity')
        ax_dn.set_title('PSN Denoised Traces\n(same condition coloring)')
        ax_dn.grid(True, alpha=0.3)

        # Match y-axis limits across both plots
        all_trace_data = np.concatenate([trial_avg_full.flatten(), denoised_full.flatten()])
        y_min = np.min(all_trace_data)
        y_max = np.max(all_trace_data)
        y_range = y_max - y_min
        y_margin = y_range * 0.05
        ax_tavg.set_ylim(y_min - y_margin, y_max + y_margin)
        ax_dn.set_ylim(y_min - y_margin, y_max + y_margin)

        # Add split-half correlation comparison plot
        ax_prog = fig.add_subplot(gs[2, 3])

        # Split trials in half
        half_idx = ntrials // 2
        data_A = data[:, :, :half_idx]  # First half of trials
        data_B = data[:, :, half_idx:]  # Second half of trials

        # Compute trial averages for each split
        trial_avg_A = np.mean(data_A, axis=2)  # (nunits, nconds)
        trial_avg_B = np.mean(data_B, axis=2)  # (nunits, nconds)

        # Denoise each split separately
        denoiser = results['denoiser']
        unit_means = results['unit_means']

        # Denoise split A
        trial_avg_A_demeaned = trial_avg_A - unit_means[:, np.newaxis]
        denoised_A = (trial_avg_A_demeaned.T @ denoiser).T + unit_means[:, np.newaxis]

        # Denoise split B
        trial_avg_B_demeaned = trial_avg_B - unit_means[:, np.newaxis]
        denoised_B = (trial_avg_B_demeaned.T @ denoiser).T + unit_means[:, np.newaxis]

        # Compute correlations for each unit
        corr_tavg_tavg = np.zeros(nunits)  # Trial avg A vs trial avg B
        corr_cross_AB = np.zeros(nunits)   # Trial avg A vs denoised B
        corr_cross_BA = np.zeros(nunits)   # Denoised A vs trial avg B
        corr_dn_dn = np.zeros(nunits)      # Denoised A vs denoised B

        for unit_idx in range(nunits):
            # Trial avg vs trial avg
            if np.std(trial_avg_A[unit_idx]) > 0 and np.std(trial_avg_B[unit_idx]) > 0:
                corr_tavg_tavg[unit_idx] = np.corrcoef(trial_avg_A[unit_idx], trial_avg_B[unit_idx])[0, 1]
            else:
                corr_tavg_tavg[unit_idx] = np.nan

            # Cross-method: trial avg A vs denoised B
            if np.std(trial_avg_A[unit_idx]) > 0 and np.std(denoised_B[unit_idx]) > 0:
                corr_cross_AB[unit_idx] = np.corrcoef(trial_avg_A[unit_idx], denoised_B[unit_idx])[0, 1]
            else:
                corr_cross_AB[unit_idx] = np.nan

            # Cross-method: denoised A vs trial avg B
            if np.std(denoised_A[unit_idx]) > 0 and np.std(trial_avg_B[unit_idx]) > 0:
                corr_cross_BA[unit_idx] = np.corrcoef(denoised_A[unit_idx], trial_avg_B[unit_idx])[0, 1]
            else:
                corr_cross_BA[unit_idx] = np.nan

            # Denoised vs denoised
            if np.std(denoised_A[unit_idx]) > 0 and np.std(denoised_B[unit_idx]) > 0:
                corr_dn_dn[unit_idx] = np.corrcoef(denoised_A[unit_idx], denoised_B[unit_idx])[0, 1]
            else:
                corr_dn_dn[unit_idx] = np.nan

        # Average the two cross-method correlations
        corr_cross = (corr_cross_AB + corr_cross_BA) / 2

        # Three x positions for the three comparison types
        x_positions = np.array([1, 2, 3])
        labels = ['TAvg vs\nTAvg', 'TAvg vs\nDenoised', 'Denoised vs\nDenoised']

        # Add small jitter to x positions
        jitter = 0.08
        x_jitter = np.random.uniform(-jitter, jitter, nunits)

        # Plot connecting lines for each unit
        for unit_idx in range(nunits):
            values = [corr_tavg_tavg[unit_idx], corr_cross[unit_idx], corr_dn_dn[unit_idx]]
            if not np.any(np.isnan(values)):
                x_vals = x_positions + x_jitter[unit_idx]
                plt.plot(x_vals, values, 'gray', linewidth=0.3, alpha=0.4, zorder=1)

        # Plot individual dots
        plt.scatter(x_positions[0] + x_jitter, corr_tavg_tavg, s=15, c='blue', alpha=0.4, zorder=2)
        plt.scatter(x_positions[1] + x_jitter, corr_cross, s=15, c='gold', alpha=0.4, zorder=2)
        plt.scatter(x_positions[2] + x_jitter, corr_dn_dn, s=15, c='limegreen', alpha=0.4, zorder=2)

        # Compute means (excluding NaN)
        mean_tavg_tavg = np.nanmean(corr_tavg_tavg)
        mean_cross = np.nanmean(corr_cross)
        mean_dn_dn = np.nanmean(corr_dn_dn)
        mean_values = [mean_tavg_tavg, mean_cross, mean_dn_dn]

        # Plot mean dots (larger, with edge)
        plt.scatter(x_positions[0], mean_tavg_tavg, s=100, c='darkblue', edgecolors='white', linewidth=2, zorder=3)
        plt.scatter(x_positions[1], mean_cross, s=100, c='gold', edgecolors='white', linewidth=2, zorder=3)
        plt.scatter(x_positions[2], mean_dn_dn, s=100, c='darkgreen', edgecolors='white', linewidth=2, zorder=3)

        # Add mean values as text above the dots
        y_text_offset = 0.08
        plt.text(x_positions[0], mean_tavg_tavg + y_text_offset, f'{mean_tavg_tavg:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(x_positions[1], mean_cross + y_text_offset, f'{mean_cross:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(x_positions[2], mean_dn_dn + y_text_offset, f'{mean_dn_dn:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Formatting
        plt.xticks(x_positions, labels, fontsize=8)
        plt.ylabel('Pearson r')
        plt.title(f'Split-Half Reliability ({nunits} units)\nSplit: {data_A.shape[2]} vs {data_B.shape[2]} trials')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xlim(0.5, 3.5)

        # Set y-axis limits based on data range with some padding
        all_corr_values = np.concatenate([corr_tavg_tavg, corr_cross, corr_dn_dn])
        valid_corr_values = all_corr_values[~np.isnan(all_corr_values)]
        if len(valid_corr_values) > 0:
            y_min = np.min(valid_corr_values)
            y_max = np.max(valid_corr_values)
            y_range = y_max - y_min
            y_padding = max(0.1, y_range * 0.15)  # At least 0.1, or 15% of range
            plt.ylim(y_min - y_padding, y_max + y_padding)
        else:
            plt.ylim(-1, 1)

        plt.axhline(y=0, color='k', linewidth=1, alpha=0.5, zorder=0)
    plt.show()
