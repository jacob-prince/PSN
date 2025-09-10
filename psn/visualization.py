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

                    matrix_max = np.max(np.abs(matrix_to_show))

                    im1 = ax1.imshow(matrix_to_show, vmin=-matrix_max, vmax=matrix_max,
                                   aspect='equal', interpolation='nearest', cmap='RdBu_r')
                    plt.colorbar(im1, ax=ax1, label='Covariance')
                    ax1.set_title(title, pad=10)
                    ax1.set_xlabel('Units')
                    ax1.set_ylabel('Units')
                else:
                    ax1.text(0.5, 0.5, f'Covariance Matrix\nNot Available for V={V}',
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('')
            else:  # V == 4
                ax1.text(0.5, 0.5, 'Random Basis\n(No Matrix to Show)',
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
        basis_max = np.max(np.abs(results['fullbasis']))
        im2 = ax2.imshow(results['fullbasis'], aspect='auto', interpolation='none',
                        clim=(-basis_max, basis_max), cmap='RdBu_r')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Full Basis Matrix')
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Units')

        # Plot 3: Eigenspectrum (top middle)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(S, linewidth=1, color='blue', label='Eigenvalues')  # Made line thinner

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
        ax3.set_ylabel('Eigenvalue')
        ax3.set_title('Denoising Basis\nEigenspectrum')
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
        if 'cv_scores' in results and results.get('opt', {}).get('cv_mode', 0) > -1:
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

            # Set proper extent for imshow - ensure no white columns
            extent = [0, len(thresholds), nunits, 0]

            im5 = ax5.imshow(cv_data.T, aspect='auto', interpolation='none',
                    clim=(vmin, vmax), extent=extent)
            plt.colorbar(im5, ax=ax5)
            
            # Update xlabel based on whether truncation is used
            truncate = results.get('opt', {}).get('truncate', 0)
            if truncate > 0:
                ax5.set_xlabel(f'PC threshold (starting from PC {truncate})')
            else:
                ax5.set_xlabel('PC exclusion threshold')
            ax5.set_ylabel('Units')
            ax5.set_title('Cross-validation scores (z)')

            # Set x-ticks to show actual threshold values
            step = max(len(thresholds) // 10, 1)  # Show ~10 ticks or less
            tick_positions = np.arange(0, len(thresholds), step) + 0.5  # Center of bins
            tick_labels = thresholds[::step]
            ax5.set_xticks(tick_positions)
            ax5.set_xticklabels(tick_labels)
            ax5.tick_params(axis='x', rotation=90)

            if results.get('opt', {}).get('cv_threshold_per') == 'unit':
                if isinstance(best_threshold, np.ndarray) and len(best_threshold) == nunits:
                    # For each unit, find the threshold index that gives maximum CV score
                    unit_indices = np.arange(nunits) + 0.5  # Center dots in cells
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

                            # Position at center of that threshold's cell
                            threshold_positions.append(max_thresh_idx + 0.5)
                    else:
                        # No unit grouping - use individual unit's maximum CV score
                        for unit_idx in range(nunits):
                            # Get CV scores for this unit across all thresholds
                            unit_cv_scores = cv_data[:, unit_idx]  # cv_data shape: (n_thresholds, n_units)

                            # Find threshold index with maximum score
                            max_thresh_idx = np.argmax(unit_cv_scores)

                            # Position at center of that threshold's cell
                            threshold_positions.append(max_thresh_idx + 0.5)

                    ax5.plot(threshold_positions, unit_indices, 'r.', markersize=4)
        else:
            ax5.text(0.5, 0.5, 'No Cross-validation\nScores Available',
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Cross-validation scores')

        # Plot 6-8: Raw data, denoised data, noise (rest of middle row)
        all_data = np.concatenate([raw_data.flatten(), denoised_data.flatten(), noise.flatten()])
        max_abs_val = np.max(np.abs(all_data))
        data_clim = (-max_abs_val, max_abs_val)

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
        denoiser_max = np.max(np.abs(results['denoiser']))
        denoiser_clim = (-denoiser_max, denoiser_max)
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
                raw_corr_per_unit[0, v] = np.corrcoef(test_avg[v], train_avg[v])[0, 1]

            # Demean before denoising for consistent handling
            train_avg_demeaned = (train_avg.T - results['unit_means']).T
            test_avg_demeaned = (test_avg.T - results['unit_means']).T
            train_avg_denoised = (train_avg_demeaned.T @ results['denoiser']).T + results['unit_means'][:, np.newaxis]
            test_avg_denoised = (test_avg_demeaned.T @ results['denoiser']).T + results['unit_means'][:, np.newaxis]

            for v in range(nunits):
                denoised_r2_per_unit[0, v] = compute_r2(test_avg[v], train_avg_denoised[v])
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

        # Function to plot bottom row with rotated histograms
        def plot_bottom_histogram(ax, r2_mean, corr_mean, r2_color, corr_color, title):
            plt.sca(ax)
            plt.axvline(x=0, color='k', linewidth=2, zorder=1)

            # Calculate histogram bins
            bins = np.linspace(-1, 1, 50)
            bin_width = bins[1] - bins[0]

            # Plot R2 histogram
            r2_hist, _ = np.histogram(r2_mean, bins=bins)  # Remove density=True
            plt.bar(bins[:-1] + bin_width/2, r2_hist, width=bin_width,
                    color=r2_color, alpha=0.6, label=f'Mean R² = {np.mean(r2_mean):.3f}')

            # Plot correlation histogram
            corr_hist, _ = np.histogram(corr_mean, bins=bins)  # Remove density=True
            plt.bar(bins[:-1] + bin_width/2, corr_hist, width=bin_width,
                    color=corr_color, alpha=0.6, label=f'Mean r = {np.mean(corr_mean):.3f}')

            plt.ylabel('# Units')  # Updated label
            plt.xlabel('R² / Pearson r')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-1, 1)

        # Plot bottom row histograms and R² progression
        train_trials = ntrials-1 if test_data is None else data.shape[2]
        test_trials = 1 if test_data is None else (test_data.shape[2] if len(test_data.shape) > 2 else 1)

        plot_bottom_histogram(fig.add_subplot(gs[2, 1]),
                            raw_r2_mean, raw_corr_mean,
                            'blue', 'lightblue',
                            f'Baseline Generalization\nTrial-avg Train ({train_trials} trials) vs\nTrial-avg Test ({test_trials} trials)')

        plot_bottom_histogram(fig.add_subplot(gs[2, 2]),
                            denoised_r2_mean, denoised_corr_mean,
                            'green', 'lightgreen',
                            f'Denoised Generalization\nTrial-avg Train + denoised ({train_trials} trials) vs\nTrial-avg Test ({test_trials} trials)')

        # Add R² progression plot
        ax_prog = fig.add_subplot(gs[2, 3])
        x_positions = [1, 2]  # Two positions for the two conditions

        # Plot lines for each unit
        for v in range(nunits):
            values = [raw_r2_mean[v], denoised_r2_mean[v]]
            plt.plot(x_positions, values, color='gray', alpha=0.2, linewidth=0.5)
            plt.scatter(x_positions[0], values[0], alpha=0.5, s=20, color='blue')
            plt.scatter(x_positions[1], values[1], alpha=0.5, s=20, color='green')

        # Plot mean performance
        mean_values = [np.mean(raw_r2_mean), np.mean(denoised_r2_mean)]
        plt.plot(x_positions, mean_values, color='pink', linewidth=2, label='Mean')
        plt.scatter(x_positions[0], mean_values[0], color='blue', s=100, edgecolor='pink', linewidth=2)
        plt.scatter(x_positions[1], mean_values[1], color='green', s=100, edgecolor='pink', linewidth=2)

        plt.xticks(x_positions, ['Trial Averaged', 'With Denoising'])
        plt.ylabel('R²')
        plt.title(f'Impact of denoising on R² ({nunits} units)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0.5, 2.5)
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='k', linewidth=2, zorder=1)
    plt.show()
