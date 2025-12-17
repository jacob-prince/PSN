"""PSN visualization - matches MATLAB visualization.m exactly"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.gridspec import GridSpec


def plot_diagnostic_figures(data, results, test_data=None):
    """
    Generate diagnostic figures for PSN denoising results (NEW API).

    This visualization works with the new PSN API and results structure.

    Parameters:
    -----------
    data : ndarray
        Training data used for denoising, shape (nunits, nconds, ntrials)
    results : dict
        Results dictionary from psn function
    test_data : ndarray, optional
        Not used in current implementation (reserved for future use)
    """

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a large figure with 4x4 grid
    fig = plt.figure(figsize=(18, 12))

    # Extract data dimensions
    nunits, nconds, ntrials = data.shape

    # Get options if stored
    if 'opt_used' in results:
        opt = results['opt_used']
    else:
        opt = {}

    # Extract basis type description
    if 'basis' in opt:
        if isinstance(opt['basis'], str):
            basis_desc = opt['basis']
        else:
            basis_desc = f"custom [{opt['basis'].shape[0]}x{opt['basis'].shape[1]}]"
    else:
        basis_desc = 'unknown'

    # Extract threshold method
    threshold_method = opt.get('threshold_method', 'unknown')

    # Extract criterion
    criterion = opt.get('criterion', 'unknown')

    # Extract basis_ordering
    basis_ordering = opt.get('basis_ordering', 'eigenvalues')

    # Check for NaNs and compute average number of trials
    has_nans = np.any(np.isnan(data))
    if has_nans:
        validcnt = np.sum(~np.any(np.isnan(data), axis=0), axis=1)
        ntrials_avg = np.sum(validcnt[validcnt > 1]) / nconds
        if ntrials_avg < 1:
            ntrials_avg = 1
    else:
        ntrials_avg = ntrials

    # Create title (order: Basis, Criterion, Method to match API)
    if has_nans:
        title_text = f'Data: {nunits} units × {nconds} conditions × {ntrials} max trials (avg {ntrials_avg:.1f})  |  Basis: {basis_desc}  |  Criterion: {criterion}  |  Method: {threshold_method}'
    else:
        title_text = f'Data: {nunits} units × {nconds} conditions × {ntrials} trials  |  Basis: {basis_desc}  |  Criterion: {criterion}  |  Method: {threshold_method}'

    plt.suptitle(title_text, fontsize=12, fontweight='bold')

    # Get trial-averaged and denoised data (use nanmean for NaN data)
    if has_nans:
        trial_avg = np.nanmean(data, axis=2)
    else:
        trial_avg = np.mean(data, axis=2)

    denoised = results['denoiseddata']
    noise = trial_avg - denoised

    # =========================================================================
    # Plot 1: Basis source matrix (covariance)
    # =========================================================================
    ax1 = plt.subplot(4, 4, 1)
    if 'gsn_result' in results and 'cSb' in results['gsn_result']:
        cSb = results['gsn_result']['cSb']
        cNb = results['gsn_result']['cNb']

        # Determine which matrix based on basis type
        basis_type = opt.get('basis', 'signal')
        if isinstance(basis_type, str):
            if basis_type == 'difference':
                plot_matrix_1 = cSb - cNb / ntrials_avg
                plot_title = f'cSb - cNb/{ntrials_avg:.1f} (difference)'
            elif basis_type == 'noise':
                plot_matrix_1 = cNb
                plot_title = 'Noise Covariance (cNb)'
            elif basis_type == 'pca':
                trial_avg_demeaned = trial_avg - results['unit_means'][:, np.newaxis]
                plot_matrix_1 = np.cov(trial_avg_demeaned)
                plot_title = 'Trial-Avg Data Covariance'
            else:  # 'signal' or default
                plot_matrix_1 = cSb
                plot_title = 'Signal Covariance (cSb)'
        else:
            plot_matrix_1 = cSb
            plot_title = 'Signal Covariance (cSb)'

        # Compute z-score based colorbar limits (±3 SD)
        if has_nans:
            data_mean = np.nanmean(plot_matrix_1)
            data_std = np.nanstd(plot_matrix_1)
        else:
            data_mean = np.mean(plot_matrix_1)
            data_std = np.std(plot_matrix_1)

        if data_std > 0:
            clim_1 = [data_mean - 3*data_std, data_mean + 3*data_std]
        else:
            clim_1 = [data_mean - 1, data_mean + 1]

        im1 = ax1.imshow(plot_matrix_1, vmin=clim_1[0], vmax=clim_1[1], cmap='RdBu_r', aspect='equal')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(plot_title)
        ax1.set_xlabel('Units')
        ax1.set_ylabel('Units')
    else:
        ax1.text(0.5, 0.5, 'Covariance\nNot Available',
                ha='center', va='center', transform=ax1.transAxes)

    # =========================================================================
    # Plot 2: Top 5 PCs as vertical line plots
    # =========================================================================
    ax2 = plt.subplot(4, 4, 2)
    if 'fullbasis' in results:
        num_pcs = min(5, results['fullbasis'].shape[1])

        # Normalize each PC for visualization
        y_units = np.arange(1, nunits + 1)
        colors = plt.cm.tab10(np.linspace(0, 1, num_pcs))

        # Find max absolute loading across top 5 PCs for scaling
        max_loading = np.max(np.abs(results['fullbasis'][:, :num_pcs]))
        if max_loading > 0:
            scale_factor = 0.4 / max_loading  # Scale to fit within 0.4 x-units
        else:
            scale_factor = 1

        for pc in range(num_pcs):
            # Center each PC at position pc+1, with loadings as horizontal deviations
            x_vals = (pc + 1) + results['fullbasis'][:, pc] * scale_factor
            ax2.plot(x_vals, y_units, linewidth=1.5, color=colors[pc])

            # Add vertical reference line at center
            ax2.axvline(x=pc + 1, color='k', linestyle='--', linewidth=0.5)

        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Units')
        ax2.set_title('Top 5 Basis Dimensions')
        ax2.set_xlim([0.5, num_pcs + 0.5])
        ax2.set_ylim([1, nunits])
        ax2.set_xticks(range(1, num_pcs + 1))
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'Basis\nNot Available',
                ha='center', va='center', transform=ax2.transAxes)

    # =========================================================================
    # Plot 3: Global dimension ranking (eigenvalues or signal variance)
    # =========================================================================
    ax3 = plt.subplot(4, 4, 3)

    # Determine if we should use log spacing (more than 50 units)
    use_log_x = nunits > 50

    # Check if eigenvalues were used for ranking (and are available)
    use_eigenvalues = (basis_ordering == 'eigenvalues' and
                      'basis_eigenvalues' in results and
                      results['basis_eigenvalues'] is not None and
                      len(results['basis_eigenvalues']) > 0)

    if use_eigenvalues:
        # Show eigenvalues (SORTED - what was actually used for ranking)
        evals = results['basis_eigenvalues']  # Already sorted in descending order
        x_vals = np.arange(1, len(evals) + 1)

        if use_log_x:
            ax3.semilogx(x_vals, evals, linewidth=1.5, color=[0.5, 0, 0.5])
        else:
            ax3.plot(x_vals, evals, linewidth=1.5, color=[0.5, 0, 0.5])

        # Add threshold indicators
        if 'best_threshold' in results:
            best_t = results['best_threshold']
            if np.isscalar(best_t) and best_t > 0:
                ax3.axvline(x=best_t, color='r', linestyle='--', linewidth=1.5,
                           label=f'Thresh: {int(best_t)}')
            elif hasattr(best_t, '__len__'):
                mean_thresh = np.mean(best_t)
                ax3.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=1.5,
                           label=f'Mean: {mean_thresh:.1f}')

        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Eigenvalue')
        ax3.set_title('Basis Eigenvalues (sorted, used for ranking)')
        ax3.grid(True)

    elif 'signalvar' in results:
        # Show signal variance (SORTED - what was actually used for ranking)
        signal_vars = results['signalvar']  # Already sorted in descending order
        x_vals = np.arange(1, len(signal_vars) + 1)

        if use_log_x:
            ax3.semilogx(x_vals, signal_vars, linewidth=1.5, color='blue')
        else:
            ax3.plot(x_vals, signal_vars, linewidth=1.5, color='blue')

        # Add threshold indicators
        if 'best_threshold' in results:
            best_t = results['best_threshold']
            if np.isscalar(best_t) and best_t > 0:
                ax3.axvline(x=best_t, color='r', linestyle='--', linewidth=1.5,
                           label=f'Thresh: {int(best_t)}')
            elif hasattr(best_t, '__len__'):
                mean_thresh = np.mean(best_t)
                ax3.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=1.5,
                           label=f'Mean: {mean_thresh:.1f}')

        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Signal Variance')
        ax3.set_title('Signal Variance Spectrum (sorted, used for ranking)')
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'Ranking Info\nNot Available',
                ha='center', va='center', transform=ax3.transAxes)

    # =========================================================================
    # Plot 4: Signal vs Noise variance
    # =========================================================================
    ax4 = plt.subplot(4, 4, 4)
    if 'signalvar' in results and 'noisevar' in results:
        if not isinstance(results['signalvar'], (list, tuple)):
            # Global or averaged
            sv = results['signalvar']
            nv = results['noisevar']

            # Left y-axis for variance
            ax4_left = ax4
            ax4_left.set_ylabel('Variance', color='tab:blue')

            x_vals = np.arange(1, len(sv) + 1)
            if use_log_x:
                line1 = ax4_left.semilogx(x_vals, sv, '-', linewidth=1.5,
                                         color='blue', label='Signal var')
                line2 = ax4_left.semilogx(x_vals, nv, '-', linewidth=1.5,
                                         color=[1, 0.5, 0], label='Noise var')
            else:
                line1 = ax4_left.plot(sv, '-', linewidth=1.5, color='blue', label='Signal var')
                line2 = ax4_left.plot(nv, '-', linewidth=1.5, color=[1, 0.5, 0], label='Noise var')

            ax4_left.tick_params(axis='y', labelcolor='tab:blue')

            # Right y-axis for NCSNR
            ax4_right = ax4_left.twinx()
            ncsnr_trace = np.sqrt(sv) / np.sqrt(nv + np.finfo(float).eps)

            if use_log_x:
                line3 = ax4_right.semilogx(x_vals, ncsnr_trace, '-', linewidth=1.5,
                                          color='magenta', label='NCSNR')
            else:
                line3 = ax4_right.plot(ncsnr_trace, '-', linewidth=1.5, color='magenta', label='NCSNR')

            ax4_right.set_ylabel('NCSNR', color='magenta')
            ax4_right.tick_params(axis='y', labelcolor='magenta')

            # Add threshold (on left axis)
            if 'best_threshold' in results:
                best_t = results['best_threshold']
                if np.isscalar(best_t):
                    ax4_left.axvline(x=best_t, color='r', linestyle='--', linewidth=1)
                else:
                    ax4_left.axvline(x=np.mean(best_t), color='r', linestyle='--', linewidth=1)

            ax4_left.set_xlabel('Dimension')
            ax4_left.set_title('Signal and Noise Variance')

            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax4_left.legend(lines, labels, loc='best')
            ax4_left.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Per-Unit Variance\n(Averaged across units)',
                    ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'Variance Info\nNot Available',
                ha='center', va='center', transform=ax4.transAxes)

    # =========================================================================
    # Plot 5: Objective function
    # =========================================================================
    ax5 = plt.subplot(4, 4, 5)
    if 'objective' in results:
        obj = results['objective']
        x_obj = np.arange(len(obj))

        # Check if unit-specific objectives are available
        if 'unit_objectives' in results and results['unit_objectives']:
            # Unit-specific mode: plot all unit curves
            for u_obj in results['unit_objectives']:
                ax5.plot(np.arange(len(u_obj)), u_obj, linewidth=0.5,
                        color=[0.5, 0.5, 0.5], alpha=0.3)

            # Plot population average as thick line
            ax5.plot(x_obj, obj, linewidth=2, color=[0.3, 0.7, 0.3])

            # Mark each unit's chosen threshold
            if 'best_threshold' in results:
                best_t = results['best_threshold']
                if hasattr(best_t, '__len__'):
                    x_thresh = []
                    y_thresh = []
                    for u, u_obj in enumerate(results['unit_objectives']):
                        if u < len(best_t):
                            k_u = int(best_t[u])
                            if k_u >= 0 and k_u < len(u_obj):
                                x_thresh.append(k_u)
                                y_thresh.append(u_obj[k_u])
                    if x_thresh:
                        ax5.scatter(x_thresh, y_thresh, s=20, color=[1, 0.3, 0.3],
                                   alpha=0.6, zorder=5)

            ax5.set_title('Objective Function (unit-specific)')
        else:
            # Global mode: single curve
            ax5.plot(x_obj, obj, linewidth=1.5, color=[0.3, 0.7, 0.3])

            # Mark maximum
            max_idx = np.argmax(obj)
            ax5.plot(max_idx, obj[max_idx], 'ro', markersize=8, linewidth=2)

            ax5.set_title('Objective Function')

        # Set x-axis scale (log for high-dimensional data)
        if use_log_x:
            ax5.set_xscale('log')

        ax5.set_xlabel('Number of Dimensions')

        # Set ylabel based on criterion
        if criterion == 'variance':
            ax5.set_ylabel('Cumulative Variance')
        else:
            ax5.set_ylabel('Cumulative Signal - Noise/ntrials')

        ax5.grid(True)
    else:
        ax5.text(0.5, 0.5, 'Objective\nNot Available',
                ha='center', va='center', transform=ax5.transAxes)

    # =========================================================================
    # Plot 6-8: Raw, Denoised, Noise
    # =========================================================================

    # Plot 6: Raw trial-averaged data
    ax6 = plt.subplot(4, 4, 6)
    data_std = np.nanstd(trial_avg) if has_nans else np.std(trial_avg)
    clim_6 = 3 * data_std * np.array([-1, 1]) if data_std > 0 else np.array([-1, 1])
    im6 = ax6.imshow(trial_avg, vmin=clim_6[0], vmax=clim_6[1], cmap='RdBu_r', aspect='auto', interpolation='none')
    plt.colorbar(im6, ax=ax6)
    title_6 = 'Input Data (trial-averaged, with NaNs)' if has_nans else 'Input Data (trial-averaged)'
    ax6.set_title(title_6)
    ax6.set_xlabel('Conditions')
    ax6.set_ylabel('Units')

    # Plot 7: Denoised data
    ax7 = plt.subplot(4, 4, 7)
    data_std = np.nanstd(denoised) if has_nans else np.std(denoised)
    clim_7 = 3 * data_std * np.array([-1, 1]) if data_std > 0 else np.array([-1, 1])
    im7 = ax7.imshow(denoised, vmin=clim_7[0], vmax=clim_7[1], cmap='RdBu_r', aspect='auto', interpolation='none')
    plt.colorbar(im7, ax=ax7)
    ax7.set_title('PSN Denoised Data')
    ax7.set_xlabel('Conditions')
    ax7.set_ylabel('Units')

    # Plot 8: Noise (residual)
    ax8 = plt.subplot(4, 4, 8)
    data_std = np.nanstd(noise) if has_nans else np.std(noise)
    clim_8 = 3 * data_std * np.array([-1, 1]) if data_std > 0 else np.array([-1, 1])
    im8 = ax8.imshow(noise, vmin=clim_8[0], vmax=clim_8[1], cmap='RdBu_r', aspect='auto', interpolation='none')
    plt.colorbar(im8, ax=ax8)
    title_8 = 'Residual (Noise, with NaNs)' if has_nans else 'Residual (Noise)'
    ax8.set_title(title_8)
    ax8.set_xlabel('Conditions')
    ax8.set_ylabel('Units')

    # =========================================================================
    # Plot 9: Denoiser matrix
    # =========================================================================
    ax9 = plt.subplot(4, 4, 9)
    denoiser = results['denoiser']
    data_std = np.nanstd(denoiser) if has_nans else np.std(denoiser)
    clim_9 = 3 * data_std * np.array([-1, 1]) if data_std > 0 else np.array([-1, 1])
    im9 = ax9.imshow(denoiser, vmin=clim_9[0], vmax=clim_9[1], cmap='RdBu_r', aspect='equal', interpolation='none')
    plt.colorbar(im9, ax=ax9)
    ax9.set_title('Denoiser Matrix')
    ax9.set_xlabel('Units')
    ax9.set_ylabel('Units')

    # =========================================================================
    # Plot 10-11: Traces
    # =========================================================================
    # Color conditions by mean response (handle NaNs)
    cond_means = np.nanmean(trial_avg, axis=0)
    sorted_indices = np.argsort(cond_means)
    colors = plt.cm.jet(np.linspace(0, 1, nconds))
    trace_colors = np.zeros((nconds, 3))
    for rank in range(nconds):
        cond_idx = sorted_indices[rank]
        trace_colors[cond_idx, :] = colors[rank, :3]  # Use only RGB, not alpha

    # Trial-averaged traces
    ax10 = plt.subplot(4, 4, 10)
    x_units = np.arange(nunits)
    for c in range(nconds):
        ax10.plot(x_units, trial_avg[:, c], color=trace_colors[c, :], linewidth=0.5)
    ax10.set_xlabel('Units')
    ax10.set_ylabel('Activity')
    ax10.set_title('Trial-Averaged Traces')
    ax10.grid(True)
    ax10.set_xlim([x_units[0], x_units[-1]])

    # Denoised traces
    ax11 = plt.subplot(4, 4, 11)
    for c in range(nconds):
        ax11.plot(x_units, denoised[:, c], color=trace_colors[c, :], linewidth=0.5)
    ax11.set_xlabel('Units')
    ax11.set_ylabel('Activity')
    ax11.set_title('PSN Denoised Traces')
    ax11.grid(True)
    ax11.set_xlim([x_units[0], x_units[-1]])

    # Match y-limits (handle NaNs)
    all_trace_data = np.concatenate([trial_avg.ravel(), denoised.ravel()])
    y_min = np.nanmin(all_trace_data) if has_nans else np.min(all_trace_data)
    y_max = np.nanmax(all_trace_data) if has_nans else np.max(all_trace_data)
    y_range = y_max - y_min
    y_margin = y_range * 0.05

    ax10.set_ylim([y_min - y_margin, y_max + y_margin])
    ax11.set_ylim([y_min - y_margin, y_max + y_margin])

    # =========================================================================
    # Plot 12: Split-half reliability
    # =========================================================================
    ax12 = plt.subplot(4, 4, 12)

    # Split trials
    half_idx = ntrials // 2
    data_A = data[:, :, :half_idx]
    data_B = data[:, :, half_idx:]

    # Trial averages (use nanmean to handle NaNs)
    tavg_A = np.nanmean(data_A, axis=2) if has_nans else np.mean(data_A, axis=2)
    tavg_B = np.nanmean(data_B, axis=2) if has_nans else np.mean(data_B, axis=2)

    # Denoise both splits
    unit_means = results['unit_means']

    # Handle symmetric vs non-symmetric denoiser
    if threshold_method == 'global':
        # Symmetric: standard multiplication
        dn_A = denoiser @ (tavg_A - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]
        dn_B = denoiser @ (tavg_B - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]
    else:
        # Non-symmetric: transpose multiplication
        dn_A = denoiser.T @ (tavg_A - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]
        dn_B = denoiser.T @ (tavg_B - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]

    # Compute correlations
    corr_tavg = np.zeros(nunits)
    corr_cross = np.zeros(nunits)
    corr_dn = np.zeros(nunits)

    for u in range(nunits):
        if np.std(tavg_A[u, :]) > 0 and np.std(tavg_B[u, :]) > 0:
            corr_tavg[u] = np.corrcoef(tavg_A[u, :], tavg_B[u, :])[0, 1]
        else:
            corr_tavg[u] = np.nan

        # Cross-method (average both directions)
        if (np.std(tavg_A[u, :]) > 0 and np.std(dn_B[u, :]) > 0 and
            np.std(dn_A[u, :]) > 0 and np.std(tavg_B[u, :]) > 0):
            corr_AB = np.corrcoef(tavg_A[u, :], dn_B[u, :])[0, 1]
            corr_BA = np.corrcoef(dn_A[u, :], tavg_B[u, :])[0, 1]
            corr_cross[u] = (corr_AB + corr_BA) / 2
        else:
            corr_cross[u] = np.nan

        if np.std(dn_A[u, :]) > 0 and np.std(dn_B[u, :]) > 0:
            corr_dn[u] = np.corrcoef(dn_A[u, :], dn_B[u, :])[0, 1]
        else:
            corr_dn[u] = np.nan

    # Plot
    x_positions = np.array([1, 2, 3])
    labels = ['TAvg vs TAvg', 'TAvg vs Denoised', 'Denoised vs Denoised']

    # Add jitter
    x_jitter = (np.random.rand(nunits) - 0.5) * 0.16

    # Connecting lines
    for u in range(nunits):
        values = [corr_tavg[u], corr_cross[u], corr_dn[u]]
        if not np.any(np.isnan(values)):
            ax12.plot(x_positions + x_jitter[u], values,
                     color=[0.5, 0.5, 0.5], linewidth=0.3)

    # Scatter points
    ax12.scatter(x_positions[0] + x_jitter, corr_tavg, s=15, color='blue',
                alpha=0.4, zorder=2)
    ax12.scatter(x_positions[1] + x_jitter, corr_cross, s=15, color=[1, 0.84, 0],
                alpha=0.4, zorder=2)
    ax12.scatter(x_positions[2] + x_jitter, corr_dn, s=15, color=[0.5, 0.8, 0.3],
                alpha=0.4, zorder=2)

    # Means
    mean_tavg = np.nanmean(corr_tavg)
    mean_cross = np.nanmean(corr_cross)
    mean_dn = np.nanmean(corr_dn)

    ax12.scatter(x_positions[0], mean_tavg, s=100, color='blue',
                edgecolors='white', linewidths=2, zorder=3)
    ax12.scatter(x_positions[1], mean_cross, s=100, color=[1, 0.84, 0],
                edgecolors='white', linewidths=2, zorder=3)
    ax12.scatter(x_positions[2], mean_dn, s=100, color=[0.2, 0.6, 0.2],
                edgecolors='white', linewidths=2, zorder=3)

    # Labels
    y_offset = 0.08
    ax12.text(x_positions[0], mean_tavg + y_offset, f'{mean_tavg:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax12.text(x_positions[1], mean_cross + y_offset, f'{mean_cross:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax12.text(x_positions[2], mean_dn + y_offset, f'{mean_dn:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax12.set_xticks(x_positions)
    ax12.set_xticklabels(labels, rotation=0, fontsize=7)
    ax12.set_ylabel('Pearson r')

    if has_nans:
        # Count actual valid trials per split
        valid_A = np.sum(~np.any(np.isnan(data_A), axis=0), axis=1)
        valid_B = np.sum(~np.any(np.isnan(data_B), axis=0), axis=1)
        avg_valid_A = np.mean(valid_A[valid_A > 0])
        avg_valid_B = np.mean(valid_B[valid_B > 0])
        ax12.set_title(f'Split-Half Reliability\n({avg_valid_A:.1f} vs {avg_valid_B:.1f} avg trials)')
    else:
        ax12.set_title(f'Split-Half Reliability\n({data_A.shape[2]} vs {data_B.shape[2]} trials)')

    ax12.grid(True)
    ax12.set_xlim([0.5, 3.5])
    ax12.axhline(y=0, color='k', linewidth=1)

    # Set y-limits
    all_corr = np.concatenate([corr_tavg, corr_cross, corr_dn])
    valid_corr = all_corr[~np.isnan(all_corr)]
    if len(valid_corr) > 0:
        y_min_c = np.min(valid_corr)
        y_max_c = np.max(valid_corr)
        y_range_c = y_max_c - y_min_c
        y_pad = max(0.1, y_range_c * 0.15)
        ax12.set_ylim([y_min_c - y_pad, y_max_c + y_pad])
    else:
        ax12.set_ylim([-1, 1])

    # =========================================================================
    # Plot 13-16: Signal/Noise Diagnostics
    # =========================================================================

    # Extract signal/noise variance data
    if 'svnv_before' in results and 'svnv_after' in results:
        sv_before = results['svnv_before'][:, 0]
        nv_before = results['svnv_before'][:, 1]
        sv_after = results['svnv_after'][:, 0]
        nv_after = results['svnv_after'][:, 1]

        # Compute noise-corrected SNR (ncsnr)
        ncsnr_before = np.sqrt(sv_before) / np.sqrt(nv_before + np.finfo(float).eps)
        ncsnr_after = np.sqrt(sv_after) / np.sqrt(nv_after + np.finfo(float).eps)

        # Compute noise ceiling percentage (use ntrials_avg for NaN data)
        noiseceiling_before = 100 * (ncsnr_before**2 / (ncsnr_before**2 + 1/ntrials_avg))
        noiseceiling_after = 100 * (ncsnr_after**2 / (ncsnr_after**2 + 1/ntrials_avg))

        # Define x positions
        x_before = 1
        x_after = 2
        x_jitter_diag = (np.random.rand(nunits) - 0.5) * 0.1

        # Plot 13: Signal Variance
        ax13 = plt.subplot(4, 4, 13)
        for u in range(nunits):
            ax13.plot([x_before, x_after] + x_jitter_diag[u],
                     [sv_before[u], sv_after[u]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax13.scatter(x_before + x_jitter_diag, sv_before, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax13.scatter(x_after + x_jitter_diag, sv_after, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        mean_sv_before = np.mean(sv_before)
        mean_sv_after = np.mean(sv_after)
        ax13.scatter(x_before, mean_sv_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax13.scatter(x_after, mean_sv_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically
        y_range_sv = np.max([sv_before.max(), sv_after.max()]) - np.min([sv_before.min(), sv_after.min()])
        y_offset_sv = y_range_sv * 0.08

        ax13.text(x_before, mean_sv_before + y_offset_sv, f'{mean_sv_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax13.text(x_after, mean_sv_after + y_offset_sv, f'{mean_sv_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax13.set_xlim([0.5, 2.5])
        ax13.set_xticks([1, 2])
        ax13.set_xticklabels(['Before', 'After'])
        ax13.set_ylabel('Signal Variance')
        ax13.set_title('Signal Variance')
        ax13.grid(True)

        # Plot 14: Noise Variance
        ax14 = plt.subplot(4, 4, 14)
        for u in range(nunits):
            ax14.plot([x_before, x_after] + x_jitter_diag[u],
                     [nv_before[u], nv_after[u]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax14.scatter(x_before + x_jitter_diag, nv_before, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax14.scatter(x_after + x_jitter_diag, nv_after, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        mean_nv_before = np.mean(nv_before)
        mean_nv_after = np.mean(nv_after)
        ax14.scatter(x_before, mean_nv_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax14.scatter(x_after, mean_nv_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically
        y_range_nv = np.max([nv_before.max(), nv_after.max()]) - np.min([nv_before.min(), nv_after.min()])
        y_offset_nv = y_range_nv * 0.08

        ax14.text(x_before, mean_nv_before + y_offset_nv, f'{mean_nv_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax14.text(x_after, mean_nv_after + y_offset_nv, f'{mean_nv_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax14.set_xlim([0.5, 2.5])
        ax14.set_xticks([1, 2])
        ax14.set_xticklabels(['Before', 'After'])
        ax14.set_ylabel('Noise Variance')
        ax14.set_title('Noise Variance')
        ax14.grid(True)

        # Set unified ylims for both signal and noise variance plots
        all_variance_vals = np.concatenate([sv_before, sv_after, nv_before, nv_after])
        y_max_unified = np.max(all_variance_vals)
        y_pad_unified = y_max_unified * 0.05  # 5% padding at bottom
        unified_ylim = [-y_pad_unified, y_max_unified + y_max_unified * 0.15]

        ax13.set_ylim(unified_ylim)
        ax13.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax14.set_ylim(unified_ylim)
        ax14.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # Plot 15: NCSNR
        ax15 = plt.subplot(4, 4, 15)
        for u in range(nunits):
            ax15.plot([x_before, x_after] + x_jitter_diag[u],
                     [ncsnr_before[u], ncsnr_after[u]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax15.scatter(x_before + x_jitter_diag, ncsnr_before, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax15.scatter(x_after + x_jitter_diag, ncsnr_after, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        mean_ncsnr_before = np.mean(ncsnr_before)
        mean_ncsnr_after = np.mean(ncsnr_after)
        ax15.scatter(x_before, mean_ncsnr_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax15.scatter(x_after, mean_ncsnr_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically
        y_range_ncsnr = np.max([ncsnr_before.max(), ncsnr_after.max()]) - np.min([ncsnr_before.min(), ncsnr_after.min()])
        y_offset_ncsnr = y_range_ncsnr * 0.08

        ax15.text(x_before, mean_ncsnr_before + y_offset_ncsnr, f'{mean_ncsnr_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax15.text(x_after, mean_ncsnr_after + y_offset_ncsnr, f'{mean_ncsnr_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Set ylims with padding
        all_ncsnr_vals = np.concatenate([ncsnr_before, ncsnr_after])
        y_max_ncsnr = np.max(all_ncsnr_vals)
        y_pad_ncsnr_bottom = y_max_ncsnr * 0.05  # 5% padding at bottom
        y_pad_ncsnr_top = y_max_ncsnr * 0.15  # 15% padding at top
        ax15.set_ylim([-y_pad_ncsnr_bottom, y_max_ncsnr + y_pad_ncsnr_top])
        ax15.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax15.set_xlim([0.5, 2.5])
        ax15.set_xticks([1, 2])
        ax15.set_xticklabels(['Before', 'After'])
        ax15.set_ylabel('NCSNR')
        ax15.set_title('Noise Ceiling SNR (NCSNR)')
        ax15.grid(True)

        # Plot 16: Noise Ceiling %
        ax16 = plt.subplot(4, 4, 16)
        for u in range(nunits):
            ax16.plot([x_before, x_after] + x_jitter_diag[u],
                     [noiseceiling_before[u], noiseceiling_after[u]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax16.scatter(x_before + x_jitter_diag, noiseceiling_before, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax16.scatter(x_after + x_jitter_diag, noiseceiling_after, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        mean_nc_before = np.mean(noiseceiling_before)
        mean_nc_after = np.mean(noiseceiling_after)
        ax16.scatter(x_before, mean_nc_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax16.scatter(x_after, mean_nc_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Fixed y_offset for noise ceiling (percentage scale 0-100)
        y_offset_nc = 100 * 0.08

        ax16.text(x_before, mean_nc_before + y_offset_nc, f'{mean_nc_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax16.text(x_after, mean_nc_after + y_offset_nc, f'{mean_nc_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax16.set_xlim([0.5, 2.5])
        ax16.set_ylim([-5, 100])  # Add negative padding to make yline at 0 visible
        ax16.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax16.set_xticks([1, 2])
        ax16.set_xticklabels(['Before', 'After'])
        ax16.set_ylabel('Noise Ceiling (%)')

        if has_nans:
            ax16.set_title(f'Noise Ceiling Percentage ({ntrials_avg:.1f} avg trials)')
        else:
            ax16.set_title(f'Noise Ceiling Percentage ({ntrials} trials)')

        ax16.grid(True)

    plt.tight_layout()
    plt.show()
