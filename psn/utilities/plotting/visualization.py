"""PSN visualization - matches MATLAB visualization.m exactly"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


def cmapsign4(n=256):
    """
    Return a cyan-blue-black-red-yellow colormap.

    This colormap is symmetric around black (center), going from
    cyan-white through cyan and blue to black, then from black
    through red and yellow to yellow-white.

    This is useful for visualizing data that has both positive and
    negative values, with zero mapped to black.

    Parameters
    ----------
    n : int, optional
        Number of colors in the colormap. Default: 256.

    Returns
    -------
    cmap : LinearSegmentedColormap
        A matplotlib colormap object.
    """
    colors = [
        (0.8, 1, 1),    # cyan-white
        (0, 1, 1),      # cyan
        (0, 0, 1),      # blue
        (0, 0, 0),      # black (center)
        (1, 0, 0),      # red
        (1, 1, 0),      # yellow
        (1, 1, 0.8),    # yellow-white
    ]
    cmap = LinearSegmentedColormap.from_list('cmapsign4', colors, N=n)
    return cmap


def redblue(n=256):
    """
    Return a red-blue diverging colormap.

    This colormap transitions from blue through white to red.
    Useful for visualizing symmetric data centered at zero
    (e.g., correlation matrices, covariance matrices, residuals).

    Parameters
    ----------
    n : int, optional
        Number of colors in the colormap. Default: 256.

    Returns
    -------
    cmap : LinearSegmentedColormap
        A matplotlib colormap object.
    """
    mid = n // 2

    # Build color array
    colors_list = []

    # Blue to white (first half)
    for i in range(mid):
        t = i / (mid - 1) if mid > 1 else 1
        colors_list.append((t, t, 1.0))  # R, G increase; B stays 1

    # White to red (second half)
    for i in range(n - mid):
        t = i / (n - mid - 1) if (n - mid) > 1 else 1
        colors_list.append((1.0, 1.0 - t, 1.0 - t))  # R stays 1; G, B decrease

    cmap = LinearSegmentedColormap.from_list('redblue', colors_list, N=n)
    return cmap


def plot_diagnostic_figures(data, results, test_data=None, figurepath=None, cmap=None,
                            split_half_metric='correlation'):
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
    figurepath : str, optional
        If specified, save figure to this path before displaying.
        The figure is saved at 150 dpi with tight bounding box.
    cmap : colormap, optional
        Colormap for input data, denoised data, and residual plots.
        Default: cmapsign4()
    split_half_metric : str, optional
        Metric for the split-half reliability plot.
        'correlation' (default) — Pearson r per unit.
        'mse' — mean squared error per unit.
    """
    # Set default colormap for data plots
    if cmap is None:
        cmap = cmapsign4()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Increase font sizes globally for this figure
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })

    # Create a large figure with custom grid layout
    # Top row: 5 subplots (cSb, cNb, basis dims (half), eigenvalues (half), sig/noise var)
    # Rows 2-4: standard 4x4 grid
    fig = plt.figure(figsize=(24, 15))

    # Create GridSpec: 4 rows, 8 columns (to allow half-width subplots)
    # Increase spacing to prevent overlap
    # Use width_ratios to make first 2 columns slightly narrower (for objective plot with twin y-axis)
    gs = GridSpec(4, 8, figure=fig, hspace=0.45, wspace=0.6,
                  left=0.05, right=0.95, top=0.93, bottom=0.05,
                  width_ratios=[0.9, 0.9, 1, 1, 1, 1, 1, 1])

    # Extract data dimensions
    nunits, nconds, ntrials = data.shape

    # Subsampling for large datasets (>500 units or >500 conditions)
    # Randomly select 100 units/conditions for certain plots, but keep full population for means
    n_subsample = 100

    # Unit subsampling
    subsample_units = nunits > 500
    if subsample_units:
        np.random.seed(42)  # For reproducibility
        subsample_idx = np.sort(np.random.choice(nunits, n_subsample, replace=False))
    else:
        subsample_idx = np.arange(nunits)

    # Condition subsampling (for trace plots)
    subsample_conds = nconds > 500
    if subsample_conds:
        np.random.seed(43)  # Different seed for conditions
        subsample_cond_idx = np.sort(np.random.choice(nconds, n_subsample, replace=False))
    else:
        subsample_cond_idx = np.arange(nconds)

    # Build suffix strings
    if subsample_units and subsample_conds:
        subsample_suffix = f'\n(randomly subsampling {n_subsample} units, {n_subsample} conditions)'
        subsample_suffix_units = f'\n(randomly subsampling {n_subsample} units)'
        subsample_suffix_conds = f'\n(randomly subsampling {n_subsample} conditions)'
        subsample_suffix_traces = f'\n(randomly subsampling {n_subsample} units, {n_subsample} conditions)'
    elif subsample_units:
        subsample_suffix = f'\n(randomly subsampling {n_subsample} units)'
        subsample_suffix_units = subsample_suffix
        subsample_suffix_conds = ''
        subsample_suffix_traces = subsample_suffix
    elif subsample_conds:
        subsample_suffix = ''
        subsample_suffix_units = ''
        subsample_suffix_conds = f'\n(randomly subsampling {n_subsample} conditions)'
        subsample_suffix_traces = subsample_suffix_conds
    else:
        subsample_suffix = ''
        subsample_suffix_units = ''
        subsample_suffix_conds = ''
        subsample_suffix_traces = ''

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
    if basis_desc == 'wiener':
        # Full-rank Wiener bypasses criterion/threshold — show simplified title
        data_str = (f'{nunits} units × {nconds} conditions × {ntrials} max trials (avg {ntrials_avg:.1f})'
                    if has_nans else f'{nunits} units × {nconds} conditions × {ntrials} trials')
        title_text = f'Data: {data_str}  |  Full-Rank Matrix Wiener Filter'
    elif has_nans:
        title_text = f'Data: {nunits} units × {nconds} conditions × {ntrials} max trials (avg {ntrials_avg:.1f})  |  Basis: {basis_desc}  |  Criterion: {criterion}  |  Method: {threshold_method}'
    else:
        title_text = f'Data: {nunits} units × {nconds} conditions × {ntrials} trials  |  Basis: {basis_desc}  |  Criterion: {criterion}  |  Method: {threshold_method}'

    # Add threshold info if conservative mode or variance criterion is used
    threshold_info = []
    if 'allowable_thresholds' in opt and opt['allowable_thresholds'] is not None:
        allowable = opt['allowable_thresholds']
        if hasattr(allowable, '__len__'):
            if len(allowable) == 1:
                threshold_info.append(f"Forced threshold: {int(allowable[0])}")
            else:
                threshold_info.append(f"Allowable thresholds: {list(allowable)}")
        else:
            threshold_info.append(f"Forced threshold: {int(allowable)}")
    if criterion in ['variance', 'variance_eigenvalues']:
        vt = opt.get('variance_threshold', 0.99)
        threshold_info.append(f"Variance threshold: {vt}")

    if threshold_info:
        title_text += '  |  ' + ', '.join(threshold_info)

    plt.suptitle(title_text, fontsize=14, fontweight='bold')

    # Get trial-averaged and denoised data (use nanmean for NaN data)
    if has_nans:
        trial_avg = np.nanmean(data, axis=2)
    else:
        trial_avg = np.mean(data, axis=2)

    denoised = results['denoiseddata']
    noise = trial_avg - denoised

    # =========================================================================
    # Plot 1: Basis source matrix (signal covariance or basis-specific)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0:2])  # First 2 columns of row 0
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

        # Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
        if has_nans:
            data_absmax = np.nanpercentile(np.abs(plot_matrix_1), 99)
        else:
            data_absmax = np.percentile(np.abs(plot_matrix_1), 99)

        if data_absmax > 0:
            clim_1 = [-data_absmax, data_absmax]
        else:
            clim_1 = [-1, 1]

        im1 = ax1.imshow(plot_matrix_1, vmin=clim_1[0], vmax=clim_1[1], cmap=redblue(), aspect='equal')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(plot_title)
        ax1.set_xlabel('Units')
        ax1.set_ylabel('Units')
    else:
        ax1.text(0.5, 0.5, 'Covariance\nNot Available',
                ha='center', va='center', transform=ax1.transAxes)

    # =========================================================================
    # Plot 2: Noise Covariance (cNb)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2:4])  # Columns 2-3 of row 0
    if 'gsn_result' in results and 'cNb' in results['gsn_result']:
        cNb = results['gsn_result']['cNb']

        # Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
        if has_nans:
            data_absmax_cNb = np.nanpercentile(np.abs(cNb), 99)
        else:
            data_absmax_cNb = np.percentile(np.abs(cNb), 99)

        if data_absmax_cNb > 0:
            clim_cNb = [-data_absmax_cNb, data_absmax_cNb]
        else:
            clim_cNb = [-1, 1]

        im2 = ax2.imshow(cNb, vmin=clim_cNb[0], vmax=clim_cNb[1], cmap=redblue(), aspect='equal')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Noise Covariance (cNb)')
        ax2.set_xlabel('Units')
        ax2.set_ylabel('Units')
    else:
        ax2.text(0.5, 0.5, 'Noise Covariance\nNot Available',
                ha='center', va='center', transform=ax2.transAxes)

    # =========================================================================
    # Plot 3: Top 5 PCs as vertical line plots (half width)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 4:5])  # Column 4 of row 0 (half width)
    if 'fullbasis' in results:
        num_pcs = min(5, results['fullbasis'].shape[1])

        # Normalize each PC for visualization (0-indexed)
        y_units = np.arange(nunits)
        colors = plt.cm.tab10(np.linspace(0, 1, num_pcs))

        # Find max absolute loading across top 5 PCs for scaling
        max_loading = np.max(np.abs(results['fullbasis'][:, :num_pcs]))
        if max_loading > 0:
            scale_factor = 0.4 / max_loading  # Scale to fit within 0.4 x-units
        else:
            scale_factor = 1

        for pc in range(num_pcs):
            # Center each PC at position pc (0-indexed), with loadings as horizontal deviations
            x_vals = pc + results['fullbasis'][:, pc] * scale_factor
            ax3.plot(x_vals, y_units, linewidth=1.5, color=colors[pc])

            # Add vertical reference line at center
            ax3.axvline(x=pc, color='k', linestyle='--', linewidth=0.5)

        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Units')
        ax3.set_title('Top 5 Basis Dims')
        ax3.set_xlim([-0.5, num_pcs - 0.5])
        ax3.set_ylim([0, nunits - 1])
        ax3.invert_yaxis()  # Flip y-axis to match heatmaps
        ax3.set_xticks(range(num_pcs))
        # Add eigenvalue labels under PC indices
        if 'basis_eigenvalues' in results and results['basis_eigenvalues'] is not None and len(results['basis_eigenvalues']) >= num_pcs:
            evals = results['basis_eigenvalues']
            tick_labels = [f'{pc}\nλ={evals[pc]:.2f}' for pc in range(num_pcs)]
            ax3.set_xticklabels(tick_labels, fontsize=7)
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'Basis\nNot Available',
                ha='center', va='center', transform=ax3.transAxes)

    # =========================================================================
    # Plot 4: Global dimension ranking (eigenvalues or signal variance) - half width
    # =========================================================================
    ax4 = fig.add_subplot(gs[0, 5:6])  # Column 5 of row 0 (half width)

    # Determine if we should use log scale for x-axis (for large datasets)
    use_logscale = nunits > 50

    # Check if eigenvalues were used for ranking (and are available)
    use_eigenvalues = (basis_ordering == 'eigenvalues' and
                      'basis_eigenvalues' in results and
                      results['basis_eigenvalues'] is not None and
                      len(results['basis_eigenvalues']) > 0)

    # Check if prediction ordering was used
    use_prediction_ordering = (basis_ordering == 'prediction')

    # Flag for full-rank Wiener (changes how threshold lines are labeled)
    is_fullrank_wiener = (basis_desc == 'wiener')

    if use_eigenvalues:
        # Show eigenvalues (SORTED - what was actually used for ranking)
        evals = results['basis_eigenvalues']  # Already sorted in descending order
        # For log scale, shift x by 1 so dimensions go 1, 2, 3... (avoids log(0))
        if use_logscale:
            x_vals = np.arange(len(evals)) + 1
        else:
            x_vals = np.arange(len(evals))  # 0-indexed
        ax4.plot(x_vals, evals, linewidth=1.5, color=[0.5, 0, 0.5], label='$\\lambda_k(\\Sigma_S)$')

        # Add threshold indicators (only if threshold > 0)
        if 'best_threshold' in results:
            best_t = results['best_threshold']
            if np.isscalar(best_t) and best_t > 0:
                if is_fullrank_wiener:
                    thresh_label = f'$\\mathrm{{tr}}(D) = {best_t:.1f}$'
                else:
                    thresh_label = f'Threshold $K = {int(best_t)}$'
                ax4.axvline(x=best_t, color='r', linestyle='--', linewidth=2, label=thresh_label)
            elif hasattr(best_t, '__len__') and np.mean(best_t) > 0:
                mean_thresh = np.mean(best_t)
                thresh_label = f'Mean threshold $= {mean_thresh:.1f}$'
                ax4.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=2, label=thresh_label)

        ax4.set_xlabel('Dimension $k$ (signal eigenbasis)')
        ax4.set_ylabel('Eigenvalue')
        if is_fullrank_wiener:
            ax4.set_title('$\\Sigma_S$ Eigenvalues (signal basis)')
        else:
            ax4.set_title('Basis Eigenvalues')
        ax4.legend(loc='best', fontsize=7)
        ax4.grid(True)
        if use_logscale:
            ax4.set_xscale('log')
            ax4.set_xlim([0.8, len(evals) + 1])
        else:
            ax4.set_xlim([-0.5, len(evals) - 0.5])

    elif use_prediction_ordering and 'signalvar' in results and 'noisevar' in results:
        # Show prediction ordering criterion (signal - noise/ntrials) and signal variance
        signal_vars = results['signalvar']
        noise_vars = results['noisevar']
        prediction_obj = signal_vars - noise_vars / ntrials_avg

        if use_logscale:
            x_vals = np.arange(len(signal_vars)) + 1
        else:
            x_vals = np.arange(len(signal_vars))

        # Plot both signal variance and prediction objective
        ax4.plot(x_vals, signal_vars, linewidth=1.5, color='blue', label='Signal Var')
        ax4.plot(x_vals, prediction_obj, linewidth=1.5, color=[0.5, 0, 0.5], label='SigVar - NoiseVar/ntrials')

        # Add threshold indicators (only if threshold > 0)
        if 'best_threshold' in results:
            best_t = results['best_threshold']
            if np.isscalar(best_t) and best_t > 0:
                ax4.axvline(x=best_t, color='r', linestyle='--', linewidth=2)
                ylims = ax4.get_ylim()
                y_pos = ylims[0] + 0.7 * (ylims[1] - ylims[0])
                ax4.text(best_t * 1.05 if use_logscale else best_t + 0.5, y_pos, f'Threshold = {int(best_t)}',
                        color='r', fontsize=9, rotation=90,
                        ha='left', va='top')
            elif hasattr(best_t, '__len__') and np.mean(best_t) > 0:
                mean_thresh = np.mean(best_t)
                ax4.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=2)
                ylims = ax4.get_ylim()
                y_pos = ylims[0] + 0.7 * (ylims[1] - ylims[0])
                ax4.text(mean_thresh * 1.05 if use_logscale else mean_thresh + 0.5, y_pos, f'Mean Threshold = {mean_thresh:.1f}',
                        color='r', fontsize=9, rotation=90,
                        ha='left', va='top')

        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Variance')
        ax4.set_title('Ordering Criterion')
        ax4.legend(loc='best', fontsize=7)
        ax4.grid(True)
        if use_logscale:
            ax4.set_xscale('log')
            ax4.set_xlim([0.8, len(signal_vars) + 1])
        else:
            ax4.set_xlim([-0.5, len(signal_vars) - 0.5])

    elif 'signalvar' in results:
        # Show signal variance (SORTED - what was actually used for ranking)
        signal_vars = results['signalvar']  # Already sorted in descending order
        # For log scale, shift x by 1 so dimensions go 1, 2, 3... (avoids log(0))
        if use_logscale:
            x_vals = np.arange(len(signal_vars)) + 1
        else:
            x_vals = np.arange(len(signal_vars))  # 0-indexed
        ax4.plot(x_vals, signal_vars, linewidth=1.5, color='blue')

        # Add threshold indicators (only if threshold > 0)
        if 'best_threshold' in results:
            best_t = results['best_threshold']
            if np.isscalar(best_t) and best_t > 0:
                ax4.axvline(x=best_t, color='r', linestyle='--', linewidth=2)
                # Add rotated text annotation (top of text on right side of line)
                ylims = ax4.get_ylim()
                y_pos = ylims[0] + 0.7 * (ylims[1] - ylims[0])
                ax4.text(best_t * 1.05 if use_logscale else best_t + 0.5, y_pos, f'Threshold = {int(best_t)}',
                        color='r', fontsize=9, rotation=90,
                        ha='left', va='top')
            elif hasattr(best_t, '__len__') and np.mean(best_t) > 0:
                mean_thresh = np.mean(best_t)
                ax4.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=2)
                # Add rotated text annotation (top of text on right side of line)
                ylims = ax4.get_ylim()
                y_pos = ylims[0] + 0.7 * (ylims[1] - ylims[0])
                ax4.text(mean_thresh * 1.05 if use_logscale else mean_thresh + 0.5, y_pos, f'Mean Threshold = {mean_thresh:.1f}',
                        color='r', fontsize=9, rotation=90,
                        ha='left', va='top')

        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Signal Var')
        ax4.set_title('Signal Variance')
        ax4.grid(True)
        if use_logscale:
            ax4.set_xscale('log')
            ax4.set_xlim([0.8, len(signal_vars) + 1])
        else:
            ax4.set_xlim([-0.5, len(signal_vars) - 0.5])
    else:
        ax4.text(0.5, 0.5, 'Ranking Info\nNot Available',
                ha='center', va='center', transform=ax4.transAxes)

    # =========================================================================
    # Plot 5: Signal vs Noise variance
    # =========================================================================
    ax5 = fig.add_subplot(gs[0, 6:8])  # Columns 6-7 of row 0
    if 'signalvar' in results and 'noisevar' in results:
        if not isinstance(results['signalvar'], (list, tuple)):
            # Global or averaged
            sv = results['signalvar']
            nv = results['noisevar']

            # Left y-axis for variance
            ax5_left = ax5
            ax5_left.set_ylabel('Variance', color='tab:blue')

            # For log scale, shift x by 1 so dimensions go 1, 2, 3... (avoids log(0))
            if use_logscale:
                x_vals = np.arange(len(sv)) + 1
            else:
                x_vals = np.arange(len(sv))  # 0-indexed
            line1 = ax5_left.plot(x_vals, sv, '-', linewidth=1.5, color='blue', label='Signal var')
            line2 = ax5_left.plot(x_vals, nv, '-', linewidth=1.5, color=[1, 0.5, 0], label='Noise var')
            line2b = ax5_left.plot(x_vals, nv / ntrials_avg, '-', linewidth=1.5, color=[1, 0.85, 0.6], label=f'Noise var / {ntrials_avg:.1f} trials')

            ax5_left.tick_params(axis='y', labelcolor='tab:blue')

            # Right y-axis for NCSNR
            ax5_right = ax5_left.twinx()
            ncsnr_trace = np.sqrt(sv) / np.sqrt(nv + np.finfo(float).eps)
            line3 = ax5_right.plot(x_vals, ncsnr_trace, '-', linewidth=1.5, color='magenta', label='NCSNR')

            ax5_right.set_ylabel('NCSNR', color='magenta')
            ax5_right.tick_params(axis='y', labelcolor='magenta')

            # Add threshold (on left axis, only if > 0)
            line_thresh = None
            if 'best_threshold' in results:
                best_t = results['best_threshold']
                if np.isscalar(best_t) and best_t > 0:
                    if is_fullrank_wiener:
                        thresh_label = f'$\\mathrm{{tr}}(D) = {best_t:.1f}$  where $D = \\Sigma_S(\\Sigma_S + \\Sigma_N/t)^{{-1}}$'
                    else:
                        thresh_label = f'Threshold $K = {int(best_t)}$'
                    line_thresh = ax5_left.axvline(x=best_t, color='r', linestyle='--', linewidth=2, label=thresh_label)
                elif hasattr(best_t, '__len__') and np.mean(best_t) > 0:
                    mean_thresh = np.mean(best_t)
                    thresh_label = f'Mean threshold $= {mean_thresh:.1f}$'
                    line_thresh = ax5_left.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=2, label=thresh_label)

            ax5_left.set_xlabel('Dimension $k$ (signal eigenbasis)' if is_fullrank_wiener else 'Dimension')
            ax5_left.set_title('Signal and Noise Variance')

            # Combine legends
            lines = line1 + line2 + line2b + line3
            if line_thresh is not None:
                lines = lines + [line_thresh]
            labels = [l.get_label() for l in lines]
            ax5_left.legend(lines, labels, loc='best', fontsize=7)
            ax5_left.grid(True)

            if use_logscale:
                ax5_left.set_xscale('log')
                ax5_left.set_xlim([0.8, len(sv) * 1.1])  # Push x-axis limit beyond final dimension
            else:
                ax5_left.set_xlim([-0.5, len(sv) * 1.02])  # Push x-axis limit beyond final dimension
        else:
            ax5.text(0.5, 0.5, 'Per-Unit Variance\n(Averaged across units)',
                    ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'Variance Info\nNot Available',
                ha='center', va='center', transform=ax5.transAxes)

    # =========================================================================
    # Plot 6: Objective function (or Wiener weights if denoiser_type='wiener')
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 0:2])  # Row 1, columns 0-1

    # Check if Wiener mode (includes all Wiener-family denoisers)
    denoiser_type = opt.get('denoiser_type', 'truncation')
    is_wiener = (denoiser_type == 'wiener')

    if is_wiener and 'wiener_weights' in results:
        # Wiener mode: show Wiener weights and cumulative objective on dual y-axes
        wiener_weights = results['wiener_weights']
        n_weights = len(wiener_weights)

        # For log scale, use x=0.5 for dimension 0, then 1, 2, 3...
        zero_placeholder = 0.5
        if use_logscale:
            x_dims = np.arange(1, n_weights + 1)
        else:
            x_dims = np.arange(n_weights)

        # Left y-axis: Wiener weights (line plot)
        line_weights, = ax6.plot(x_dims, wiener_weights, linewidth=2, color=[0.3, 0.5, 0.8],
                                  label='Wiener weights (w_k)')

        # Add a horizontal line at w=0.5 for reference
        ax6.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Mark the effective dimensionality (sum of weights)
        effective_dims = np.sum(wiener_weights)
        ax6.axvline(x=effective_dims, color='red', linestyle='--', linewidth=2)

        ax6.set_ylabel('Wiener Weight (w_k)', color=[0.3, 0.5, 0.8])
        ax6.tick_params(axis='y', labelcolor=[0.3, 0.5, 0.8])
        ax6.set_ylim([0, 1.05])

        # Right y-axis: Cumulative objective (SignalVar - NoiseVar/ntrials)
        ax6_right = ax6.twinx()
        if 'objective' in results:
            obj = results['objective']
            # Objective has n+1 values (0 to n dims), align with weights
            if use_logscale:
                x_obj = np.concatenate([[zero_placeholder], np.arange(1, len(obj))])
            else:
                x_obj = np.arange(len(obj))
            line_obj, = ax6_right.plot(x_obj, obj, linewidth=2, color=[0.3, 0.7, 0.3],
                                        label='Cumsum SignalVar - NoiseVar/nt')
            ax6_right.set_ylabel('Cumulative SignalVar - NoiseVar/ntrials', color=[0.3, 0.7, 0.3])
            ax6_right.tick_params(axis='y', labelcolor=[0.3, 0.7, 0.3])

            # Star at max of global cumsum objective
            max_idx = np.argmax(obj)
            h_star, = ax6_right.plot(x_obj[max_idx], obj[max_idx], '*', markersize=14,
                                      color=[0.3, 0.7, 0.3], markeredgecolor='black',
                                      markeredgewidth=0.8, zorder=10,
                                      label=f'Max objective (K={max_idx})')

            # Combined legend
            lines = [line_weights, line_obj, h_star]
            labels = [l.get_label() for l in lines]
            ax6.legend(lines, labels, loc='best', fontsize=8)

        ax6.set_xlabel('Dimension')
        ax6.set_title(f'Wiener Weights & Objective (eff. dims = {effective_dims:.1f})')
        ax6.grid(True, alpha=0.3)

        # Use log scale for x-axis if many dimensions
        if use_logscale:
            ax6.set_xscale('log')
            ax6.set_xlim([0.8, n_weights * 1.1])
        else:
            ax6.set_xlim([-0.5, n_weights - 0.5])

    elif 'objective' in results and results['objective'] is not None:
        obj = results['objective']

        # For log scale, we need to handle x=0 specially
        # Use x=0.5 for the zero-dimension case, then 1, 2, 3... for rest
        zero_placeholder = 0.5

        # Check if unit-specific objectives are available
        if 'unit_objectives' in results and results['unit_objectives']:
            # Unit-specific mode: use dual y-axes
            # Left axis: unit curves (gray) - use subsampling if needed
            ax6_left = ax6
            h_units = None
            for u in subsample_idx:
                u_obj = results['unit_objectives'][u]
                if use_logscale:
                    # x=0 -> zero_placeholder, x=1 -> 1, x=2 -> 2, etc.
                    x_unit = np.concatenate([[zero_placeholder], np.arange(1, len(u_obj))])
                else:
                    x_unit = np.arange(len(u_obj))
                line, = ax6_left.plot(x_unit, u_obj, linewidth=0.5,
                        color=[0.5, 0.5, 0.5], alpha=0.3)
                if h_units is None:
                    h_units = line

            # Mark each unit's chosen threshold (on left axis) - use subsampling
            # Color by unit_groups if available
            if 'best_threshold' in results:
                best_t = results['best_threshold']
                if hasattr(best_t, '__len__'):
                    # Check if unit_groups are available for coloring
                    unit_groups = opt.get('unit_groups', None)
                    if unit_groups is not None:
                        unique_groups = np.unique(unit_groups)
                        n_groups = len(unique_groups)
                        # Use hsv colormap which handles arbitrary numbers of groups
                        group_colors = plt.cm.hsv(np.linspace(0, 0.9, n_groups))  # 0.9 to avoid wrapping back to red
                        # Create mapping from group to color
                        group_to_color = {g: group_colors[i] for i, g in enumerate(unique_groups)}

                    x_thresh = []
                    y_thresh = []
                    c_thresh = []  # colors for each point
                    for u in subsample_idx:
                        if u < len(best_t):
                            u_obj = results['unit_objectives'][u]
                            k_u = int(best_t[u])
                            if k_u >= 0 and k_u < len(u_obj):
                                if use_logscale:
                                    if k_u == 0:
                                        x_thresh.append(zero_placeholder)
                                    else:
                                        x_thresh.append(k_u)
                                else:
                                    x_thresh.append(k_u)
                                y_thresh.append(u_obj[k_u])
                                # Get color based on unit group
                                if unit_groups is not None and u < len(unit_groups):
                                    c_thresh.append(group_to_color[unit_groups[u]])
                                else:
                                    c_thresh.append([1, 0.3, 0.3, 0.6])  # default red
                    if x_thresh:
                        if unit_groups is not None:
                            ax6_left.scatter(x_thresh, y_thresh, s=20, c=c_thresh,
                                       alpha=0.6, zorder=5)
                        else:
                            ax6_left.scatter(x_thresh, y_thresh, s=20, color=[1, 0.3, 0.3],
                                       alpha=0.6, zorder=5)

            ax6_left.set_ylabel('Unit-Specific Objective\n(SignalVar - NoiseVar/ntrials)', color=[0.4, 0.4, 0.4])
            ax6_left.tick_params(axis='y', labelcolor=[0.4, 0.4, 0.4])

            # Right axis: population sum (green) - FULL population
            ax6_right = ax6.twinx()
            if use_logscale:
                x_obj = np.concatenate([[zero_placeholder], np.arange(1, len(obj))])
            else:
                x_obj = np.arange(len(obj))
            h_sum, = ax6_right.plot(x_obj, obj, linewidth=2, color=[0.3, 0.7, 0.3])
            ax6_right.set_ylabel('Population Objective', color=[0.3, 0.7, 0.3])
            ax6_right.tick_params(axis='y', labelcolor=[0.3, 0.7, 0.3])

            # Star at max of global cumsum objective
            max_idx = np.argmax(obj)
            h_star, = ax6_right.plot(x_obj[max_idx], obj[max_idx], '*', markersize=14,
                                      color=[0.3, 0.7, 0.3], markeredgecolor='black',
                                      markeredgewidth=0.8, zorder=10)

            # Add legend
            ax6_left.legend([h_units, h_sum, h_star], ['Units', 'Population (=Global)', f'Max objective (K={max_idx})'], loc='best')

            ax6.set_title(f'Objective Function (unit-specific){subsample_suffix_units}')
        else:
            # Global mode: single curve
            if use_logscale:
                x_obj = np.concatenate([[zero_placeholder], np.arange(1, len(obj))])
            else:
                x_obj = np.arange(len(obj))
            ax6.plot(x_obj, obj, linewidth=1.5, color=[0.3, 0.7, 0.3])

            # Mark chosen threshold (not maximum - threshold may be constrained)
            if 'best_threshold' in results and np.isscalar(results['best_threshold']):
                k = int(results['best_threshold'])
                if k >= 0 and k < len(obj):
                    if use_logscale:
                        if k == 0:
                            x_marker = zero_placeholder
                        else:
                            x_marker = k
                    else:
                        x_marker = k
                    ax6.plot(x_marker, obj[k], 'ro', markersize=8, linewidth=2)

            # Star at max of global cumsum objective
            max_idx = np.argmax(obj)
            if use_logscale:
                x_star = zero_placeholder if max_idx == 0 else max_idx
            else:
                x_star = max_idx
            ax6.plot(x_star, obj[max_idx], '*', markersize=14,
                     color=[0.3, 0.7, 0.3], markeredgecolor='black',
                     markeredgewidth=0.8, zorder=10)

            ax6.set_title('Objective Function')

        ax6.set_xlabel('Number of Dimensions')

        # Set ylabel based on criterion (only for global mode - unit-specific mode already set ylabels)
        if not ('unit_objectives' in results and results['unit_objectives']):
            if criterion == 'variance':
                ax6.set_ylabel('Cumulative SignalVar')
            else:
                ax6.set_ylabel('Cumulative SignalVar - NoiseVar/ntrials')

        ax6.grid(True)

        # Apply log scale and fix tick labels if needed
        if use_logscale:
            ax6.set_xscale('log')
            # Set xlim to start at zero_placeholder (so "0" point is visible)
            n_dims = len(obj) - 1  # max dimension
            ax6.set_xlim([zero_placeholder * 0.8, n_dims * 1.1])
            # Set custom ticks to show "0" at the zero_placeholder position
            tick_vals = [zero_placeholder]
            tick_labels = ['0']
            # Add powers of 10 and intermediate values
            log_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
            for lt in log_ticks:
                if lt <= n_dims:
                    tick_vals.append(lt)
                    tick_labels.append(str(lt))
            ax6.set_xticks(tick_vals)
            ax6.set_xticklabels(tick_labels)
    elif basis_desc == 'wiener' and 'signalvar' in results and 'noisevar' in results:
        # Full-rank Wiener: show implied per-dimension weights and cumulative objective
        signal_proj = results['signalvar']
        noise_proj = results['noisevar']
        n_dims = len(signal_proj)

        # Compute implied Wiener weights: w_k = s_k / (s_k + n_k/t)
        denom = signal_proj + noise_proj / ntrials_avg
        wiener_weights = np.zeros_like(signal_proj)
        valid = denom > 0
        wiener_weights[valid] = signal_proj[valid] / denom[valid]
        wiener_weights = np.clip(wiener_weights, 0.0, 1.0)

        # Compute cumulative prediction objective: cumsum(s_k - n_k/t)
        prediction_obj = signal_proj - noise_proj / ntrials_avg
        cumsum_obj = np.concatenate([[0], np.cumsum(prediction_obj)])

        if use_logscale:
            x_dims = np.arange(1, n_dims + 1)
            zero_placeholder = 0.5
        else:
            x_dims = np.arange(n_dims)

        # Left y-axis: Wiener weights
        line_weights, = ax6.plot(x_dims, wiener_weights, linewidth=2, color=[0.3, 0.5, 0.8],
                                  label='Wiener weight $w_k = s_k / (s_k + n_k/t)$')
        ax6.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax6.set_ylabel('Wiener Weight ($w_k$)', color=[0.3, 0.5, 0.8])
        ax6.tick_params(axis='y', labelcolor=[0.3, 0.5, 0.8])
        ax6.set_ylim([0, 1.05])

        # Right y-axis: Cumulative prediction objective
        ax6_right = ax6.twinx()
        if use_logscale:
            x_obj = np.concatenate([[zero_placeholder], np.arange(1, len(cumsum_obj))])
        else:
            x_obj = np.arange(len(cumsum_obj))
        line_obj, = ax6_right.plot(x_obj, cumsum_obj, linewidth=2, color=[0.3, 0.7, 0.3],
                                    label='Cumsum($s_k - n_k/t$)')
        ax6_right.set_ylabel('Cumulative $s_k - n_k/t$', color=[0.3, 0.7, 0.3])
        ax6_right.tick_params(axis='y', labelcolor=[0.3, 0.7, 0.3])

        # Star at max of global cumsum objective
        max_idx = np.argmax(cumsum_obj)
        h_star, = ax6_right.plot(x_obj[max_idx], cumsum_obj[max_idx], '*', markersize=14,
                                  color=[0.3, 0.7, 0.3], markeredgecolor='black',
                                  markeredgewidth=0.8, zorder=10,
                                  label=f'Max objective (K={max_idx})')

        # Mark effective dimensionality
        effective_dims = results.get('best_threshold', np.sum(wiener_weights))
        ax6.axvline(x=effective_dims if not use_logscale else max(effective_dims, 0.5),
                    color='red', linestyle='--', linewidth=2, label=f'Eff. dims = {effective_dims:.1f}')

        # Legend
        lines = [line_weights, line_obj, h_star]
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='best', fontsize=8)

        ax6.set_xlabel('Dimension (cSb eigenbasis)')
        ax6.set_title(f'Implied Wiener Weights in Signal Basis (eff. dims = {effective_dims:.1f})')
        ax6.grid(True, alpha=0.3)

        if use_logscale:
            ax6.set_xscale('log')
            ax6.set_xlim([0.8, n_dims * 1.1])
        else:
            ax6.set_xlim([-0.5, n_dims - 0.5])

    else:
        ax6.text(0.5, 0.5, 'Objective\nNot Available',
                ha='center', va='center', transform=ax6.transAxes)

    # =========================================================================
    # Plot 7-9: Raw, Denoised, Noise
    # =========================================================================

    # Compute shared colorbar limits across all three plots (mean-centered)
    all_data_789 = np.concatenate([trial_avg.ravel(), denoised.ravel(), noise.ravel()])
    if has_nans:
        shared_mean = np.nanmean(all_data_789)
        shared_std = np.nanstd(all_data_789)
    else:
        shared_mean = np.mean(all_data_789)
        shared_std = np.std(all_data_789)
    if shared_std > 0:
        clim_shared = [shared_mean - 2*shared_std, shared_mean + 2*shared_std]
    else:
        clim_shared = [shared_mean - 1, shared_mean + 1]

    # Plot 7: Raw trial-averaged data
    ax7 = fig.add_subplot(gs[1, 2:4])  # Row 1, columns 2-3
    im7 = ax7.imshow(trial_avg, vmin=clim_shared[0], vmax=clim_shared[1], cmap=cmap, aspect='auto', interpolation='none')
    plt.colorbar(im7, ax=ax7)
    title_7 = 'Input Data (trial-averaged, with NaNs)' if has_nans else 'Input Data (trial-averaged)'
    ax7.set_title(title_7)
    ax7.set_xlabel('Conditions')
    ax7.set_ylabel('Units')

    # Plot 8: Denoised data
    ax8 = fig.add_subplot(gs[1, 4:6])  # Row 1, columns 4-5
    im8 = ax8.imshow(denoised, vmin=clim_shared[0], vmax=clim_shared[1], cmap=cmap, aspect='auto', interpolation='none')
    plt.colorbar(im8, ax=ax8)
    ax8.set_title('PSN Denoised Data')
    ax8.set_xlabel('Conditions')
    ax8.set_ylabel('Units')

    # Plot 9: Noise (residual)
    ax9 = fig.add_subplot(gs[1, 6:8])  # Row 1, columns 6-7
    im9 = ax9.imshow(noise, vmin=clim_shared[0], vmax=clim_shared[1], cmap=cmap, aspect='auto', interpolation='none')
    plt.colorbar(im9, ax=ax9)
    title_9 = 'Residual (Noise, with NaNs)' if has_nans else 'Residual (Noise)'
    ax9.set_title(title_9)
    ax9.set_xlabel('Conditions')
    ax9.set_ylabel('Units')

    # =========================================================================
    # Plot 10: Denoiser matrix
    # =========================================================================
    ax10 = fig.add_subplot(gs[2, 0:2])  # Row 2, columns 0-1
    denoiser = results['denoiser']
    if basis_desc == 'wiener' and 'wiener_matrix' in results:
        plot_matrix_10 = results['wiener_matrix']
        title_10 = 'Wiener Filter Matrix'
    else:
        plot_matrix_10 = denoiser
        title_10 = 'Denoiser Matrix'
    # Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
    data_absmax = np.nanpercentile(np.abs(plot_matrix_10), 99) if has_nans else np.percentile(np.abs(plot_matrix_10), 99)
    clim_10 = [-data_absmax, data_absmax] if data_absmax > 0 else [-1, 1]
    im10 = ax10.imshow(plot_matrix_10, vmin=clim_10[0], vmax=clim_10[1], cmap=redblue(), aspect='equal', interpolation='none')
    plt.colorbar(im10, ax=ax10)
    ax10.set_title(title_10)
    ax10.set_xlabel('Units')
    ax10.set_ylabel('Units')

    # =========================================================================
    # Plot 11-12: Traces (use subsampled units and/or conditions if needed)
    # =========================================================================
    # Color conditions by mean response (handle NaNs)
    # Use subsampled conditions for coloring
    n_conds_to_plot = len(subsample_cond_idx)
    cond_means = np.nanmean(trial_avg, axis=0)
    cond_means_sub = cond_means[subsample_cond_idx]
    sorted_indices = np.argsort(cond_means_sub)
    colors = plt.cm.jet(np.linspace(0, 1, n_conds_to_plot))
    trace_colors = np.zeros((n_conds_to_plot, 3))
    for rank in range(n_conds_to_plot):
        cond_idx = sorted_indices[rank]
        trace_colors[cond_idx, :] = colors[rank, :3]  # Use only RGB, not alpha

    # Get subsampled data for traces (both units and conditions)
    trial_avg_sub = trial_avg[np.ix_(subsample_idx, subsample_cond_idx)]
    denoised_sub = denoised[np.ix_(subsample_idx, subsample_cond_idx)]

    # Trial-averaged traces
    ax11 = fig.add_subplot(gs[2, 2:4])  # Row 2, columns 2-3
    x_units = np.arange(len(subsample_idx))
    for c in range(n_conds_to_plot):
        ax11.plot(x_units, trial_avg_sub[:, c], color=trace_colors[c, :], linewidth=0.5)
    ax11.set_xlabel('Units')
    ax11.set_ylabel('Activity')
    ax11.set_title(f'Trial-Averaged Traces{subsample_suffix_traces}')
    ax11.grid(True)
    ax11.set_xlim([x_units[0], x_units[-1]])

    # Denoised traces
    ax12 = fig.add_subplot(gs[2, 4:6])  # Row 2, columns 4-5
    for c in range(n_conds_to_plot):
        ax12.plot(x_units, denoised_sub[:, c], color=trace_colors[c, :], linewidth=0.5)
    ax12.set_xlabel('Units')
    ax12.set_ylabel('Activity')
    ax12.set_title(f'PSN Denoised Traces{subsample_suffix_traces}')
    ax12.grid(True)
    ax12.set_xlim([x_units[0], x_units[-1]])

    # Match y-limits (handle NaNs) - use subsampled data for ylim
    all_trace_data = np.concatenate([trial_avg_sub.ravel(), denoised_sub.ravel()])
    y_min = np.nanmin(all_trace_data) if has_nans else np.min(all_trace_data)
    y_max = np.nanmax(all_trace_data) if has_nans else np.max(all_trace_data)
    y_range = y_max - y_min
    y_margin = y_range * 0.05

    ax11.set_ylim([y_min - y_margin, y_max + y_margin])
    ax12.set_ylim([y_min - y_margin, y_max + y_margin])

    # =========================================================================
    # Plot 13: Split-half reliability (use subsampling for scatter, full pop for means)
    # =========================================================================
    ax13 = fig.add_subplot(gs[2, 6:8])  # Row 2, columns 6-7

    # Split trials by odd/even indices (interleaved) to handle NaN patterns
    # where later trials may have more NaNs due to variable repetition counts
    odd_idx = np.arange(0, ntrials, 2)   # 0, 2, 4, ...
    even_idx = np.arange(1, ntrials, 2)  # 1, 3, 5, ...
    data_A = data[:, :, odd_idx]
    data_B = data[:, :, even_idx]

    # Trial averages (use nanmean to handle NaNs)
    tavg_A = np.nanmean(data_A, axis=2) if has_nans else np.mean(data_A, axis=2)
    tavg_B = np.nanmean(data_B, axis=2) if has_nans else np.mean(data_B, axis=2)

    # Denoise both splits
    unit_means = results['unit_means']

    # Apply denoiser via denoiser.T @ x (correct for all modes:
    # symmetric global denoisers have denoiser.T == denoiser,
    # and non-symmetric denoisers like hybrid/unit/wiener need the transpose)
    dn_A = denoiser.T @ (tavg_A - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]
    dn_B = denoiser.T @ (tavg_B - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]

    # Compute per-unit split-half metric for ALL units
    metric_tavg = np.zeros(nunits)
    metric_cross = np.zeros(nunits)
    metric_dn = np.zeros(nunits)

    use_mse = (split_half_metric == 'mse')

    for u in range(nunits):
        if use_mse:
            # MSE: mean squared error across conditions
            mask = ~(np.isnan(tavg_A[u, :]) | np.isnan(tavg_B[u, :]))
            metric_tavg[u] = np.mean((tavg_A[u, mask] - tavg_B[u, mask])**2) if np.sum(mask) > 0 else np.nan

            mask_AB = ~(np.isnan(tavg_A[u, :]) | np.isnan(dn_B[u, :]))
            mask_BA = ~(np.isnan(dn_A[u, :]) | np.isnan(tavg_B[u, :]))
            mse_AB = np.mean((tavg_A[u, mask_AB] - dn_B[u, mask_AB])**2) if np.sum(mask_AB) > 0 else np.nan
            mse_BA = np.mean((dn_A[u, mask_BA] - tavg_B[u, mask_BA])**2) if np.sum(mask_BA) > 0 else np.nan
            metric_cross[u] = (mse_AB + mse_BA) / 2 if not (np.isnan(mse_AB) or np.isnan(mse_BA)) else np.nan

            mask_dn = ~(np.isnan(dn_A[u, :]) | np.isnan(dn_B[u, :]))
            metric_dn[u] = np.mean((dn_A[u, mask_dn] - dn_B[u, mask_dn])**2) if np.sum(mask_dn) > 0 else np.nan
        else:
            # Correlation (default)
            if np.nanstd(tavg_A[u, :]) > 0 and np.nanstd(tavg_B[u, :]) > 0:
                mask = ~(np.isnan(tavg_A[u, :]) | np.isnan(tavg_B[u, :]))
                if np.sum(mask) > 1:
                    metric_tavg[u] = np.corrcoef(tavg_A[u, mask], tavg_B[u, mask])[0, 1]
                else:
                    metric_tavg[u] = np.nan
            else:
                metric_tavg[u] = np.nan

            if (np.nanstd(tavg_A[u, :]) > 0 and np.nanstd(dn_B[u, :]) > 0 and
                np.nanstd(dn_A[u, :]) > 0 and np.nanstd(tavg_B[u, :]) > 0):
                mask_AB = ~(np.isnan(tavg_A[u, :]) | np.isnan(dn_B[u, :]))
                mask_BA = ~(np.isnan(dn_A[u, :]) | np.isnan(tavg_B[u, :]))
                if np.sum(mask_AB) > 1 and np.sum(mask_BA) > 1:
                    corr_AB = np.corrcoef(tavg_A[u, mask_AB], dn_B[u, mask_AB])[0, 1]
                    corr_BA = np.corrcoef(dn_A[u, mask_BA], tavg_B[u, mask_BA])[0, 1]
                    metric_cross[u] = (corr_AB + corr_BA) / 2
                else:
                    metric_cross[u] = np.nan
            else:
                metric_cross[u] = np.nan

            if np.nanstd(dn_A[u, :]) > 0 and np.nanstd(dn_B[u, :]) > 0:
                mask = ~(np.isnan(dn_A[u, :]) | np.isnan(dn_B[u, :]))
                if np.sum(mask) > 1:
                    metric_dn[u] = np.corrcoef(dn_A[u, mask], dn_B[u, mask])[0, 1]
                else:
                    metric_dn[u] = np.nan
            else:
                metric_dn[u] = np.nan

    # Subsampled metrics for plotting
    metric_tavg_sub = metric_tavg[subsample_idx]
    metric_cross_sub = metric_cross[subsample_idx]
    metric_dn_sub = metric_dn[subsample_idx]
    n_sub = len(subsample_idx)

    # Plot
    x_positions = np.array([1, 2, 3])
    labels = ['TAvg vs TAvg', 'TAvg vs Denoised', 'Denoised vs Denoised']

    # Add jitter for subsampled units
    x_jitter_sub = (np.random.rand(n_sub) - 0.5) * 0.16

    # Connecting lines (subsampled)
    for ii in range(n_sub):
        values = [metric_tavg_sub[ii], metric_cross_sub[ii], metric_dn_sub[ii]]
        if not np.any(np.isnan(values)):
            ax13.plot(x_positions + x_jitter_sub[ii], values,
                     color=[0.5, 0.5, 0.5], linewidth=0.3)

    # Scatter points (subsampled)
    ax13.scatter(x_positions[0] + x_jitter_sub, metric_tavg_sub, s=15, color='blue',
                alpha=0.4, zorder=2)
    ax13.scatter(x_positions[1] + x_jitter_sub, metric_cross_sub, s=15, color=[1, 0.84, 0],
                alpha=0.4, zorder=2)
    ax13.scatter(x_positions[2] + x_jitter_sub, metric_dn_sub, s=15, color=[0.5, 0.8, 0.3],
                alpha=0.4, zorder=2)

    # Means (FULL population)
    mean_tavg = np.nanmean(metric_tavg)
    mean_cross = np.nanmean(metric_cross)
    mean_dn = np.nanmean(metric_dn)

    ax13.scatter(x_positions[0], mean_tavg, s=100, color='blue',
                edgecolors='white', linewidths=2, zorder=3)
    ax13.scatter(x_positions[1], mean_cross, s=100, color=[1, 0.84, 0],
                edgecolors='white', linewidths=2, zorder=3)
    ax13.scatter(x_positions[2], mean_dn, s=100, color=[0.2, 0.6, 0.2],
                edgecolors='white', linewidths=2, zorder=3)

    # Labels (FULL population means)
    all_vals_sub = np.concatenate([metric_tavg_sub, metric_cross_sub, metric_dn_sub])
    valid_vals = all_vals_sub[~np.isnan(all_vals_sub)]
    y_range_metric = (np.max(valid_vals) - np.min(valid_vals)) if len(valid_vals) > 0 else 1
    y_offset = y_range_metric * 0.06
    ax13.text(x_positions[0], mean_tavg + y_offset, f'{mean_tavg:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax13.text(x_positions[1], mean_cross + y_offset, f'{mean_cross:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax13.text(x_positions[2], mean_dn + y_offset, f'{mean_dn:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax13.set_xticks(x_positions)
    ax13.set_xticklabels(labels, rotation=0, fontsize=7)
    ax13.set_ylabel('MSE' if use_mse else 'Pearson r')

    metric_label = 'MSE' if use_mse else 'Reliability'
    if has_nans:
        valid_A = np.sum(~np.any(np.isnan(data_A), axis=0), axis=1)
        valid_B = np.sum(~np.any(np.isnan(data_B), axis=0), axis=1)
        avg_valid_A = np.mean(valid_A[valid_A > 0])
        avg_valid_B = np.mean(valid_B[valid_B > 0])
        ax13.set_title(f'Split-Half {metric_label}\n({avg_valid_A:.1f} vs {avg_valid_B:.1f} avg trials){subsample_suffix_units}')
    else:
        ax13.set_title(f'Split-Half {metric_label}\n({data_A.shape[2]} vs {data_B.shape[2]} trials){subsample_suffix_units}')

    ax13.grid(True)
    ax13.set_xlim([0.5, 3.5])
    if not use_mse:
        ax13.axhline(y=0, color='k', linewidth=1)

    # Set y-limits
    if len(valid_vals) > 0:
        y_min_c = np.min(valid_vals)
        y_max_c = np.max(valid_vals)
        y_range_c = y_max_c - y_min_c
        y_pad = max(0.1 if not use_mse else y_range_c * 0.1, y_range_c * 0.15)
        ax13.set_ylim([y_min_c - y_pad, y_max_c + y_pad])
    else:
        ax13.set_ylim([-1, 1] if not use_mse else [0, 1])

    # =========================================================================
    # Plot 14-17: Signal/Noise Diagnostics (use subsampling for scatter, full pop for means)
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

        # Subsampled versions for plotting
        sv_before_sub = sv_before[subsample_idx]
        sv_after_sub = sv_after[subsample_idx]
        nv_before_sub = nv_before[subsample_idx]
        nv_after_sub = nv_after[subsample_idx]
        ncsnr_before_sub = ncsnr_before[subsample_idx]
        ncsnr_after_sub = ncsnr_after[subsample_idx]
        noiseceiling_before_sub = noiseceiling_before[subsample_idx]
        noiseceiling_after_sub = noiseceiling_after[subsample_idx]
        n_sub = len(subsample_idx)

        # Define x positions
        x_before = 1
        x_after = 2
        x_jitter_diag = (np.random.rand(n_sub) - 0.5) * 0.1

        # Plot 14: Signal Variance
        ax14 = fig.add_subplot(gs[3, 0:2])  # Row 3, columns 0-1
        for ii in range(n_sub):
            ax14.plot([x_before, x_after] + x_jitter_diag[ii],
                     [sv_before_sub[ii], sv_after_sub[ii]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax14.scatter(x_before + x_jitter_diag, sv_before_sub, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax14.scatter(x_after + x_jitter_diag, sv_after_sub, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        # Means from FULL population
        mean_sv_before = np.mean(sv_before)
        mean_sv_after = np.mean(sv_after)
        ax14.scatter(x_before, mean_sv_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax14.scatter(x_after, mean_sv_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically (use subsampled for range)
        y_range_sv = np.max([sv_before_sub.max(), sv_after_sub.max()]) - np.min([sv_before_sub.min(), sv_after_sub.min()])
        y_offset_sv = y_range_sv * 0.08

        ax14.text(x_before, mean_sv_before + y_offset_sv, f'{mean_sv_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax14.text(x_after, mean_sv_after + y_offset_sv, f'{mean_sv_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax14.set_xlim([0.5, 2.5])
        ax14.set_xticks([1, 2])
        ax14.set_xticklabels(['Before', 'After'])
        ax14.set_ylabel('Signal Variance')
        ax14.set_title(f'Signal Variance{subsample_suffix_units}')
        ax14.grid(True)

        # Plot 15: Noise Variance
        ax15 = fig.add_subplot(gs[3, 2:4])  # Row 3, columns 2-3
        for ii in range(n_sub):
            ax15.plot([x_before, x_after] + x_jitter_diag[ii],
                     [nv_before_sub[ii], nv_after_sub[ii]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax15.scatter(x_before + x_jitter_diag, nv_before_sub, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax15.scatter(x_after + x_jitter_diag, nv_after_sub, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        # Means from FULL population
        mean_nv_before = np.mean(nv_before)
        mean_nv_after = np.mean(nv_after)
        ax15.scatter(x_before, mean_nv_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax15.scatter(x_after, mean_nv_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically
        y_range_nv = np.max([nv_before_sub.max(), nv_after_sub.max()]) - np.min([nv_before_sub.min(), nv_after_sub.min()])
        y_offset_nv = y_range_nv * 0.08

        ax15.text(x_before, mean_nv_before + y_offset_nv, f'{mean_nv_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax15.text(x_after, mean_nv_after + y_offset_nv, f'{mean_nv_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax15.set_xlim([0.5, 2.5])
        ax15.set_xticks([1, 2])
        ax15.set_xticklabels(['Before', 'After'])
        ax15.set_ylabel('Noise Variance / ntrials')
        ax15.set_title(f'Trial-Averaged Noise Variance{subsample_suffix_units}')
        ax15.grid(True)

        # Set unified ylims for both signal and noise variance plots (use subsampled for range)
        all_variance_vals_sub = np.concatenate([sv_before_sub, sv_after_sub, nv_before_sub, nv_after_sub])
        y_max_unified = np.max(all_variance_vals_sub)
        y_pad_unified = y_max_unified * 0.05  # 5% padding at bottom
        unified_ylim = [-y_pad_unified, y_max_unified + y_max_unified * 0.15]

        ax14.set_ylim(unified_ylim)
        ax14.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax15.set_ylim(unified_ylim)
        ax15.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # Plot 16: NCSNR
        ax16 = fig.add_subplot(gs[3, 4:6])  # Row 3, columns 4-5
        for ii in range(n_sub):
            ax16.plot([x_before, x_after] + x_jitter_diag[ii],
                     [ncsnr_before_sub[ii], ncsnr_after_sub[ii]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax16.scatter(x_before + x_jitter_diag, ncsnr_before_sub, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax16.scatter(x_after + x_jitter_diag, ncsnr_after_sub, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        # Means from FULL population
        mean_ncsnr_before = np.mean(ncsnr_before)
        mean_ncsnr_after = np.mean(ncsnr_after)
        ax16.scatter(x_before, mean_ncsnr_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax16.scatter(x_after, mean_ncsnr_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically
        y_range_ncsnr = np.max([ncsnr_before_sub.max(), ncsnr_after_sub.max()]) - np.min([ncsnr_before_sub.min(), ncsnr_after_sub.min()])
        y_offset_ncsnr = y_range_ncsnr * 0.08

        ax16.text(x_before, mean_ncsnr_before + y_offset_ncsnr, f'{mean_ncsnr_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax16.text(x_after, mean_ncsnr_after + y_offset_ncsnr, f'{mean_ncsnr_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Set ylims with padding (use subsampled for range)
        all_ncsnr_vals_sub = np.concatenate([ncsnr_before_sub, ncsnr_after_sub])
        y_max_ncsnr = np.max(all_ncsnr_vals_sub)
        y_pad_ncsnr_bottom = y_max_ncsnr * 0.05  # 5% padding at bottom
        y_pad_ncsnr_top = y_max_ncsnr * 0.15  # 15% padding at top
        ax16.set_ylim([-y_pad_ncsnr_bottom, y_max_ncsnr + y_pad_ncsnr_top])
        ax16.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax16.set_xlim([0.5, 2.5])
        ax16.set_xticks([1, 2])
        ax16.set_xticklabels(['Before', 'After'])
        ax16.set_ylabel('NCSNR')
        ax16.set_title(f'Noise Ceiling SNR (NCSNR){subsample_suffix_units}')
        ax16.grid(True)

        # Plot 17: Noise Ceiling %
        ax17 = fig.add_subplot(gs[3, 6:8])  # Row 3, columns 6-7
        for ii in range(n_sub):
            ax17.plot([x_before, x_after] + x_jitter_diag[ii],
                     [noiseceiling_before_sub[ii], noiseceiling_after_sub[ii]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax17.scatter(x_before + x_jitter_diag, noiseceiling_before_sub, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax17.scatter(x_after + x_jitter_diag, noiseceiling_after_sub, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        # Means from FULL population
        mean_nc_before = np.mean(noiseceiling_before)
        mean_nc_after = np.mean(noiseceiling_after)
        ax17.scatter(x_before, mean_nc_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax17.scatter(x_after, mean_nc_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Fixed y_offset for noise ceiling (percentage scale 0-100)
        y_offset_nc = 100 * 0.08

        ax17.text(x_before, mean_nc_before + y_offset_nc, f'{mean_nc_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax17.text(x_after, mean_nc_after + y_offset_nc, f'{mean_nc_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax17.set_xlim([0.5, 2.5])
        ax17.set_ylim([-5, 100])  # Add negative padding to make yline at 0 visible
        ax17.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax17.set_xticks([1, 2])
        ax17.set_xticklabels(['Before', 'After'])
        ax17.set_ylabel('Noise Ceiling (%)')

        if has_nans:
            ax17.set_title(f'Noise Ceiling Percentage ({ntrials_avg:.1f} avg trials){subsample_suffix_units}')
        else:
            ax17.set_title(f'Noise Ceiling Percentage ({ntrials} trials){subsample_suffix_units}')

        ax17.grid(True)

    # Save figure if figurepath specified, otherwise show it
    if figurepath is not None:
        fig.savefig(figurepath, dpi=150, bbox_inches='tight')
        # Don't show - just save and return (caller will close)
    else:
        plt.show()

    return fig
