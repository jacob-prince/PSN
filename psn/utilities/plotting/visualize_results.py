"""Visualization wrapper for PSN."""

import matplotlib.pyplot as plt


def visualize_results(results, opt):
    """VISUALIZE_RESULTS  Create diagnostic figures

    visualize_results(results, opt) generates diagnostic visualizations of
    PSN denoising results by calling the external visualization module.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <results> - dict containing PSN results (output from psn main function)

    <opt> - dict with PSN options. Relevant fields:
      .wantverbose - if True, print messages about visualization status
      .figurepath  - if specified, save figure to this path and close it

    -------------------------------------------------------------------------
    Behavior:
    -------------------------------------------------------------------------

    If visualization module exists, calls it with (input_data, results).
    Otherwise, prints a message (if wantverbose=True) and skips visualization.

    If figurepath is specified, saves the figure to that path and closes it.
    """

    figurepath = opt.get('figurepath', None)
    cmap = opt.get('cmap', None)

    try:
        from .visualization import plot_diagnostic_figures

        # Call visualization with figurepath and cmap - it will save before plt.show()
        fig = plot_diagnostic_figures(results['input_data'], results, figurepath=figurepath, cmap=cmap)

        if figurepath is not None:
            if opt.get('wantverbose', True):
                print(f'PSN: Diagnostic figure saved to: {figurepath}')
            # Close the figure after saving to free memory
            plt.close(fig)

    except ImportError:
        # Visualization module not found
        if opt.get('wantverbose', True):
            print('  (visualization module not found - skipping figures)')
