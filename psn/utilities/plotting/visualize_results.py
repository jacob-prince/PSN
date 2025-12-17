"""Visualization wrapper for PSN."""


def visualize_results(results, opt):
    """VISUALIZE_RESULTS  Create diagnostic figures

    visualize_results(results, opt) generates diagnostic visualizations of
    PSN denoising results by calling the external visualization module.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <results> - dict containing PSN results (output from psn main function)

    <opt> - dict with PSN options. Relevant field:
      .wantverbose - if True, print messages about visualization status

    -------------------------------------------------------------------------
    Behavior:
    -------------------------------------------------------------------------

    If visualization module exists, calls it with (input_data, results).
    Otherwise, prints a message (if wantverbose=True) and skips visualization
    """

    try:
        from .visualization import plot_diagnostic_figures
        plot_diagnostic_figures(results['input_data'], results)
    except ImportError:
        # Visualization module not found
        if opt.get('wantverbose', True):
            print('  (visualization module not found - skipping figures)')
