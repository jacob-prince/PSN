"""Data validation utility for PSN."""

import numpy as np


def validate_data(data):
    """VALIDATE_DATA  Check that data has correct shape and valid values

    [nunits, nconds, ntrials, ntrials_avg, has_nans] = validate_data(data)
    validates the input data array and extracts dimension information.

    -------------------------------------------------------------------------
    Inputs:
    -------------------------------------------------------------------------

    <data> - [nunits x nconds x ntrials] numeric array. Must have at least
      2 trials and 2 conditions. May contain NaNs to indicate missing trials.

    -------------------------------------------------------------------------
    Returns:
    -------------------------------------------------------------------------

    <nunits> - number of units (first dimension of data)

    <nconds> - number of conditions (second dimension of data)

    <ntrials> - maximum number of trials (third dimension of data)

    <ntrials_avg> - average number of valid trials per condition (used in
      PSN formulas). For data without NaNs, equals <ntrials>. For data with
      NaNs, computed as sum(validcnt(validcnt>1))/nconds, following GSN's
      approach in rsanoiseceiling.m

    <has_nans> - boolean. True if data contains any NaN values
    """

    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError('Data must be a 3D numeric array [nunits x nconds x ntrials]')

    nunits, nconds, ntrials = data.shape

    if ntrials < 2:
        raise ValueError(f'Data must have at least 2 trials (needed to estimate noise). Got {ntrials} trials.')

    if nconds < 2:
        raise ValueError(f'Data must have at least 2 conditions (needed to estimate covariance). Got {nconds} conditions.')

    # Check for NaNs and compute average number of trials
    # Following GSN's approach in rsanoiseceiling.m:195-239
    has_nans = np.isnan(data).any()

    if has_nans:
        # Count valid trials per condition (trials with no NaNs across all units)
        validcnt = np.sum(~np.any(np.isnan(data), axis=0), axis=1)  # [nconds]

        # Validate that each condition has at least 1 valid trial
        if np.any(validcnt < 1):
            raise ValueError('All conditions must have at least 1 valid trial (no NaNs)')

        # Compute average number of trials across conditions with >= 2 trials
        # This follows GSN's formula: ntrialBC = sum(validcnt(validcnt>1))/ncond
        ntrials_avg = np.sum(validcnt[validcnt > 1]) / nconds

        if ntrials_avg < 1:
            import warnings
            warnings.warn('Average number of trials is lopsided! Setting to 1')
            ntrials_avg = 1.0
    else:
        # No NaNs: average equals actual
        ntrials_avg = float(ntrials)

    return nunits, nconds, ntrials, ntrials_avg, has_nans
