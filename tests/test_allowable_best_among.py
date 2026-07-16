"""best-among-allowable threshold selection.

When allowable_thresholds restricts the choice, PSN must pick the BEST threshold
among the allowable values, rather than finding the unconstrained optimum and snapping to the
nearest. These tests pin that behavior and, in the distinguishing cases, assert
the result differs from the old snap-to-nearest result. Mirrored by the MATLAB
matlab/tests/test_allowable_best_among.m.
"""

import numpy as np
import pytest

from psn import psn
from psn.utilities.threshold.constrain_to_allowable import constrain_to_allowable
from psn.utilities.threshold.max_tradeoff import max_tradeoff_threshold
from psn.utilities.threshold.select_allowable import argmax_allowable, first_reach_allowable
from psn.utilities.threshold.select_threshold_analytic import select_threshold_analytic


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_argmax_allowable_basic_and_ties():
    obj = np.array([0, 5, 9, 10, 8, 3, 1.0])      # unconstrained argmax = 3
    # obj[1]=5 > obj[5]=3 -> pick 1 (the farther allowable wins on objective)
    assert argmax_allowable(obj, [1, 5]) == 1
    # ...whereas snap-to-nearest would round the tie up to 5 -> they differ
    assert constrain_to_allowable(3, [1, 5]) == 5


def test_first_reach_allowable_basic():
    cumcurve = np.array([0, 2, 4, 6, 8, 10.0])
    assert first_reach_allowable(cumcurve, 3, [1, 5]) == 5      # smallest allowable reaching 3
    assert first_reach_allowable(cumcurve, 99, [1, 2]) == 2     # none reach -> largest allowable


def test_single_value_forces():
    obj = np.array([0, 5, 9, 10, 8, 3, 1.0])
    cumcurve = np.array([0, 2, 4, 6, 8, 10.0])
    assert argmax_allowable(obj, [4]) == 4
    assert first_reach_allowable(cumcurve, 3, [4]) == 4


# --------------------------------------------------------------------------- #
# select_threshold_analytic: best-among-allowable != snap-to-nearest
# --------------------------------------------------------------------------- #
def test_prediction_best_not_snap():
    # diff = [5,4,1,-2,-5,-2] -> cumsum objective [0,5,9,10,8,3,1], peak at k=3
    signal = np.array([5.0, 4, 1, 0, 0, 0])
    noise = np.array([0.0, 0, 0, 2, 5, 2])
    opt = {'criterion': 'prediction', 'basis': 'signal',
           'variance_threshold': 0.99, 'alpha': None, 'allowable_thresholds': np.array([1, 5])}
    k, _ = select_threshold_analytic(signal, noise, None, 1, opt)
    assert k == 1                                  # best among {1,5}
    assert constrain_to_allowable(3, [1, 5]) == 5  # old snap would have given 5


def test_variance_best_not_snap():
    signal = np.array([2.0, 2, 2, 2, 2])           # cumsum [0,2,4,6,8,10]
    opt = {'criterion': 'variance', 'basis': 'signal',
           'variance_threshold': 0.30, 'alpha': None, 'allowable_thresholds': np.array([1, 5])}
    k, _ = select_threshold_analytic(signal, np.zeros(5), None, 1, opt)
    assert k == 5                                   # smallest allowable reaching 0.3*10=3 -> 5
    assert constrain_to_allowable(2, [1, 5]) == 1   # old snap of unconstrained k=2 -> 1


# --------------------------------------------------------------------------- #
# end-to-end across criteria: membership + matches recomputed best-among-allowable
# --------------------------------------------------------------------------- #
@pytest.fixture
def lowrank_data():
    rng = np.random.RandomState(1)
    nunits, nconds, ntrials = 20, 60, 4
    sig = (rng.randn(nunits, 6)) @ rng.randn(6, nconds)
    return sig[:, :, None] + 0.7 * rng.randn(nunits, nconds, ntrials)


def test_end_to_end_membership_global(lowrank_data):
    allow = [2, 5, 9, 14]
    base = dict(wantfig=0, wantverbose=0, threshold_method='global', basis='signal')

    r = psn(lowrank_data, {**base, 'criterion': 'prediction', 'allowable_thresholds': allow})
    k, obj = int(r['best_threshold']), np.asarray(r['objective'])
    assert k in allow and k == argmax_allowable(obj, allow)

    r = psn(lowrank_data, {**base, 'criterion': 'variance',
                           'variance_threshold': 0.9, 'allowable_thresholds': allow})
    k, obj = int(r['best_threshold']), np.asarray(r['objective'])
    assert k in allow and k == first_reach_allowable(obj, 0.9 * obj[-1], allow)

    r = psn(lowrank_data, {**base, 'criterion': 'max-tradeoff', 'allowable_thresholds': allow})
    k = int(r['best_threshold'])
    sigp, noip = np.asarray(r['signalvar']), np.asarray(r['noisevar'])
    assert k in allow and k == max_tradeoff_threshold(sigp, noip, 4, allowable=allow)


def test_end_to_end_single_value_forces(lowrank_data):
    r = psn(lowrank_data, dict(wantfig=0, wantverbose=0, threshold_method='global',
                               basis='signal', criterion='prediction',
                               allowable_thresholds=[7]))
    assert int(r['best_threshold']) == 7


def test_end_to_end_hybrid_membership(lowrank_data):
    allow = [2, 5, 9, 14]
    r = psn(lowrank_data, dict(wantfig=0, wantverbose=0, threshold_method='hybrid',
                               basis='signal', criterion='max-tradeoff',
                               allowable_thresholds=allow))
    bt = np.asarray(r['best_threshold'])
    assert np.all(np.isin(bt, allow))
