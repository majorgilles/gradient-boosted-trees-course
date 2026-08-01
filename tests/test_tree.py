"""Focused checks for the one-feature CART split helpers from Lesson 2."""

import numpy as np

from boosted_trees_course.tree import best_split_1d, midpoint_thresholds, squared_error


def test_best_split_matches_visible_housing_slice() -> None:
    """The helper should reproduce the best row in the lesson notebook."""
    x = np.array(
        [
            0.4999,
            1.8472,
            2.2756,
            2.6406,
            3.0179,
            3.3427,
            3.7031,
            4.1136,
            4.6071,
            5.2500,
            6.2946,
            15.0001,
        ]
    )
    y = np.array(
        [
            0.675,
            1.370,
            2.214,
            1.297,
            2.725,
            1.542,
            1.733,
            2.799,
            2.399,
            2.472,
            4.500,
            4.000,
        ]
    )

    result = best_split_1d(x, y)

    assert np.isclose(result.threshold, 5.7723)
    assert np.isclose(result.left_prediction, 1.9226)
    assert np.isclose(result.right_prediction, 4.25)
    assert np.isclose(result.left_loss, 4.4596864)
    assert np.isclose(result.right_loss, 0.125)
    assert np.isclose(result.total_loss, 4.5846864)


def test_midpoint_thresholds_ignore_duplicate_feature_values() -> None:
    """Equal feature values should not introduce redundant candidate gaps."""
    x = np.array([1.0, 2.0, 2.0, 5.0])

    thresholds = midpoint_thresholds(x)

    assert np.allclose(thresholds, np.array([1.5, 3.5]))


def test_squared_error_is_zero_for_constant_targets() -> None:
    """Targets equal to their mean should have no residual error."""
    y = np.array([5.0, 5.0, 5.0])

    assert squared_error(y) == 0.0


def test_valid_split_has_zero_loss_for_constant_targets() -> None:
    """Both children should retain zero RSS when every target is constant."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([5.0, 5.0, 5.0])

    result = best_split_1d(x, y)

    assert result.left_loss == 0.0
    assert result.right_loss == 0.0
    assert result.total_loss == 0.0
