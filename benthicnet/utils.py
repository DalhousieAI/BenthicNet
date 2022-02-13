"""
Utility functions.
"""

import numpy as np


def unique_map(arr):
    """
    Generate a mapping from unique value in an array to their locations.

    Parameters
    ----------
    arr : array_like
        Input array, containing repeated values.

    Returns
    -------
    mapping : dict
        Dictionary whose keys are the unique values in `arr`, and values are a
        list of indices to the entries in `arr` whose value is that key.
    """
    mapping = {}
    for i, x in enumerate(arr):
        if x not in mapping:
            mapping[x] = [i]
        else:
            mapping[x].append(i)
    return mapping


def first_nonzero(arr, axis=-1, invalid_val=-1):
    """
    Find the index of the first non-zero element in an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to search.
    axis : int, optional
        Axis along which to search for a non-zero element. Default is `-1`.
    invalid_val : any, optional
        Value to return if all elements are zero. Default is `-1`.
    """
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
