"""
Utility functions.
"""

import os

import numpy as np
import pandas as pd


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


def read_csv(fname, **kwargs):
    """
    Load a BenthicNet CSV file with dtype set correctly for expected fields.
    """
    return pd.read_csv(
        fname,
        dtype={
            "altitude": float,
            "dataset": str,
            "depth": float,
            "image": str,
            "latitude": float,
            "longitude": float,
            "site": str,
            "timestamp": str,
            "url": str,
        },
        parse_dates=["timestamp"],
        **kwargs,
    )


def clean_df(df, inplace=True):
    """
    Clean dataframe by removing trailing spaces from string columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe in BenthicNet format.
    inplace : bool, optional
        Whether the cleaning operation should be inplace.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned dataframe.
    """
    if not inplace:
        df = df.copy()
    for column in ["dataset", "site", "image", "url"]:
        if column in df.columns:
            df[column] = df[column].str.strip()
    return df


def count_lines(filename):
    """
    Count the number of lines in a file.

    Parameters
    ----------
    filename : str
        Path to file.

    Returns
    -------
    int
        Number of lines in file.
    """
    with open(filename, "rb") as f:
        for _i, _ in enumerate(f):
            pass
    return _i + 1


def file_size(filename, sf=3):
    """
    Get the size of a file, in human readable units.

    Parameters
    ----------
    filename : str
        Path to file.
    sf : int, optional
        Number of significant figures to include. Default is `3`.

    Returns
    -------
    str
        The size of the file in human readable format.
    """

    def convert_bytes(num):
        """
        Convert units of bytes into human readable kB, MB, GB, or TB.
        """
        for unit in ["bytes", "kB", "MB", "GB", "TB"]:
            if num >= 1024:
                num /= 1024
                continue
            digits = sf
            if num >= 1:
                digits -= 1
            if num >= 10:
                digits -= 1
            if num >= 100:
                digits -= 1
            if num >= 1000:
                digits -= 1
            digits = max(0, digits)
            fmt = "{:." + str(digits) + "f} {}"
            return fmt.format(num, unit)

    return convert_bytes(os.stat(filename).st_size)


def sanitize_filename_series(series, allow_dotfiles=False):
    """
    Sanitise strings in a Series, to remove characters forbidden in filenames.

    Parameters
    ----------
    series : pandas.Series
        Input Series of values.
    allow_dotfiles : bool, optional
        Whether to allow leading periods. Leading periods indicate hidden files
        on Linux. Default is `False`.

    Returns
    -------
    pandas.Series
        Sanitised version of `series`.
    """
    # Folder names cannot end with a period in Windows. In Unix, a leading
    # period means the file or folder is normally hidden.
    # For this reason, we trim away leading and trailing periods as well as
    # spaces.
    if allow_dotfiles:
        series = series.str.strip()
        series = series.str.rstrip(".")
    else:
        series = series.str.strip(" .")

    # On Windows, a filename cannot contain any of the following characters:
    # \ / : * ? " < > |
    # Other operating systems are more permissive.
    # Replace / with a hyphen
    series = series.str.replace("/", "-", regex=False)
    # Use a blacklist to remove any remaining forbidden characters
    series = series.str.replace(r'[\/:*?"<>|]+', "", regex=True)
    return series
