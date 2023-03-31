"""
CSV file handling functions.
"""

import os
import re
import time
from functools import partial

import numpy as np
import pandas as pd


def delayed_delete(fname, t_wait=5, verbose=1, padding=""):
    """
    Delete a file, after a delay in which the user can interrupt the process.

    Parameters
    ----------
    fname : str
        Path to file to delete.
    t_wait : int or float, default=5
        Number of seconds to wait before deletion.
    verbose : int, default=1
        Level of verbosity.
    padding : str, default=""
        String to prepend to each line of printed text.
    """
    if not os.path.isfile(fname):
        return
    if verbose >= 1:
        print(f"{padding}File {fname} will be deleted in {t_wait} seconds...")
        print(f"{padding}Deleting in", end="", flush=True)
        for i in range(t_wait // 1):
            print(" {}...".format(t_wait - i), end="", flush=True)
            time.sleep(1)
        print(" Deleting!")
        if t_wait % 1 > 0:
            time.sleep(t_wait % 1)
    else:
        time.sleep(t_wait)
    os.remove(fname)
    if verbose >= 1:
        print(f"{padding}Existing file {fname} deleted", flush=True)


def read_csv(fname, expect_datetime=True, **kwargs):
    """
    Load a BenthicNet CSV file with dtype set correctly for expected fields.

    Parameters
    ----------
    fname : str
        Path to CSV file.
    expect_datetime : bool, default=True
        Whether to expect a datetime column. If ``True``, this will be parsed
        as a datetime column. This must be set to ``False`` if the CSV file
        does not have a ``datetime`` column.
    **kwargs
        Additional arguments as per :func:`pandas.read_csv`.
    """
    extra_columns = {}
    if expect_datetime:
        if "parse_dates" not in kwargs:
            kwargs["parse_dates"] = ["datetime"]

    return pd.read_csv(
        fname,
        dtype={
            "altitude": float,
            "dataset": str,
            "datetime": str,
            "depth": float,
            "image": str,
            "latitude": float,
            "longitude": float,
            "platform": str,
            "site": str,
            "url": str,
            **extra_columns,
        },
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


def sanitize_filename(text, allow_dotfiles=False):
    """
    Sanitise strings in a Series, to remove characters forbidden in filenames.

    Parameters
    ----------
    text : str
        A string.
    allow_dotfiles : bool, optional
        Whether to allow leading periods. Leading periods indicate hidden files
        on Linux. Default is `False`.

    Returns
    -------
    pandas.Series
        Sanitised version of `series`.
    """
    # Replace NaN values with empty string
    if pd.isna(text):
        return ""

    # Remove non-ascii characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Folder names cannot end with a period in Windows. In Unix, a leading
    # period means the file or folder is normally hidden.
    # For this reason, we trim away leading and trailing periods as well as
    # spaces.
    if allow_dotfiles:
        text = text.strip().rstrip(".")
    else:
        text = text.strip(" .")

    # On Windows, a filename cannot contain any of the following characters:
    # \ / : * ? " < > |
    # Other operating systems are more permissive.
    # Replace / with a hyphen
    text = text.replace("/", "-")
    # Use a blacklist to remove any remaining forbidden characters
    text = re.sub(r'[\/:*?"<>|]+', "", text)
    return text


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
    # Make a copy so we do not modify the original
    series = series.copy()
    # Replace NaN values with empty string
    series[pd.isna(series)] = ""

    # Remove non-ascii characters
    series = series.astype(str).str.encode("ascii", "ignore").str.decode("ascii")

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


def row2basename(row, use_url_extension=True):
    """
    Map row to image basename, with extension as per the URL.

    The basename is taken from the image field, extension from the URL field.
    If the image field is empty, the basename of the URL is used instead.

    Parameters
    ----------
    row : dict
        Pandas row or otherwise dictionary-like object with field ``"url"``
        and, optionally, ``"image"``.
    use_url_extension : bool, default=True
        Whether to override the file extension in the image column with the
        extension in the URL.

    Returns
    -------
    basename : str
        The basename of the image to download.
    """
    if "image" in row:
        basename = row["image"]
    else:
        basename = ""
    if pd.isna(basename) or not basename:
        if pd.isna(row["url"]):
            return ""
        basename = row["url"].rstrip("/").split("/")[-1]
    basename = sanitize_filename(basename)
    if not use_url_extension or "url" not in row or pd.isna(row["url"]):
        return basename
    if "url" in row and not pd.isna(row["url"]):
        # Ensure we enclude the extension from the URL
        ext = os.path.splitext(basename)[1]
        expected_ext = os.path.splitext(row["url"].rstrip("/"))[1]
        if expected_ext and ext.lower() != expected_ext.lower():
            if ext.lower() in {".jpg", ".jpeg"}:
                basename = os.path.splitext(basename)[0] + expected_ext
            else:
                basename = basename + expected_ext
    return basename


def determine_outpath(df, use_url_extension=True):
    """
    Determine output path (assuming non-tar file output) to each image.

    The output path is in the format <dataset>/<site>/<imagename.jpg>.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with columns ``"dataset"``,  ``"site"``, ``"url"`` and
        (optionally) ``"image"``.
    use_url_extension : bool, default=True
        Whether to override the file extension in the image column with the
        extension in the URL.

    Returns
    -------
    df : pandas.Series of str
        Output paths to each image in the dataset.
    """
    return (
        sanitize_filename_series(df["dataset"])
        + "/"
        + sanitize_filename_series(df["site"])
        + "/"
        + df.apply(partial(row2basename, use_url_extension=use_url_extension), axis=1)
    )


def fixup_repeated_output_paths(df, inplace=True, verbose=1):
    """
    Rename image values to prevent output path collisions.

    The new name takes extra content from the url field to prevent collisions.
    If more than 4 components of the URL would be needed, image is padded with
    an incrementing counter instead (_0, _1, ..., _N).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with columns ``"dataset"``,  ``"site"``, ``"url"`` and
        (optionally) ``"image"``.
    inplace : bool, default=True
        Whether to overwrite the contents of the input dataframe.
    verbose : int, default=1
        Level of output verbosity.

    Returns
    -------
    df : pandas.DataFrame
        Like input, but with the content of the ``"image"`` field updated to
        prevent collisions.
    """
    if not inplace:
        df = df.copy()
    df["_outpath"] = determine_outpath(df)
    dup_outpaths = df[df.duplicated(subset="_outpath")]["_outpath"].unique()
    if len(dup_outpaths) == 0:
        if verbose >= 1:
            print("There are no duplicated output paths in this dataframe")
        df.drop(columns="_outpath", inplace=True)
        return df
    if verbose >= 1:
        print(
            f"There are {len(dup_outpaths)} duplicated output paths in this dataframe"
        )
    for dup_outpath in dup_outpaths:
        is_bad = df["_outpath"] == dup_outpath
        n_bad = sum(is_bad)
        if verbose >= 2:
            print(f"Trying to fix up {n_bad} repetitions of the path {dup_outpath}")
        # 1. Try taking the basename from the URL instead
        # We will try taking just the last bit (photoname.jpg), then including preceding bits
        # of the URL (subsite/photoname.jpg, site/subsite/photoname.jpg)
        subdf = df[is_bad]
        resolved = False
        for k in range(1, 5):
            new_basenames = subdf["url"].apply(
                lambda x: sanitize_filename(
                    "-".join(x.rstrip("/").split("/")[-k:])  # noqa: B023
                )
            )
            # All URL basenames are unique, so we are done
            if len(new_basenames.unique()) != len(new_basenames):
                continue
            df.loc[is_bad, "image"] = new_basenames
            if verbose >= 2:
                print(f"  Using last {k} part(s) of the URL as the basename")
            resolved = True
            break
        if resolved:
            continue
        # 2. If that didn't work, just append _0, _1, ... _N to the image names
        if verbose >= 2:
            print("  Appending a suffix to the basename to prevent collisions")
        new_basenames = subdf.apply(row2basename, axis=1)
        new_basenames = (
            new_basenames.apply(lambda x: os.path.splitext(x)[0])
            + "_"
            + pd.Series([str(x) for x in range(n_bad)], index=new_basenames.index)
            + new_basenames.apply(lambda x: os.path.splitext(x)[1])
        )
        df.loc[is_bad, "image"] = new_basenames
    df.drop(columns="_outpath", inplace=True)
    return df
