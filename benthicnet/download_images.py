#!/usr/bin/env python

"""
Downloading BenthicNet images from CSV file.
"""

import datetime
import functools
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import tqdm

import benthicnet.io
from benthicnet import __meta__


def download_images_from_dataframe(
    df,
    output_dir,
    skip_existing=True,
    inplace=True,
    delete_partial=True,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
):
    """
    Download all images from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of images to download.
    output_dir : str
        Path to output directory.
    skip_existing : bool, optional
        Whether to skip downloading files for which the destination already
        exist. Default is ``True``.
    inplace : bool, optional
        Whether operations on ``df`` can be performed in place. Default is
        ``True``.
    delete_partial : bool, optional
        Whether to delete partially downloaded files in the event of an error,
        such as running out of disk space or keyboard interrupt.
        Default is ``True``.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if stdout is being
        written to a file. Default is ``True``.
    print_indent : int, optional
        Amount of whitespace padding to precede print statements.
        Default is ``0``.

    Returns
    -------
    pandas.DataFrame
        Like `df`, but with the ``image`` column changed to the exact basename of
        the output file within the tarball, including extension. Only entries
        which could be downloaded are included; URLs which could not be found
        are omitted.
    """
    t0 = time.time()

    padding = " " * print_indent
    innerpad = padding + " " * 4
    if verbose >= 1:
        print(padding + "Downloading {} images".format(len(df)), flush=True)

    if not inplace:
        df = df.copy()

    if verbose >= 3:
        print(
            padding + "Sanitizing fields used to build filenames",
            flush=True,
        )
    df["dataset"] = benthicnet.io.sanitize_filename_series(df["dataset"])
    df["site"] = benthicnet.io.sanitize_filename_series(df["site"])
    df["image"] = df.apply(benthicnet.io.row2basename, axis=1)
    df["url"] = df["url"].str.strip()

    if verbose != 1:
        use_tqdm = False

    n_already_downloaded = 0
    n_download = 0
    n_error = 0

    t1 = time.time()

    is_valid = np.zeros(len(df), dtype=bool)
    for i_row, (index, row) in enumerate(
        tqdm.tqdm(df.iterrows(), total=len(df), disable=not use_tqdm)
    ):
        if pd.isna(row["url"]) or row["url"] == "":
            n_error += 1
            if verbose >= 2:
                print(
                    "{}Missing URL for entry\n{}".format(innerpad, row),
                    flush=True,
                )
            continue

        destination = row["image"]
        destination = os.path.join(
            output_dir,
            row["dataset"],
            row["site"],
            destination,
        )
        if i_row > n_error and (
            verbose >= 3
            or (verbose >= 1 and not use_tqdm and (i_row <= 5 or i_row % 100 == 0))
        ):
            t_elapsed = time.time() - t1
            if n_download > 0:
                t_remain = t_elapsed / n_download * (len(df) - i_row)
            else:
                t_remain = t_elapsed / i_row * (len(df) - i_row)
            print(
                "{}Processed {:4d}/{} urls ({:6.2f}%) in {} (approx. {} remaining)"
                "".format(
                    padding,
                    i_row,
                    len(df),
                    100 * i_row / len(df),
                    datetime.timedelta(seconds=t_elapsed),
                    datetime.timedelta(seconds=t_remain),
                ),
                flush=True,
            )

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if skip_existing and os.path.isfile(destination):
            n_already_downloaded += 1
            if verbose >= 3:
                print(
                    "{}Skipping download of {}\n"
                    "{}Destination exists: {}".format(
                        innerpad, row["url"], innerpad, destination
                    ),
                    flush=True,
                )
        else:
            if verbose >= 2:
                print(
                    "{}Downloading {} to {}".format(innerpad, row["url"], destination),
                    flush=True,
                )
            try:
                _, headers = urllib.request.urlretrieve(
                    row["url"].strip().replace(" ", "%20"), filename=destination
                )
                n_download += 1
            except Exception as err:
                n_error += 1
                print(
                    "{}An error occured while processing {}".format(
                        innerpad, row["url"]
                    )
                )
                if os.path.isfile(destination) and delete_partial:
                    print("{}Deleting partial file {}".format(innerpad, destination))
                    os.remove(destination)
                if isinstance(
                    err,
                    (
                        ValueError,
                        urllib.error.ContentTooShortError,
                        urllib.error.HTTPError,
                        urllib.error.URLError,
                    ),
                ):
                    print(err)
                    continue
                raise

        # Record that this row was successfully downloaded
        is_valid[i_row] = True
        # Update this row's image field to be the actual destination basename
        df.at[index, "image"] = os.path.basename(destination)

    if verbose >= 1:
        print(
            "{}Finished processing {} images in {}.".format(
                padding, len(df), datetime.timedelta(seconds=time.time() - t0)
            )
        )
        messages = []
        if n_already_downloaded > 0:
            s = ""
            if n_already_downloaded == len(df):
                s = "All"
            elif n_already_downloaded == 1:
                s = "There was"
            else:
                s = "There were"
            s += " {} image{} already downloaded.".format(
                n_already_downloaded,
                "" if n_already_downloaded == 1 else "s",
            )
            messages.append(s)
        if n_error > 0:
            messages.append(
                "There {} {} download error{}.".format(
                    "was" if n_error == 1 else "were",
                    n_error,
                    "" if n_error == 1 else "s",
                )
            )
        if n_download > 0:
            s = ""
            if n_download == len(df):
                s = "All {} images were downloaded.".format(n_download)
            else:
                s = "The remaining {} image{} downloaded.".format(
                    n_download,
                    " was" if n_download == 1 else "s were",
                )
            messages.append(s)
        if messages:
            print(padding + " ".join(messages), flush=True)

    # Remove invalid rows
    return df.loc[is_valid]


def download_images_from_csv(
    input_csv,
    output_dir,
    *args,
    output_csv=None,
    skip_existing=True,
    verbose=1,
    i_proc=None,
    n_proc=None,
    **kwargs,
):
    """
    Download all images from a CSV file.

    Parameters
    ----------
    input_csv : str
        Path to CSV file.
    output_dir : str
        Path to output directory.
    output_csv : str, optional
        Path to output CSV file, which will have rows containing invalid URLs
        dropped and columns sanitized to match the output file path.
        If omitted, no output CSV is generated.
    skip_existing : bool, optional
        Whether to skip downloading files for which the destination already
        exist. Default is ``True``.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    i_proc : int or None, optional
        Run on only a partition of the CSV file. If ``None`` (default), the
        entire dataset will be downloaded by this process. Otherwise, ``n_proc``
        must also be set.
    n_proc : int or None, optional
        Number of partitions being run. Default is ``None``.
    **kwargs : optional
        Additional arguments as per :func:`download_images_from_dataframe``.

    Returns
    -------
    None
    """
    t0 = time.time()
    if (i_proc is not None and n_proc is None) or (
        i_proc is None and n_proc is not None
    ):
        raise ValueError(
            "Both i_proc and n_proc must be defined when partitioning the CSV file."
        )
    skiprows = []
    if n_proc:
        part_str = "(part {} of {})".format(i_proc, n_proc)
        n_lines = benthicnet.io.count_lines(input_csv) - 1
        partition_size = n_lines / n_proc
        i_proc = 0 if i_proc == n_proc else i_proc
        start_idx = round(i_proc * partition_size)
        end_idx = round((i_proc + 1) * partition_size)
        skiprows = list(range(1, 1 + start_idx)) + list(range(1 + end_idx, 1 + n_lines))

    if verbose >= 1:
        print(
            "Will download {} images {}listed in {}".format(
                "all" if not n_proc else end_idx - start_idx,
                "" if not n_proc else part_str + " ",
                input_csv,
            )
        )
        print("To output directory {}".format(output_dir))
        if skip_existing:
            print("Existing outputs will be skipped.")
        else:
            print("Existing outputs will be overwritten.")
        if output_csv:
            print(
                "An output CSV file containing only the valid URLs will be"
                " created at {}".format(output_csv)
            )
            if os.path.isfile(output_csv):
                print("The existing file {} will be overwritten.".format(output_csv))
        print(
            "Reading CSV file ({})...".format(benthicnet.io.file_size(input_csv)),
            flush=True,
        )
    df = benthicnet.io.read_csv(input_csv, skiprows=skiprows)
    if verbose >= 1:
        print("Loaded CSV file in {:.1f} seconds".format(time.time() - t0), flush=True)
    output_df = download_images_from_dataframe(
        df,
        output_dir,
        *args,
        skip_existing=skip_existing,
        verbose=verbose,
        **kwargs,
    )
    if output_csv is not None:
        if verbose >= 1:
            print("Saving valid CSV rows to {}".format(output_csv))
        output_df.to_csv(output_csv, index=False)

    if verbose >= 1:
        print(
            "Total runtime: {}".format(datetime.timedelta(seconds=time.time() - t0)),
            flush=True,
        )
    return


def get_parser():
    """
    Build CLI parser for downloading BenthicNet image dataset.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import textwrap

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Download all images listed in a BenthicNet CSV file",
        add_help=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s {version}".format(version=__meta__.version),
        help="Show program's version number and exit.",
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Input CSV file, in the BenthicNet format.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Root directory for downloaded images.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Output CSV file.",
    )
    parser.add_argument(
        "--no-progress-bar",
        dest="use_tqdm",
        action="store_false",
        help="Disable tqdm progress bar.",
    )
    parser.add_argument(
        "--nproc",
        dest="n_proc",
        metavar="NPROC",
        type=int,
        help="Number of processing partitions being run.",
    )
    parser.add_argument(
        "--iproc",
        "--proc",
        dest="i_proc",
        metavar="IPROC",
        type=int,
        help="Partition index for this process.",
    )
    parser.add_argument(
        "--clobber",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing outputs instead of skipping their download.",
    )
    parser.add_argument(
        "--keep-on-error",
        dest="delete_partial",
        action="store_false",
        help="Keep partially downloaded files in the event of an error.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help=textwrap.dedent(
            """
            Increase the level of verbosity of the program. This can be
            specified multiple times, each will increase the amount of detail
            printed to the terminal. The default verbosity level is %(default)s.
        """
        ),
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="count",
        default=0,
        help=textwrap.dedent(
            """
            Decrease the level of verbosity of the program. This can be
            specified multiple times, each will reduce the amount of detail
            printed to the terminal.
        """
        ),
    )
    return parser


def main():
    """
    Run command line interface for downloading images.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return download_images_from_csv(**kwargs)


if __name__ == "__main__":
    main()
