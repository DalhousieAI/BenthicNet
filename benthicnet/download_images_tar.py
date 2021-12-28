#!/usr/bin/env python

"""
Downloading BenthicNet images from CSV file to tarfiles.
"""

import datetime
import functools
import os
import sys
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
import PIL.Image
import requests
import tqdm

from benthicnet import __meta__, utils


def download_images(
    df,
    tar_fname,
    convert_to_jpeg=False,
    jpeg_quality=95,
    skip_existing=True,
    check_image=True,
    error_stream=None,
    inplace=True,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
):
    r"""
    Download images into a tarball.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of images to download.
    tar_fname : str
        Path to output tarball file.
    convert_to_jpeg : bool, optional
        Whether to convert PNG images to JPEG format. Default is ``False``.
    jpeg_quality : int or float, optional
        Quality to use when converting to JPEG. Default is ``95``.
    skip_existing : bool, optional
        Whether to skip existing outputs. If ``False``, an error is raised when
        the destination already exists. Default is ``True``.
    check_image : bool, default=True
        Whether to check the image can be opened with PIL. If ``True``,
        downloads which can not be opened are not added to the tarball.
    error_stream : text_stream or None, optional
        A text stream where errors should be recorded. If provided, all URLs
        which could not be downloaded due to an error will be written to this
        stream, separated by ``"\n"`` characters.
    inplace : bool, optional
        Whether operations on `df` can be performed in place. Default is
        ``True``.
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
        print(
            padding
            + "Downloading {} images into tarball {}".format(len(df), tar_fname),
            flush=True,
        )

    if not inplace:
        df = df.copy()

    if verbose >= 3:
        print(
            padding + "Sanitizing dataset, site, image and url fields",
            flush=True,
        )
    df["dataset"] = utils.sanitize_filename_series(df["dataset"])
    df["site"] = utils.sanitize_filename_series(df["site"])
    df["image"] = utils.sanitize_filename_series(df["image"])
    df["url"] = df["url"].str.strip()

    if verbose != 1:
        use_tqdm = False

    n_already_downloaded = 0
    n_download = 0
    n_error = 0

    os.makedirs(os.path.dirname(tar_fname), exist_ok=True)
    contents = {}
    if os.path.exists(tar_fname):
        try:
            with tarfile.open(tar_fname, mode="r") as tar:
                contents = tar.getnames()
        except tarfile.ReadError:
            t_wait = 5
            if verbose >= 1:
                print(
                    "{}Unable to open {}.\n{}File {} will be deleted in {} seconds..."
                    "".format(padding, tar_fname, padding, tar_fname, t_wait),
                )
                print("{}Deleting in".format(padding), end="", flush=True)
                for i in range(t_wait // 1):
                    print(" {}...".format(t_wait - i), end="", flush=True)
                    time.sleep(1)
                print(" Deleting!")
                if t_wait % 1 > 0:
                    time.sleep(t_wait % 1)
            else:
                time.sleep(t_wait)
            os.remove(tar_fname)
            if verbose >= 1:
                print(
                    "{}Existing file {} deleted".format(padding, tar_fname), flush=True
                )

    t1 = time.time()

    is_valid = np.zeros(len(df), dtype=bool)
    for i_row, (index, row) in enumerate(
        tqdm.tqdm(df.iterrows(), len(df), disable=not use_tqdm)
    ):
        if pd.isna(row["url"]) or row["url"] == "":
            n_error += 1
            if verbose >= 2:
                print(
                    "{}Missing URL for entry\n{}".format(innerpad, row),
                    flush=True,
                )
            if error_stream:
                error_stream.write("{}\n".format(row["url"]))
            continue

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

        destination = row["image"]
        if not destination:
            destination = row["url"].rstrip("/").split("/")[-1]
        ext = os.path.splitext(destination)[1]
        expected_ext = os.path.splitext(row["url"].rstrip("/"))[1]
        if expected_ext and ext.lower() != expected_ext.lower():
            if ext.lower() in {".jpg", ".jpeg"}:
                destination = os.path.splitext(destination)[0] + expected_ext
            else:
                destination = destination + expected_ext
        destination = os.path.join(row["site"], destination)
        ext = os.path.splitext(destination)[1]
        if convert_to_jpeg and ext.lower() not in {".jpg", ".jpeg"}:
            needs_conversion = True
            destination = os.path.splitext(destination)[0] + ".jpg"
        else:
            needs_conversion = False

        if destination in contents:
            if not skip_existing:
                raise EnvironmentError(
                    "Destination {} already exists within {}".format(
                        destination, tar_fname
                    )
                )
            if verbose >= 3:
                print(innerpad + "Already downloaded {}".format(destination))
            n_already_downloaded += 1
        else:
            try:
                r = requests.get(row["url"], stream=True)
            except requests.exceptions.RequestException as err:
                print("Error while handling: {}".format(row["url"]))
                print(err)
                n_error += 1
                if error_stream:
                    error_stream.write(row["url"] + "\n")
                continue
            if r.status_code != 200:
                if verbose >= 1:
                    print(
                        innerpad
                        + "Bad URL (HTTP Status {}): {}".format(
                            r.status_code, row["url"]
                        )
                    )
                n_error += 1
                if error_stream:
                    error_stream.write(row["url"] + "\n")
                continue

            with tempfile.TemporaryDirectory() as dir_tmp:
                if verbose >= 3:
                    print(innerpad + "Downloading {}".format(row["url"]))
                fname_tmp = os.path.join(
                    dir_tmp,
                    os.path.basename(row["url"].rstrip("/")),
                )
                with open(fname_tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1048576):
                        f.write(chunk)
                if verbose >= 4:
                    print(innerpad + "  Wrote to {}".format(fname_tmp))

                # Check the image can be opened with PIL
                if check_image or needs_conversion:
                    try:
                        im = PIL.Image.open(fname_tmp)
                    except BaseException as err:
                        if isinstance(err, KeyboardInterrupt):
                            raise
                        print("Error while handling: {}".format(row["url"]))
                        print(err)
                        n_error += 1
                        if error_stream:
                            error_stream.write(row["url"] + "\n")
                        continue

                if needs_conversion:
                    fname_tmp_new = os.path.join(
                        dir_tmp,
                        os.path.basename(destination),
                    )
                    if verbose >= 4:
                        print(
                            innerpad
                            + "  Converting {} to JPEG {}".format(
                                fname_tmp, fname_tmp_new
                            )
                        )
                    im = im.convert("RGB")
                    im.save(fname_tmp_new, quality=jpeg_quality)
                    fname_tmp = fname_tmp_new

                if verbose >= 4:
                    print(
                        innerpad
                        + "  Adding {} to archive as {}".format(fname_tmp, destination)
                    )
                with tarfile.open(tar_fname, mode="a") as tar:
                    contents = tar.getnames()
                    tar.add(fname_tmp, arcname=destination)
                n_download += 1

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


def download_images_by_dataset(
    df,
    output_dir,
    skip_existing=True,
    i_proc=None,
    n_proc=None,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
    **kwargs,
):
    """
    Download all images from a DataFrame into tarfiles, one for each dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame, contining URLs to download.
    output_dir : str
        Path to output directory.
    skip_existing : bool, optional
        Whether to silently skip downloading files for which the destination
        already exist. Otherwise an error is raised. Default is ``True``.
    i_proc : int or None, optional
        Run on only a subset of the datasets in the CSV file. If ``None``
        (default), the entire dataset will be downloaded by this process.
        Otherwise, `n_proc` must also be set, and ``1/n_proc``-th of the
        datasets will be processed.
    n_proc : int or None, optional
        Number of processes being run. Default is ``None``.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if stdout is being
        written to a file. Default is ``True``.
    print_indent : int, optional
        Amount of whitespace padding to precede print statements.
        Default is ``0``.
    **kwargs
        Additional keword arguments as per :func:`download_images`.

    Returns
    -------
    None
    """
    padding = " " * print_indent

    t0 = time.time()

    if (i_proc is not None and n_proc is None) or (
        i_proc is None and n_proc is not None
    ):
        raise ValueError(
            "Both i_proc and n_proc must be defined when partitioning the CSV file."
        )

    if verbose >= 2 and not skip_existing:
        print("Warning: Existing outputs will result in an error.", flush=True)

    # Sanitise dataset names
    df["dataset"] = utils.sanitize_filename_series(df["dataset"])

    # Create mapping from unique datasets to rows which bear that dataset
    dataset2idx = utils.unique_map(df["dataset"])
    datasets_to_process = sorted(dataset2idx)

    if n_proc:
        partition_size = len(dataset2idx) / n_proc
        i_proc == 0 if i_proc == n_proc else i_proc
        start_idx = round(i_proc * partition_size)
        end_idx = round((i_proc + 1) * partition_size)
        datasets_to_process = datasets_to_process[start_idx:end_idx]

    n_to_process = len(datasets_to_process)

    if verbose >= 1:
        print(
            "{}There are {} datasets in the CSV file.".format(
                padding, len(dataset2idx)
            ),
            flush=True,
        )
        if n_proc:
            print(
                "{}Worker {} of {}. Will process {} datasets:".format(
                    padding, i_proc, n_proc, n_to_process
                )
            )
            print(datasets_to_process)

    using_tqdm = verbose == 1 and use_tqdm

    for i_dataset, dataset in enumerate(
        tqdm.tqdm(datasets_to_process, disable=not using_tqdm)
    ):
        if not n_proc:
            pass

        if verbose >= 1 and not using_tqdm and i_dataset > 0:
            t_elapsed = time.time() - t0
            t_remain = t_elapsed / i_dataset * (n_to_process - i_dataset)
            print(
                "{}Processed {:3d}/{} dataset{} ({:6.2f}%) in {} (approx. {}"
                " remaining)".format(
                    padding,
                    i_dataset,
                    n_to_process,
                    "" if n_to_process == 1 else "s",
                    100 * i_dataset / n_to_process,
                    datetime.timedelta(seconds=t_elapsed),
                    datetime.timedelta(seconds=t_remain),
                )
            )
        if verbose >= 1 and not using_tqdm:
            print(
                '\n{}Processing dataset "{}" ({}/{})'.format(
                    padding, dataset, i_dataset + 1, n_to_process
                ),
                flush=True,
            )

        subdf = df.loc[dataset2idx[dataset]]
        tar_fname = os.path.join(output_dir, "tar", dataset + ".tar")

        error_fname = os.path.join(output_dir, "errors", dataset + ".log")
        os.makedirs(os.path.dirname(error_fname), exist_ok=True)

        with open(error_fname, "w") as file:
            outdf = download_images(
                subdf,
                tar_fname,
                skip_existing=skip_existing,
                error_stream=file,
                verbose=verbose - 1,
                use_tqdm=use_tqdm,
                print_indent=print_indent + 4,
                **kwargs,
            )
            empty_error_log = file.tell() == 0

        if empty_error_log:
            # Delete empty file
            os.remove(error_fname)

        # Save CSV output
        csv_fname = os.path.join(output_dir, "csv", dataset + ".csv")
        os.makedirs(os.path.dirname(csv_fname), exist_ok=True)
        outdf.to_csv(csv_fname, index=False)

    if verbose >= 1:
        if not using_tqdm:
            print()
        print(
            "Processed {} dataset{} in {}".format(
                n_to_process,
                "" if n_to_process == 1 else "s",
                datetime.timedelta(seconds=time.time() - t0),
            ),
            flush=True,
        )
    return


def download_images_by_dataset_from_csv(
    input_csv,
    output_dir,
    *args,
    skip_existing=True,
    verbose=1,
    **kwargs,
):
    """
    Download all images from a CSV file into tarfiles, one for each dataset.

    Parameters
    ----------
    input_csv : str
        Path to CSV file.
    output_dir : str
        Path to output directory.
    skip_existing : bool, optional
        Whether to skip downloading files for which the destination already
        exist. Default is ``True``.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    **kwargs
        Additional arguments as per :func:`download_images_by_dataset`.

    Returns
    -------
    None
    """
    t0 = time.time()

    if verbose >= 1:
        print(
            "Will download images listed in {} into tarfiles by dataset name"
            "\nInto directory: {}".format(
                input_csv,
                output_dir,
            )
        )
        if skip_existing:
            print("Existing outputs will be skipped.")
        else:
            print("Existing outputs will generate an error.")
        print("Reading CSV file ({})...".format(utils.file_size(input_csv)), flush=True)
    df = pd.read_csv(input_csv)
    if verbose >= 1:
        print("Loaded CSV file in {:.1f} seconds".format(time.time() - t0), flush=True)

    ret = download_images_by_dataset(
        df, output_dir, *args, skip_existing=skip_existing, verbose=verbose, **kwargs
    )

    if verbose >= 1:
        print(
            "Total runtime: {}".format(datetime.timedelta(seconds=time.time() - t0)),
            flush=True,
        )
    return ret


def get_parser():
    """
    Build CLI parser for downloading a dataset into tarfiles by dataset.

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
        description="""
            Download all images listed in a CSV file into tarfiles.

            A tarfile (a.k.a. tarball) is created for each dataset. Within
            the tarfile, a directory for each deployment is created.
            Each image that is part of that dataset is downloaded as
            site/image.jpg within the corresponding tarfile named as
            output_dir/dataset.tar.
        """,
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
        "--jpeg",
        dest="convert_to_jpeg",
        action="store_true",
        help="Convert images to JPEG format.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=float,
        default=95,
        help="Quality to use when converting to JPEG. Default is %(default)s.",
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
        "--fail-existing",
        dest="skip_existing",
        action="store_false",
        help=textwrap.dedent(
            """
            Raise an error if any outputs already exist in the target tarfile.
            Default behaviour is to quietly skip processing any existing files.
        """
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=2,
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
    Run command line interface for downloading images to tarfiles by dataset.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return download_images_by_dataset_from_csv(**kwargs)


if __name__ == "__main__":
    main()
