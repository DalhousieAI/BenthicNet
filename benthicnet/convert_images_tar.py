#!/usr/bin/env python

"""
Convert images in a tarball into smaller JPEGs, in a new tarball.
"""

import copy
import datetime
import os
import pathlib
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
import PIL.Image
from tqdm.auto import tqdm

import benthicnet.io
import benthicnet.utils
from benthicnet import __meta__

VALID_IMG_EXTENSIONS = (".bmp", ".cr2", ".jpeg", ".jpg", ".png", ".tif", ".tiff")


def shrink_image_by_length(im, target_len=512, tolerance=0.1, return_is_shrunk=False):
    """
    Shrink an image by reducing the shortest side, preserving aspect ratio.

    Parameters
    ----------
    im : PIL.Image
        Input image
    target_len : int, default=512
        Target image side length.
    tolerance : float, default=0.1
        Fraction of ``target_len`` that the
    return_is_shrunk : bool, optional
        Return ``is_shrunk``, indicating whether the image was shruk.

    Returns
    -------
    im : PIL.Image
        Shrunk image, or pointer to original image if the image did not need to
        be shrunk.
    is_shrunk : bool, optional
        Whether the image was shrunk.
    """
    if min(im.size) <= target_len * (1 + tolerance):
        if return_is_shrunk:
            return im, False
        else:
            return im
    if im.width < im.height:
        new_width = target_len
        new_height = new_width * im.height / im.width
        new_height = int(round(new_height))
    else:
        new_height = target_len
        new_width = new_height * im.width / im.height
        new_width = int(round(new_width))

    im_new = im.resize((new_width, new_height))
    if return_is_shrunk:
        return im_new, True
    else:
        return im_new


def handle_input_dfs(
    df_source,
    df_dest=None,
    inplace=True,
    unique_inputs=False,
    unique_outputs=False,
    verbose=1,
):
    """
    Join source and destination dataframes to make a todo list.

    Parameters
    ----------
    df_source : pandas.DataFrame
        Dataframe, in BenthicNet format, containing details about source images
    df_dest : pandas.DataFrame, optional
        Dataframe, in BenthicNet format, containing metadata about destination
        images. If this is not given, ``df_source`` is used.
    inplace : bool, default=True
        Whether the input dataframes can be changes in place, otherwise a copy
        will be used.
    unique_inputs : bool, default=False
        Whether to only use input images which have unique paths.
    unique_outputs : bool, default=False
        Whether to drop repeated output paths in the destination dataframe.
    verbose : int, default=1
        Verbosity level.

    Returns
    -------
    df_todo : pandas.DataFrame
        The rows in ``df_dest`` which have paths appearing in ``df_source``.
        Columns ``member_source`` and ``member_dest`` added to indicate the
        paths to use in each tarball.
    """
    if df_source is None:
        df_source = df_dest
    if df_source is None:
        raise ValueError("At least one of df_source and df_dest must be given.")
    # Remember whether we started with an outpath column in the input, and
    # should preserve it in the output, or added it ourselves and should strip
    # it away before saving the CSV file
    if df_dest is None:
        given_outpath = "outpath" in df_source.columns
    else:
        given_outpath = "outpath" in df_dest.columns
    if "outpath" not in df_source.columns:
        df_source["outpath"] = benthicnet.io.determine_outpath(df_source)
    if df_dest is None:
        df_dest = df_source
    if not inplace:
        df_dest = df_dest.copy()
    if "outpath" not in df_dest.columns:
        df_dest["outpath"] = benthicnet.io.determine_outpath(df_dest)

    if unique_inputs:
        # Don't try to copy duplicated image paths in the source, since we can't be
        # sure which one is present in the source.
        df_source.drop_duplicates(subset=["outpath"], keep=False, inplace=inplace)

    if unique_outputs:
        # Ensure all the new output paths are unique
        dup_outpaths = df_dest[df_dest.duplicated(subset="outpath")]["outpath"].unique()
        if len(dup_outpaths) > 0:
            raise EnvironmentError(
                f"There are {len(dup_outpaths)} duplicated output paths."
            )

    # Align source and destination dataframes
    df_todo_parts = []
    if "url" not in df_dest.columns or any(pd.isna(df_dest["url"])):
        if verbose >= 1:
            print("Aligning records by outpath")
        df_todo1 = pd.merge(
            df_source[["outpath"]],
            df_dest
            if "url" not in df_dest.columns
            else df_dest[pd.isna(df_dest["url"])],
            how="inner",
            on="outpath",
            suffixes=("_source", "_dest"),
        )
        df_todo1["member_source"] = df_todo1["outpath"]
        df_todo1.rename(columns={"outpath": "member_dest"}, inplace=True)
        df_todo_parts.append(df_todo1)

    if (
        "url" in df_source.columns
        and "url" in df_dest.columns
        and any(~pd.isna(df_dest["url"]))
    ):
        if verbose >= 1:
            print("Aligning records by URL")
        # Merge the two dataframes together to get common URLs
        df_todo2 = pd.merge(
            df_source[["url", "outpath"]],
            df_dest[~pd.isna(df_dest["url"])],
            how="inner",
            on="url",
            suffixes=("_source", "_dest"),
        )
        df_todo2.rename(
            columns={"outpath_source": "member_source", "outpath_dest": "member_dest"},
            inplace=True,
        )
        df_todo_parts.append(df_todo2)

    df_todo = pd.concat(df_todo_parts)
    if given_outpath:
        df_todo["outpath"] = df_todo["member_dest"]

    return df_todo


def convert_images(
    fname_source,
    fname_dest,
    df_todo,
    output_csv=None,
    jpeg_quality=95,
    target_len=512,
    skip_existing=True,
    error_stream=None,
    missing_stream=None,
    flexible_source_path=True,
    inplace_df=False,
    verbose=1,
    use_tqdm=True,
    print_indent=0,
):
    r"""
    Convert images in one tarball into JPEG format in another.

    Parameters
    ----------
    fname_source : str
        Path to input tarball file.
    fname_dest : str
        Path to output tarball file.
    df_todo : pandas.DataFrame
        DataFrame containing paths of files to convert.
    jpeg_quality : int or float, optional
        Quality to use when converting to JPEG. Default is ``95``.
    target_len : int or None, default=512
        Maximum length of shortest side of the image.
        Set to ``0`` or ``-1`` to disable.
    skip_existing : bool, optional
        Whether to skip existing images within output file ``fname_dest``.
        If ``False``, an error is raised when the destination already exists.
        Default is ``True``.
    error_stream : text_stream or None, optional
        A text stream where errors should be recorded.
        If provided, all images which could not be converted due to an error
        will be written to this stream, separated by ``"\n"`` characters.
    missing_stream : text_stream or None, optional
        A text stream where missing images should be recorded.
        If provided, all images which could not be converted due to an error
        will be written to this stream, separated by ``"\n"`` characters.
    flexible_source_path : bool, default=True
        Whether to also check output paths with/without dataset as a prefix
        directory.
    inplace_df : bool, optional
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
        Like ``df_todo``, but filtered to only include converted files and
        with the ``image`` field changed to the new basename of the output file
        within the tarball, including ``".jpg"`` extension.
    """
    t0 = time.time()

    padding = " " * print_indent
    innerpad = padding + " " * 4

    if target_len is None or target_len == 0 or target_len == -1:
        target_len = None

    if verbose >= 1:
        if target_len:
            len_str = f", length={target_len}px"
        else:
            len_str = ", original size"
        print(
            f"{padding}Converting {len(df_todo):5d} images from tarball {fname_source}"
            f"\n{padding}                                  to {fname_dest}"
            f"\n{padding}into JPEG format (quality={jpeg_quality}%{len_str})",
            flush=True,
        )

    if not inplace_df:
        df_todo = df_todo.copy()

    if verbose != 1:
        use_tqdm = False

    n_already_converted = 0
    n_converted = 0
    n_missing = 0
    n_error = 0

    # Check what already exists so we know when we can skip a file
    os.makedirs(os.path.dirname(fname_dest), exist_ok=True)
    output_contents = {}
    if os.path.exists(fname_dest):
        try:
            with tarfile.open(fname_dest, mode="r") as tar:
                output_contents = tar.getnames()
        except tarfile.ReadError:
            print(f"{padding}Unable to open {fname_dest}.")
            benthicnet.io.delayed_delete(fname_dest)

    is_valid = np.zeros(len(df_todo), dtype=bool)

    t1 = time.time()

    with tarfile.open(fname_source, mode="r") as tar_in:
        input_contents = tar_in.getnames()

        for i_row, (index, row) in enumerate(
            tqdm(
                df_todo.iterrows(),
                total=len(df_todo),
                disable=not use_tqdm,
                desc="Images",
            )
        ):
            # Progress report
            if i_row > n_error and (
                verbose >= 3
                or (verbose >= 1 and not use_tqdm and (i_row <= 5 or i_row % 100 == 0))
            ):
                t_elapsed = time.time() - t1
                if n_converted > 0:
                    t_remain = t_elapsed / n_converted * (len(df_todo) - i_row)
                else:
                    t_remain = t_elapsed / i_row * (len(df_todo) - i_row)
                print(
                    "{}Processed {:4d}/{} urls ({:6.2f}%) in {} (approx. {} remaining)"
                    "".format(
                        padding,
                        i_row,
                        len(df_todo),
                        100 * i_row / len(df_todo),
                        datetime.timedelta(seconds=t_elapsed),
                        datetime.timedelta(seconds=t_remain),
                    ),
                    flush=True,
                )

            # Change the file extension of the output file
            destination = row["member_dest"]
            ext = os.path.splitext(destination)[1]
            if ext.lower() in VALID_IMG_EXTENSIONS:
                destination = os.path.splitext(destination)[0] + ".jpg"
            else:
                destination = destination + ".jpg"

            # Check if we already did this file
            if destination in output_contents:
                if not skip_existing:
                    raise EnvironmentError(
                        f"Destination {destination} already exists within {fname_dest}"
                    )
                if verbose >= 3:
                    print(f"{innerpad}Output already exists at {destination}")
                n_already_converted += 1
                is_valid[i_row] = True
                # Update this row's image column to have the correct extension
                df_todo.at[index, "member_dest"] = destination
                df_todo.at[index, "image"] = os.path.basename(destination)
                continue

            # Process the file
            with tempfile.TemporaryDirectory() as dir_tmp:
                fname_in = row["member_source"]
                fname_tmp_new = os.path.join(dir_tmp, os.path.basename(destination))
                if verbose >= 4:
                    print(
                        f"{innerpad}  Converting {destination} to JPEG {fname_tmp_new}"
                    )
                # Check if the input file name needs to be tweaked
                if fname_in not in input_contents and flexible_source_path:
                    # Check to see if the input exists if we drop the dataset name
                    # prefix on the path
                    fname_in2 = "/".join(fname_in.split("/")[1:])
                    if fname_in2 in input_contents:
                        fname_in = fname_in2
                if fname_in not in input_contents:
                    if verbose >= 2:
                        print(f"{innerpad}  Missing input {fname_in} in {fname_source}")
                    n_missing += 1
                    if missing_stream:
                        missing_stream.write(fname_in + "\n")
                    continue
                # Extract file contents
                try:
                    im = PIL.Image.open(tar_in.extractfile(fname_in))
                    # Do the conversion
                    if target_len:
                        im, is_shrunk = shrink_image_by_length(
                            im, target_len=target_len, return_is_shrunk=True
                        )
                    else:
                        is_shrunk = False
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    quality = (
                        "keep"
                        if not is_shrunk and im.format == "JPEG"
                        else jpeg_quality
                    )
                    im.save(fname_tmp_new, quality=quality)
                except BaseException as err:
                    if verbose >= 0:
                        print("Error while handling: {}".format(row["url"]))
                        print(err)
                    n_error += 1
                    if error_stream:
                        error_stream.write(fname_in + "\n")
                    continue
                # Add to tarball
                if verbose >= 4:
                    print(
                        f"{innerpad}  Adding {fname_tmp_new} to archive as {destination}"
                    )
                with tarfile.open(fname_dest, mode="a") as tar_out:
                    tar_out.add(fname_tmp_new, arcname=destination)
                    output_contents = tar_out.getnames()
                n_converted += 1

            # Record that this row was successfully converted
            is_valid[i_row] = True
            # Update this row's image column to have the correct extension
            df_todo.at[index, "member_dest"] = destination
            df_todo.at[index, "image"] = os.path.basename(destination)

    if verbose >= 1:
        print(
            "{}Finished processing {} images in {}.".format(
                padding, len(df_todo), datetime.timedelta(seconds=time.time() - t0)
            )
        )
        messages = []
        if n_already_converted > 0:
            s = ""
            if n_already_converted == len(df_todo):
                s = "All"
            elif n_already_converted == 1:
                s = "There was"
            else:
                s = "There were"
            s += " {} image{} already converted.".format(
                n_already_converted,
                "" if n_already_converted == 1 else "s",
            )
            messages.append(s)
        if n_error > 0:
            messages.append(
                "There {} {} conversion error{}.".format(
                    "was" if n_error == 1 else "were",
                    n_error,
                    "" if n_error == 1 else "s",
                )
            )
        if n_missing > 0:
            messages.append(f"There were {n_missing} missing input images.")
        if n_converted > 0:
            s = ""
            if n_converted == len(df_todo):
                s = "All {} images were converted.".format(n_converted)
            else:
                s = "The remaining {} image{} converted.".format(
                    n_converted,
                    " was" if n_converted == 1 else "s were",
                )
            messages.append(s)
        if messages:
            print(padding + " ".join(messages), flush=True)

    # Remove invalid rows
    df_todo = df_todo.iloc[is_valid]
    return df_todo


def convert_images_by_dataset(
    tar_dir_source,
    output_dir,
    df_source,
    df_dest=None,
    skip_existing=True,
    inplace_df=True,
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
    tar_dir_source : str
        Path to directory containing input tarballs.
    output_dir : str
        Path to output directory.
    df_source : pandas.DataFrame
        Input DataFrame, containing mappings from URLs to image paths.
    df_dest : pandas.DataFrame
        Destination DataFrame, listing files to process.
    skip_existing : bool, optional
        Whether to silently skip converting files for which the destination
        already exist. Otherwise an error is raised. Default is ``True``.
    inplace_df : bool, optional
        Whether operations on dataframes can be performed in place.
        Default is ``True``.
    i_proc : int or None, optional
        Run on only a subset of the datasets in the CSV file. If ``None``
        (default), the entire dataset will be converted by this process.
        Otherwise, `n_proc` must also be set, and ``1/n_proc``-th of the
        datasets will be processed.
    n_proc : int or None, optional
        Number of processes being run in parallel. Default is ``None``.
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
    """
    padding = " " * print_indent

    t0 = time.time()

    if df_source is None and df_dest is None:
        raise ValueError("At least one of df_source and df_dest must be given.")

    if (i_proc is not None and n_proc is None) or (
        i_proc is None and n_proc is not None
    ):
        raise ValueError(
            "Both i_proc and n_proc must be defined when partitioning the CSV file."
        )

    if verbose >= 2 and not skip_existing:
        print("Warning: Existing outputs will result in an error.", flush=True)

    df_todo = handle_input_dfs(df_source, df_dest, inplace=inplace_df, verbose=verbose)

    # Sanitise dataset names
    df_todo["dataset"] = benthicnet.io.sanitize_filename_series(df_todo["dataset"])
    # Get source tarball file names
    df_todo["_tarball_source"] = (
        df_todo["member_source"].apply(lambda x: x.split("/")[0]) + ".tar"
    )

    # Create mapping from unique datasets to rows which bear that dataset
    dataset2idx = benthicnet.utils.unique_map(df_todo["dataset"])
    datasets_to_process = sorted(dataset2idx)

    if n_proc:
        partition_size = len(dataset2idx) / n_proc
        i_proc = 0 if i_proc == n_proc else i_proc
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

    using_tqdm = False

    for i_dataset, dataset in enumerate(
        tqdm(datasets_to_process, desc="Datasets", disable=not using_tqdm)
    ):
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

        subdf = df_todo.iloc[dataset2idx[dataset]]
        fname_dest = os.path.join(output_dir, "tar", dataset + ".tar")

        sourcedataset2subidx = benthicnet.utils.unique_map(subdf["_tarball_source"])

        error_fname = os.path.join(output_dir, "errors", dataset + ".log")
        os.makedirs(os.path.dirname(error_fname), exist_ok=True)
        missing_fname = os.path.join(output_dir, "missing", dataset + ".log")
        os.makedirs(os.path.dirname(missing_fname), exist_ok=True)

        # Check if any source tarball exists for this dataset. If not, we won't
        # clobber the error log file.
        for fname_source in sourcedataset2subidx:
            fname_source_full = os.path.join(tar_dir_source, fname_source)
            if os.path.isfile(fname_source_full):
                break
        else:
            if verbose >= 2:
                print(
                    f"{padding}No source tarballs available for {dataset}",
                    flush=True,
                )
            continue

        outdfs = []
        with open(error_fname, "w") as error_stream, open(
            missing_fname, "w"
        ) as missing_stream:
            for fname_source in sourcedataset2subidx:
                fname_source_full = os.path.join(tar_dir_source, fname_source)
                if not os.path.isfile(fname_source_full):
                    if verbose >= 2:
                        print(
                            f"{padding}Missing tarball: {fname_source_full}\n",
                            flush=True,
                        )
                    continue
                outdfs.append(
                    convert_images(
                        fname_source_full,
                        fname_dest,
                        subdf.iloc[sourcedataset2subidx[fname_source]],
                        skip_existing=skip_existing,
                        error_stream=error_stream,
                        missing_stream=missing_stream,
                        verbose=verbose - 1,
                        use_tqdm=use_tqdm,
                        print_indent=print_indent + 4,
                        **kwargs,
                    )
                )
            empty_error_log = error_stream.tell() == 0
            missing_error_log = missing_stream.tell() == 0

        if empty_error_log:
            # Delete empty file
            os.remove(error_fname)
        if missing_error_log:
            # Delete empty file
            os.remove(missing_fname)

        if len(outdfs) > 0:
            # Merge output dataframe
            outdf = pd.concat(outdfs)
            # Drop or rename working columns
            outdf.drop(
                columns=["member_source", "_tarball_source", "member_dest"],
                inplace=True,
            )
            # Save CSV output
            csv_fname = os.path.join(output_dir, "csv", dataset + ".csv")
            os.makedirs(os.path.dirname(csv_fname), exist_ok=True)
            if verbose >= 2:
                print(f"{padding}Saving as {csv_fname}")
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


def convert_images_by_dataset_from_csv(
    input_dir,
    output_dir,
    csv_fname=None,
    align_input_csv=False,
    skip_existing=True,
    verbose=1,
    use_tqdm=True,
    **kwargs,
):
    """
    Download all images from a CSV file into tarfiles, one for each dataset.

    Parameters
    ----------
    input_dir : str
        Path to input directory.
    output_dir : str
        Path to output directory.
    csv_fname : str, optional
        Path to CSV file containing the todo list. If this is not given, every
        file in the input CSV files will be processed.
    align_input_csv : str or bool, optional
        Input CSV to align the todo list CSV against. Alignment is done by URL
        where possible, and otherwise by output path.
        If this argument is given without a file, the CSV files will be taken
        from the ``"csv"`` subdirectory of ``input_dir``.
    skip_existing : bool, optional
        Whether to skip downloading files for which the destination already
        exist. Default is ``True``.
    verbose : int, optional
        Verbosity level. Default is ``1``.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if stdout is being
        written to a file. Default is ``True``.
    **kwargs
        Additional arguments as per :func:`convert_images_by_dataset`.
    """
    t0 = time.time()

    if not csv_fname and not align_input_csv:
        raise ValueError("At least one of csv_fname and align_input_csv must be given.")

    if verbose >= 1:
        print(
            f"Will convert images from {input_dir} into smaller JPEGs"
            f"\nInto directory: {output_dir}"
        )
        if skip_existing:
            print("Existing outputs will be skipped.")
        else:
            print("Existing outputs will generate an error.")

    # Load input CSV to align against
    if isinstance(align_input_csv, (str, pathlib.PurePath)):
        if verbose >= 3:
            print(
                "Reading CSV file {} ({})...".format(
                    align_input_csv, benthicnet.io.file_size(align_input_csv)
                ),
                flush=True,
            )
        df_source = pd.read_csv(align_input_csv, low_memory=False)
        if verbose >= 2:
            print(
                "Loaded CSV file {} in {:.1f} seconds".format(
                    align_input_csv, time.time() - t0
                ),
                flush=True,
            )

    elif align_input_csv:
        input_csvs = sorted(os.listdir(os.path.join(input_dir, "csv")))
        input_dfs = []

        using_tqdm = verbose == 1 and use_tqdm

        for csv_file_input_i in tqdm(
            input_csvs, desc="Input CSV files", disable=not using_tqdm
        ):

            if os.path.splitext(csv_file_input_i)[1].lower() != ".csv":
                continue
            fname = os.path.join(input_dir, "csv", csv_file_input_i)
            if verbose >= 3:
                print(
                    "Reading CSV file {} ({})...".format(
                        fname, benthicnet.io.file_size(fname)
                    ),
                    flush=True,
                )
            input_dfs.append(pd.read_csv(fname, low_memory=False))
            if verbose >= 2:
                print(
                    "Loaded CSV file {} in {:.1f} seconds".format(
                        fname, time.time() - t0
                    ),
                    flush=True,
                )

        df_source = pd.concat(input_dfs)

    else:
        df_source = None

    # Destination/TODO CSV file
    if csv_fname is None:
        df_todo = None
    else:
        if verbose >= 1:
            print(
                "Reading CSV file ({})...".format(benthicnet.io.file_size(csv_fname)),
                flush=True,
            )
        df_todo = benthicnet.io.read_csv(csv_fname)
        if verbose >= 1:
            print(
                "Loaded CSV file in {:.1f} seconds".format(time.time() - t0), flush=True
            )

    if os.path.isdir(os.path.join(input_dir, "tar")):
        input_dir = os.path.join(input_dir, "tar")

    ret = convert_images_by_dataset(
        input_dir,
        output_dir,
        df_source,
        df_todo,
        skip_existing=skip_existing,
        verbose=verbose,
        use_tqdm=use_tqdm,
        **kwargs,
    )

    if verbose >= 1:
        print(
            "Total runtime: {}".format(datetime.timedelta(seconds=time.time() - t0)),
            flush=True,
        )
    return ret


def get_parser():
    """
    Build CLI parser for converting a tarfile of images into smaller JPEGs.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import sys
    import textwrap

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Convert files from tarball of images to another, by matching on URL.",
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
        "input_dir",
        type=str,
        help="Directory containing tar and csv sources.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory in which output tarballs and csv files with be placed.",
    )
    parser.add_argument(
        "--csv-todo",
        dest="csv_fname",
        type=str,
        help="Path to CSV file specifying images to process, in the BenthicNet format.",
    )
    parser.add_argument(
        "--align-input-csv",
        type=str,
        nargs="?",
        default=False,
        const=True,
        help=textwrap.dedent(
            """
            Input CSV to align todo list CSV against. Alignment is done by URL
            where possible, and otherwise by output path.
            If this argument is given without a file, the collection of CSV
            files from INPUT_DIR will be used.
        """
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=float,
        default=95,
        help="Quality to use when converting to JPEG. Default is %(default)s.",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=512,
        help=textwrap.dedent(
            """
            Minimum image side length. Default is %(default)s.
            Set to 0 or -1 to disable.
        """
        ),
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

    return convert_images_by_dataset_from_csv(**kwargs)


if __name__ == "__main__":
    main()
