#!/usr/bin/env python

"""
Move files from tarball of images to another, by matching on URL.
"""

import os
import tarfile
import time

import pandas as pd
from tqdm.auto import tqdm

import benthicnet.utils
from benthicnet import __meta__


def tar2tar(tar_dir_source, tar_dir_dest, csv_source, csv_dest, verbose=1):
    """
    Move files from tarball of images to another, by matching on URL.

    Tarfile names and member paths are generated using
    :func:`benthicnet.utils.determine_outpath`.
    Duplicated member paths in the source are dropped. Destination member paths
    must be unique.

    Parameters
    ----------
    tar_dir_source : str
        Directory containing source tarballs.
    tar_dir_dest : str
        Directory in which to place output tarballs.
    csv_source : str
        Path to CSV file listing source files and their URLs.
    csv_dest : str
        Path to CSV file listing destination files and their URLs.
    verbose : bool, default=1
        Verbosity level.
    """
    df_source = pd.read_csv(csv_source)
    df_dest = benthicnet.utils.read_csv(csv_dest, expect_datetime=False)

    # Determine output paths
    df_source["_outpath"] = benthicnet.utils.determine_outpath(df_source)
    df_dest["_outpath"] = benthicnet.utils.determine_outpath(df_dest)

    # Convert output paths into tarfile name and member path within the tarball
    df_source["_outtar"] = (
        df_source["_outpath"].apply(lambda x: x.split("/")[0]) + ".tar"
    )
    df_source["_outinner"] = df_source["_outpath"].apply(
        lambda x: "/".join(x.split("/")[1:])
    )
    df_dest["_outtar"] = df_dest["_outpath"].apply(lambda x: x.split("/")[0]) + ".tar"
    df_dest["_outinner"] = df_dest["_outpath"].apply(
        lambda x: "/".join(x.split("/")[1:])
    )

    # Ensure all the new output paths are unique
    dup_outpaths = df_dest[df_dest.duplicated(subset="_outpath")]["_outpath"].unique()
    if len(dup_outpaths) > 0:
        raise EnvironmentError(
            f"There are {len(dup_outpaths)} duplicated output paths."
        )

    # Don't try to copy duplicated image paths in the source, since we can't be
    # sure which one is present in the source.
    df_source.drop_duplicates(subset=["_outpath"], keep=False, inplace=True)

    # Merge the two dataframes together to get common URLs
    df_todo = pd.merge(
        df_source, df_dest, how="inner", on="url", suffixes=("_source", "_dest")
    )

    # Iterate over every output file
    for dest_tar_fname in tqdm(
        df_todo["_outtar_dest"].unique(), desc="Destinations", disable=verbose < 1
    ):
        dest_tar_full_path = os.path.join(tar_dir_dest, dest_tar_fname)
        if verbose >= 1:
            print(f"Opening {dest_tar_full_path} to write to")
        if os.path.exists(dest_tar_full_path):
            print(f"Will overwrite existing file {dest_tar_full_path}")
            t_wait = 3
            print("Overwriting in", end="", flush=True)
            for i in range(t_wait // 1):
                print(" {}...".format(t_wait - i), end="", flush=True)
                time.sleep(1)
            print(" Overwriting!")
            if t_wait % 1 > 0:
                time.sleep(t_wait % 1)
        os.makedirs(tar_dir_dest, exist_ok=True)

        with tarfile.open(dest_tar_full_path, "w") as tar_dest:
            # List the sources which need to map to this destination
            source_tar_fnames = df_todo.loc[
                df_todo["_outtar_dest"] == dest_tar_fname, "_outtar_source"
            ].unique()
            for source_tar_fname in tqdm(
                source_tar_fnames,
                desc="Sources",
                disable=verbose < 1 or len(source_tar_fnames) <= 1,
            ):
                source_tar_full_path = os.path.join(tar_dir_source, source_tar_fname)
                if verbose >= 1:
                    print(f"Opening {source_tar_full_path} to copy from")
                subdf_todo = df_todo[
                    (df_todo["_outtar_dest"] == dest_tar_fname)
                    & (df_todo["_outtar_source"] == source_tar_fname)
                ]
                with tarfile.open(source_tar_full_path, "r") as tar_source:
                    contents = set(tar_source.getnames())
                    if verbose >= 2:
                        print(
                            f"There are {len(contents)} files in {source_tar_fname}."
                            "Trying to copy {len(subdf_todo)} files."
                        )
                    for _, row in tqdm(
                        subdf_todo.iterrows(),
                        total=len(subdf_todo),
                        desc="Files",
                        disable=verbose < 1,
                    ):
                        if row["_outinner_source"] not in contents:
                            if verbose >= 2:
                                print(
                                    f"{row['_outinner_source']} is missing from source"
                                )
                            continue
                        # print(f"Copying {row['_outinner_source']} -> {row['_outinner_dest']}")
                        member = tar_source.getmember(row["_outinner_source"])
                        member.name = row["_outinner_dest"]
                        tar_dest.addfile(
                            member, tar_source.extractfile(row["_outinner_source"])
                        )


def get_parser():
    """
    Build CLI parser for downloading a dataset into tarfiles by dataset.

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
        description="Move files from tarball of images to another, by matching on URL.",
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
        "tar_dir_source",
        type=str,
        help="Directory containing source tarballs.",
    )
    parser.add_argument(
        "tar_dir_dest",
        type=str,
        help="Directory in which output tarballs with be placed.",
    )
    parser.add_argument(
        "csv_source",
        type=str,
        help="Input CSV file, in the BenthicNet format.",
    )
    parser.add_argument(
        "csv_dest",
        type=str,
        help="Destination metadata CSV file, in the BenthicNet format.",
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

    return tar2tar(**kwargs)


if __name__ == "__main__":
    main()
