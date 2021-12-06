#!/usr/bin/env python

"""
Tools for subsampling datasets.
"""

import datetime
import os
import sys
import time

import haversine
import numpy as np
import pandas as pd
import tqdm

from benthicnet import __meta__, utils


def subsample_distance(
    df, threshold=2.5, method="closest", verbose=0, include_last=None
):
    """
    Subsample data based on distance between records.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to process, with latitude and longitude columns. Data must be
        ordered with sequential samples in sequence.
    threshold : float, optional
        Minimum distance between samples, in metres. Default is ``2.5``.
    method : {"closest", "threshold"}, optional
        Which method to use when subsampling. Default is ``"closest"``.

        threshold
            A row must be at least ``threshold`` metres from the last included
            row to be included.
        closest
            The distance ``threshold`` is treated as a target distance.
            The closest row to the target distance from the last included row
            is included. This may be closer or further away from the last row
            than ``threshold``.

    verbose : int, optional
        Verbosity level. Default is ``0``.
    include_last : bool or None, optional
        Whether to include the last row regardless of its distance from the
        preceding included row.
        If ``False``, the last row in `df` is included only if it is further than
        ``threshold`` metres from the preceding included row.
        If ``None``, the last row in `df` is included only if it is further than
        ``threshold / 2`` metres from the preceding included row.
        If ``True``, the last row in `df` is always included.
        Default is ``None``.

    Returns
    -------
    pandas.DataFrame
        Dataframe like `df`, but with only a subset of the rows included.
    """
    # Remove entries without latitude/longitude
    df = df[df["latitude"].notna() & df["longitude"].notna()]
    # Only two entries, so return both
    if len(df) < 3:
        return df
    # Measure distance between entries using Haversine method
    distances = haversine.haversine_vector(
        np.stack([df["latitude"][:-1], df["longitude"][:-1]], axis=-1),
        np.stack([df["latitude"][1:], df["longitude"][1:]], axis=-1),
        unit=haversine.Unit.METERS,
    )
    # If all entries are further apart than the threshold, return everything
    if (method == "closest" and np.all(distances > threshold / 2)) or np.all(
        distances >= threshold
    ):
        return df
    # Create list of indices of entries to include
    # Always include the first image
    idx = 0
    indices_to_keep = [idx]
    # Determine cumulative distance traversed
    cumulative_distances = np.cumsum(distances)
    if verbose:
        print(
            "Added row {}. {} records remaining to traverse.".format(
                idx, len(cumulative_distances)
            )
        )
    # Find where the distance travelled exceeds threshold
    while idx < len(df) - 1:
        offset = utils.first_nonzero(cumulative_distances > threshold)
        if offset < 0:
            break
        if method == "threshold":
            offset += 1
        elif method == "closest":
            if (
                offset == 0
                or cumulative_distances[offset - 1] < threshold / 2
                or cumulative_distances[offset] - threshold
                <= threshold - cumulative_distances[offset - 1]
            ):
                offset += 1
        else:
            raise ValueError("Unsupported method: {}".format(method))
        idx += offset
        indices_to_keep.append(idx)
        this_distance = cumulative_distances[offset - 1]
        cumulative_distances = cumulative_distances[offset:] - this_distance
        if verbose:
            print(
                f"Added row {idx} (moved {this_distance:.2f} m)."
                f" {len(cumulative_distances)} records remaining to traverse."
            )
    if indices_to_keep[-1] != len(df) - 1 and (
        include_last
        or (include_last is None and cumulative_distances[-1] >= threshold / 2)
    ):
        # Add the last frame
        indices_to_keep.append(-1)
        if verbose:
            print("Added row {} (last row).".format(indices_to_keep[-1]))
    return df.iloc[indices_to_keep]


def subsample_distance_sitewise(
    df,
    distance=2.5,
    min_population=500,
    target_population=1000,
    max_factor=4,
    verbose=1,
    use_tqdm=True,
):
    """
    Subsample a dataframe by distance between entries, for each site separately.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns named ``latitude``, ``longitude``, and ``site``.
    distance : float, default=2.5
        Target distance between consecutive entries in the dataframe.
    min_population : int, default=500
        Sites with fewer than this many entries will not be subsampled.
    target_population : int or None, default=1000
        Desired number of samples per site. The specified distance will
        by multiplied by a factor 1, 2, 3, ..., `max_factor` until the population
        falls below `target_population`; the largest factor before falling
        below the `target_population` will be used.
        Set to ``0``, ``-1``, or ``None`` to disable additional subsampling.
    max_factor : int, default=4
        Maximum distance factor to use when performing additional subsampling
        for over populated sites.
        The maximum distance at which to subsample is ``distance * max_factor``.
    verbose : int, default=1
        Verbosity level.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if stdout is sent to a
        file. Default is ``True``.
    """
    t0 = time.time()

    site2indices = utils.unique_map(df["site"])

    print(
        f"Will subsample {len(df)} records over {len(site2indices)} sites."
        f"\n  distance = {distance}m"
        f"\n  min_population = {min_population}"
    )

    n_below_thr = 0
    n_unchanged = 0
    n_changed = 0
    tally_factors = {k: 0 for k in range(1, 1 + max_factor)}

    dfs_redacted = []

    if verbose != 1:
        use_tqdm = False

    for deployment in tqdm.tqdm(site2indices, disable=not use_tqdm):
        df_i = df.iloc[site2indices[deployment]]
        pre_population = len(df_i)
        if len(df_i) <= min_population:
            n_below_thr += 1
        else:
            df_i = subsample_distance(df_i, threshold=distance, verbose=verbose - 1)
            factor_used = 1
            if target_population is not None and target_population > 0:
                # Try further subsampling at increased distances to reduce pop
                df_j = df_i
                for factor in range(2, max_factor + 1):
                    df_prev = df_j
                    df_j = subsample_distance(
                        df_i, threshold=distance * factor, verbose=verbose - 1
                    )
                    if len(df_j) < target_population:
                        df_j = df_prev
                        factor_used = factor - 1
                        break
                else:
                    factor_used = factor
                df_i = df_j

            if len(df_i) == pre_population:
                n_unchanged += 1
            else:
                n_changed += 1
                tally_factors[factor_used] += 1
        dfs_redacted.append(df_i)

    df_redacted = pd.concat(dfs_redacted)

    if verbose >= 1:
        print("Finished downsampling in {:.1f} seconds".format(time.time() - t0))

        n_kept = len(df_redacted)
        out_str = "Kept {:5.2f}% of rows ({:7d}/{:7d}).".format(
            n_kept / len(df) * 100.0, n_kept, len(df)
        )
        out_str += (
            f"\nThere were {n_below_thr} sites below threshold ({min_population})."
        )
        out_str += (
            f"\nThere were {n_unchanged} other sites which also remained unchanged."
        )
        out_str += f"\nThere were {n_changed} sites which were subsampled."
        if target_population is not None and target_population > 0:
            for k, v in tally_factors.items():
                out_str += f"\n{v:8d} sites subsampled at factor={k} (distance={k * distance}m)"
        print(out_str, flush=True)

    return df_redacted


def subsample_distance_sitewise_from_csv(
    input_csv,
    output_csv,
    *args,
    verbose=1,
    **kwargs,
):
    """
    Subsample a CSV file by distance between records.

    Parameters
    ----------
    input_csv : str
        Path to input CSV file.
    output_csv : str
        Path to output CSV file.
    verbose : int, default=1
        Verbosity level
    **kwargs
        Additional arguments as per subsample_distance_sitewise.
    """
    t0 = time.time()

    if verbose >= 1:
        print(f"Will subsample {input_csv}")
        print(f"            -> {output_csv}")
        print("Reading CSV file ({})...".format(utils.file_size(input_csv)), flush=True)

    df = pd.read_csv(input_csv)

    if verbose >= 1:
        print("Loaded CSV file in {:.1f} seconds".format(time.time() - t0), flush=True)

    if (
        "latitude" not in df.columns
        and "lat" in df.columns
        and "longitude" not in df.columns
    ):
        mapper = {"lat": "latitude"}
        if "long" in df.columns:
            mapper["long"] = "longitude"
            df = df.rename(columns=mapper)
        elif "lon" in df.columns:
            mapper["lon"] = "longitude"
            df = df.rename(columns=mapper)
        remapper = {v: k for k, v in mapper.items()}
    else:
        remapper = None

    if "latitude" not in df.columns and "longitude" not in df.columns:
        raise ValueError("Latitude and longitude are required columns.")

    df = subsample_distance_sitewise(df, *args, verbose=verbose, **kwargs)

    if remapper is not None:
        df = df.rename(columns=remapper)

    if verbose >= 1:
        print(f"Saving CSV file {output_csv}", flush=True)

    df.to_csv(output_csv, index=False)

    if verbose >= 1:
        print(
            "Total runtime: {}".format(datetime.timedelta(seconds=time.time() - t0)),
            flush=True,
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
    import textwrap

    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="""
            Subsample a dataframe, based on distance between samples.
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
        help="Path to input CSV file, in the BenthicNet format.",
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=2.5,
        help=textwrap.dedent(
            """
            Target distance between consecutive entries in the dataframe.
            Default is %(default)s.
        """
        ),
    )
    parser.add_argument(
        "--min-population",
        type=int,
        default=500,
        help=textwrap.dedent(
            """
            Minimum number of samples in a single site for that site to be
            subsampled. Default is %(default)s.
        """
        ),
    )
    parser.add_argument(
        "--target-population",
        type=int,
        default=1000,
        help=textwrap.dedent(
            """
            Desired number of samples per site. The specified distance will
            by multiplied by a factor 1, 2, 3, ..., MAX_FACTOR until the
            population falls below TARGET_POPULATION; the largest factor before
            falling below the TARGET_POPULATION will be used.
            Set to 0 or -1 to disable.
            Default is %(default)s.
        """
        ),
    )
    parser.add_argument(
        "--max-factor",
        type=int,
        default=4,
        help=textwrap.dedent(
            """
            Maximum distance factor to use when performing additional subsampling
            for over populated sites.
            The maximum distance at which to subsample is DISTANCE * MAX_FACTOR.
            Default is %(default)s.
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
    Run command line interface for subsampling dataframe by distance.
    """
    parser = get_parser()
    kwargs = vars(parser.parse_args())

    kwargs["verbose"] -= kwargs.pop("quiet", 0)

    return subsample_distance_sitewise_from_csv(**kwargs)


if __name__ == "__main__":
    main()
