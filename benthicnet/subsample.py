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
import sklearn.neighbors
import tqdm

import benthicnet.io
import benthicnet.utils
from benthicnet import __meta__

from .kde_tools import EARTH_RADIUS


def subsample_distance(
    df,
    threshold=2.5,
    method="closest",
    exhaustive=False,
    verbose=0,
    include_last=None,
    tree=None,
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

    exhaustive : bool, default=True
        Whether to check the distance to all previous points before marking a
        new point point to be included in the subsampled series.
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
    tree : sklearn.neighbors.BallTree or None, optional
        Pre-built BallTree object, made from this set of latitude and longitude
        coordinates.

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
    xy = np.stack([df["latitude"], df["longitude"]], axis=-1)
    distances = haversine.haversine_vector(xy[:-1], xy[1:], unit=haversine.Unit.METERS)

    # If all entries are further apart than the threshold, return everything
    if (method == "closest" and np.all(distances > threshold / 2)) or np.all(
        distances >= threshold
    ):
        return df
    # Create ball tree for fast search
    if exhaustive and not tree:
        tree = sklearn.neighbors.BallTree(np.radians(xy), metric="haversine")
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
        offset = benthicnet.utils.first_nonzero(cumulative_distances >= threshold)
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
        if exhaustive:
            set_to_keep = set(indices_to_keep)
            while True:
                if idx + offset >= len(df):
                    break
                this_coord = [
                    df.iloc[idx + offset]["latitude"],
                    df.iloc[idx + offset]["longitude"],
                ]
                neighbours = tree.query_radius(
                    [np.radians(this_coord)],
                    threshold / 2 / EARTH_RADIUS,
                )[0]
                if len(set_to_keep.intersection(neighbours)) == 0:
                    break
                offset += 1
                if offset >= len(cumulative_distances):
                    break
            if idx + offset >= len(df):
                break
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


def subsample_index(df, target_population):
    """
    Subsample data based on distance between records.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to process. Data is assumed to be ordered by sample datetime.
    target_population : int
        Desired number of samples.

    Returns
    -------
    pandas.DataFrame
        Dataframe like `df`, but with only a subset of the rows included.
    """
    if len(df) <= target_population:
        return df
    idx = np.round(np.linspace(0, len(df) - 1, target_population)).astype(int)
    idx = np.unique(idx)
    return df.iloc[idx]


def coordinate_centroid(latitudes, longitudes, degrees=True):
    """
    Take the center coordinate via normal vectors, $n_E$.

    https://www.ffi.no/en/research/n-vector/n-vector-explained

    Parameters
    ----------
    latitudes : array-like
        Latitude of coordinates.
    longitudes : array-like
        Longitudes of coordinates.
    degrees : bool, default=True
        Whether the inputs are in degrees, otherwise they are expected to be
        radians.

    Returns
    -------
    c_latitude : float
        The latitude of the centroid, in degrees if ``degrees=True``,
        otherwise in radians.
    c_longitude : float
        The longitude of the centroid, in degrees if ``degrees=True``,
        otherwise in radians.
    """
    if degrees:
        latitudes = np.radians(latitudes)
        longitudes = np.radians(longitudes)
    x = np.mean(np.cos(latitudes) * np.cos(longitudes))
    y = np.mean(np.cos(latitudes) * np.sin(longitudes))
    z = np.mean(np.sin(latitudes))

    c_longitude = np.arctan2(y, x)
    c_norm = np.sqrt(x * x + y * y)
    c_latitude = np.arctan2(z, c_norm)

    if degrees:
        c_latitude = np.degrees(c_latitude)
        c_longitude = np.degrees(c_longitude)

    return c_latitude, c_longitude


def coordinate_center_independent(latitudes, longitudes, degrees=True):
    """
    Take the center of latitude and longitude, independently.

    Parameters
    ----------
    latitudes : array-like
        Latitude of coordinates.
    longitudes : array-like
        Longitudes of coordinates.
    degrees : bool, default=True
        Whether the inputs are in degrees, otherwise they are expected to be
        radians.

    Returns
    -------
    c_latitude : float
        The central latitude, in degrees if ``degrees=True``,
        otherwise in radians.
    c_longitude : float
        The central longitude, in degrees if ``degrees=True``,
        otherwise in radians.
    """
    if degrees:
        latitudes = np.radians(latitudes)
        longitudes = np.radians(longitudes)

    x = np.sum(np.cos(latitudes))
    y = np.sum(np.sin(latitudes))
    c_latitude = np.arctan2(y, x)

    x = np.sum(np.cos(longitudes))
    y = np.sum(np.sin(longitudes))
    c_longitude = np.arctan2(y, x)

    if degrees:
        c_latitude = np.degrees(c_latitude)
        c_longitude = np.degrees(c_longitude)
    return c_latitude, c_longitude


def count_subsites(df, subsite_distance=500.0, return_tree=False):
    """
    Count the number of subsites, detected by distance between consecutive samples.

    Parameters
    ----------
    df : pandas.DataFrame
    subsite_distance : float, default=500.0
    return_tree : bool, optional

    Returns
    -------
    num_subsites : int
        The number of subsites in the dataframe.
    tree : sklearn.neighbors.BallTree or None, optional
        The fitted BallTree object used to verify subsite, if there were any
        to check.
    """
    # Ensure there are coordinates to compare
    select = ~pd.isna(df["latitude"]) & ~pd.isna(df["longitude"])
    if sum(select) < 2:
        n_subsite = 1
        # No lat/lon data to evaluate subsites with
        if return_tree:
            return n_subsite, None
        return n_subsite
    # Measure distance between entries using Haversine method
    xy = np.stack([df[select]["latitude"], df[select]["longitude"]], axis=-1)
    distances = haversine.haversine_vector(xy[:-1], xy[1:], unit=haversine.Unit.METERS)
    # Check where the distance between consecutive samples exceeds the subsite
    # distance
    is_subsite_boundary = distances > subsite_distance
    n_subsite = 1 + sum(is_subsite_boundary)

    if n_subsite < 2:
        # All datapoints are in one subsite
        if return_tree:
            return n_subsite, None
        return n_subsite

    # Check these subsites to make sure there's actually a gap with no samples
    # in between this pair
    xy_r = np.radians(xy)
    tree = sklearn.neighbors.BallTree(xy_r, metric="haversine")
    boundary_indices = np.nonzero(is_subsite_boundary)[0]

    n_true_subsite = 1
    for idx in boundary_indices:
        # Find the point in the middle of these two coordinates
        c_r = coordinate_centroid(
            xy_r[idx : idx + 2][:, 0],
            xy_r[idx : idx + 2][:, 1],
            degrees=False,
        )
        # Check if there are any points within 40% of the subsite distance
        # of the midpoint
        neighbours = tree.query_radius([c_r], 0.4 * subsite_distance / EARTH_RADIUS)[0]
        # If there are any neighbours, this is not a true change in subsite,
        # just a bad ordering of the original coordinates.
        if len(neighbours) == 0:
            n_true_subsite += 1

    if return_tree:
        return n_true_subsite, tree
    return n_true_subsite


def subsample_distance_sitewise(
    df,
    distance=2.5,
    min_population=500,
    allow_nonspatial=True,
    target_population=1000,
    max_factor=None,
    factors=None,
    exhaustive=False,
    subsite_distance=500.0,
    subsite_population_bonus=None,
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
    allow_nonspatial : bool, default=True
        Handle sites with heavily duplicated coordinates with non-spatial
        downsampling instead. The threshold is when the number of unique lat/lon
        coordinates is higher than the number of samples per coordinate.
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
    factors : list, default=None
        Distance factors to use when performing additional subsampling
        for over populated sites.
        The distance will be multiplied by these factors to increase subsampling
        for sites bearing more than the ``target_population``.
    exhaustive : int, default=0
        Whether to check the distance to all previous points before marking a
        new point point to be included in the subsampled series.
        If this is ``1``, an exhaustive search is performed for the initial
        subsampling step.
        If this is ``2``, an exhaustive search is performed for subsequent
        subsampling steps to satisfy the target population limit as well.
    subsite_distance : float, default=500.0
        Distance in meters between consecutive samples to detect a subsite.
    subsite_population_bonus : int or None, optional
        Amount of increase in target population for every additional subsite
        within the site.
        Subsites are detected when there is a gap of at least
        ``subsite_distance`` between samples.
        Default is half of ``target_population``.
        Set to ``0`` or ``-1`` to disable.
    verbose : int, default=1
        Verbosity level.
    use_tqdm : bool, optional
        Whether to use tqdm to print progress. Disable if stdout is sent to a
        file. Default is ``True``.
    """
    t0 = time.time()

    site2indices = benthicnet.utils.unique_map(df["site"])
    n_site = len(site2indices)

    if factors is not None and max_factor is not None:
        raise ValueError("Only one of factors and max_factor should be set.")
    if factors is None and (max_factor is None or max_factor <= 0):
        factors = [1]
        max_factor = 1
    if max_factor is not None:
        factors = list(range(1, max_factor + 1))
    else:
        max_factor = max(factors)
    factors = sorted(factors)
    if factors[0] < 1:
        raise ValueError("Factor can not be smaller than 1")
    if factors[0] != 1:
        factors.insert(0, 1)

    if target_population is None or target_population <= 0:
        target_population = None
    if subsite_population_bonus is None:
        subsite_population_bonus = target_population / 2
    if subsite_population_bonus <= 0:
        subsite_population_bonus = 0

    print(
        f"Will subsample {len(df)} records over {n_site} sites."
        f"\n  distance = {distance}m"
        f"\n  min_population = {min_population}"
        f"\n  target_population = {target_population}"
        f"\n  distance factors = {factors}"
        f"\n  allow_nonspatial = {allow_nonspatial}"
        f"\n  exhaustive = {exhaustive}"
        f"\n  subsite_distance = {subsite_distance}m"
        f"\n  subsite_population_bonus = {subsite_population_bonus}"
    )

    n_below_thr = 0
    n_unchanged = 0
    n_subspatial = 0
    n_subindex = 0
    total_subsite = 0
    n_subsite_check = 0
    n_multiple_subsite = 0
    tally_factors = {1: 0}
    for k in factors:
        tally_factors[k] = 0

    dfs_redacted = []

    if verbose != 1:
        use_tqdm = False

    for deployment in tqdm.tqdm(site2indices, disable=not use_tqdm):
        df_i = df.iloc[site2indices[deployment]]
        pre_population = len(df_i)

        n_coords = len(df_i.drop_duplicates(subset=["latitude", "longitude"]))
        samp_per_coord = pre_population / n_coords

        if not allow_nonspatial or (
            target_population and n_coords >= target_population
        ):
            use_spatial = True
        else:
            use_spatial = n_coords > samp_per_coord

        target_population_i = target_population
        tree = None
        if target_population and subsite_population_bonus:
            n_subsite, tree = count_subsites(df_i, subsite_distance, return_tree=True)
            total_subsite += n_subsite
            n_subsite_check += 1
            if n_subsite > 1:
                n_multiple_subsite += 1
            target_population_i = target_population + subsite_population_bonus * (
                n_subsite - 1
            )

        if len(df_i) <= min_population:
            n_below_thr += 1
        elif not use_spatial:
            if target_population:
                df_i = subsample_index(df_i, target_population)
            if len(df_i) == pre_population:
                n_unchanged += 1
            else:
                n_subindex += 1
        else:
            df_i = subsample_distance(
                df_i,
                threshold=distance,
                verbose=verbose - 1,
                exhaustive=exhaustive,
                tree=tree,
            )
            factor_used = 1
            if target_population:
                # Try further subsampling at increased distances to reduce pop
                df_j = df_i
                for i_factor, factor in enumerate(factors):
                    if factor == 1:
                        continue
                    df_prev = df_j
                    df_j = subsample_distance(
                        df_i,
                        threshold=distance * factor,
                        verbose=verbose - 1,
                        exhaustive=exhaustive >= 2,
                    )
                    if len(df_j) < target_population_i:
                        df_j = df_prev
                        factor_used = factors[i_factor - 1]
                        break
                else:
                    factor_used = factor
                df_i = df_j

            if len(df_i) == pre_population:
                n_unchanged += 1
            else:
                n_subspatial += 1
                tally_factors[factor_used] += 1
        dfs_redacted.append(df_i)

    df_redacted = pd.concat(dfs_redacted)

    if verbose >= 1:
        print("Finished downsampling in {:.1f} seconds".format(time.time() - t0))
        if subsite_population_bonus:
            print(
                f"There were {total_subsite} subsites across {n_subsite_check}"
                f" sites."
                f" {n_multiple_subsite} sites had more than one subsite."
            )

        n_kept = len(df_redacted)
        out_str = "Kept {:5.2f}% of rows ({:7d}/{:7d}).".format(
            n_kept / len(df) * 100.0, n_kept, len(df)
        )
        out_str += (
            f"\nThere were {n_below_thr:5d} sites below threshold ({min_population})."
        )
        out_str += (
            f"\nThere were {n_unchanged:5d} other sites which also remained unchanged."
        )
        if allow_nonspatial:
            out_str += f"\nThere were {n_subindex:5d} sites which were subsampled non-spatially."
        out_str += (
            f"\nThere were {n_subspatial:5d} sites which were subsampled spatially."
        )
        if target_population is not None and target_population > 0:
            for k, v in tally_factors.items():
                out_str += f"\n{v:8d} sites subsampled at factor={k:>2d} (distance={k * distance}m)"
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
        print(
            "Reading CSV file ({})...".format(benthicnet.io.file_size(input_csv)),
            flush=True,
        )

    df = benthicnet.io.read_csv(input_csv)

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
        elif "lon" in df.columns:
            mapper["lon"] = "longitude"
        else:
            raise ValueError(
                "Must have a column named 'longitude', 'lon', or 'long', but"
                f" column names are: {df.columns}"
            )
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

    dirname = os.path.dirname(output_csv)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
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
        "--forbid-nonspatial",
        dest="allow_nonspatial",
        action="store_false",
        help=textwrap.dedent(
            """
            Disallow handling duplicated coordinates with non-spatial
            downsampling.
            If this is set, all sites will be spatially downsampled, even if
            they have fewer samples per coordinate than unique coordinates.
        """
        ),
    )
    parser.add_argument(
        "--min-population",
        type=int,
        default=50,
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--max-factor",
        type=int,
        default=None,
        help=textwrap.dedent(
            """
            Maximum distance factor to use when performing additional subsampling
            for over populated sites.
            The maximum distance at which to subsample is DISTANCE * MAX_FACTOR.
            Default is [2, 3, 4].
        """
        ),
    )
    group.add_argument(
        "--factors",
        type=int,
        nargs="+",
        help=textwrap.dedent(
            """
            Set of distance factors consider when performing additional
            subsampling for over populated sites.
            Default is %(default)s.
        """
        ),
    )
    parser.add_argument(
        "--exhaustive",
        type=int,
        nargs="?",
        default=0,
        const=1,
        help=textwrap.dedent(
            """
            Perform an exhaustive search of the previous
            samples to ensure none are within DISTANCE/2 before including the
            next sample.
        """
        ),
    )
    parser.add_argument(
        "--subsite-distance",
        type=float,
        default=500.0,
        help=textwrap.dedent(
            """
            Distance in meters between consecutive samples to detect a subsite.
            Each subsite grants an increase in the target population equal to
            SUBSITE_POPULATION_BONUS.
            Default is %(default)s.
        """
        ),
    )
    parser.add_argument(
        "--subsite-population-bonus",
        type=int,
        default=None,
        help=textwrap.dedent(
            """
            Amount of increase in target population for every additional subsite
            within the site.
            Subsites are detected when there is a gap of at least
            SUBSITE_DISTANCE between samples.
            Default is half of TARGET_POPULATION.
            Set to ``0`` or ``-1`` to disable.
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
