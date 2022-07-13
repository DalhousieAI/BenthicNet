#!/usr/bin/env python
# coding: utf-8

import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.neighbors
from tqdm.auto import tqdm

import benthicnet.io
import benthicnet.plotting
from benthicnet.kde_tools import EARTH_RADIUS

PARTITION2COLOR = {
    "train": "g",
    "test": "m",
    "train_exclusion": "#bb0",
    "test_exclusion": "r",
    "": "#222",
}


def partition_dataset(
    df,
    label_col,
    verbosity=1,
    min_train_pc=50,
    min_test_pc=15,
    target_pc_median=35,
    target_pc_largest=15,
    partition_radius=0,
    exclusion_distance=50,
    exclusion_distance_strict=10,
    only_exclude_same_label=None,
    show_plots=None,
):
    """
    Partition a dataset, using spatial (lat/lon) coordinates.

    df : pandas.DataFrame
        The dataframe to partition.
    label_col : str
        Column containing the labels (classes) to partition.
    verbosity int, default=1
        Verbosity level.
    min_train_pc : float, default=50
        Minimum percentage of each class to place in train partition.
    min_test_pc : float, default=15
        Minimum percentage of each class to place in test partition.
    target_pc_median : float, default=35
        Percentage of the median class to place in the test partition.
    target_pc_largest : float, default=15
        Percentage of the largest class to place in the test partition.
    partition_radius : float, default=0
        Distance, in metres, around the centre point to place in the partition
        at once. Recommended to set to ``0`` (default).
    exclusion_distance : float, default=50
        Distance, in metres, within which samples should be the same partition
        if possible.
    exclusion_distance_strict : float, default=10
        The absolute minimum distance, in metres, which must occur between the
        train and test partitions.
    only_exclude_same_label : bool or None, default=None
        If ``True``, partitioning is done for each class without reference
        to the partitioning of other classes (i.e. spatial relationships only
        hold within a class, not between them). This will not work for multiply
        annotated images. Default behaviour is to partition classes separately
        when images are singly labelled.
    show_plots : bool or None, default=None
        Whether to plot map showing sample locations.
        If ``None`` (default), maps are shown if ``verbosity >= 2``.
    """
    if show_plots is None:
        show_plots = verbosity >= 2

    if verbosity >= 1:
        print("Parameters...")
        print("label_col:", label_col)
        print("min_train_pc:", min_train_pc)
        print("min_test_pc:", min_test_pc)
        print("target_pc_median:", target_pc_median)
        print("target_pc_largest:", target_pc_largest)
        print("partition_radius:", partition_radius)
        print("exclusion_distance:", exclusion_distance)
        print("exclusion_distance_strict:", exclusion_distance_strict)
        print("only_exclude_same_label:", only_exclude_same_label)
        print("show_plots:", show_plots)
        print()

    if verbosity >= 1:
        print(f"Input dataframe has {len(df)} rows")

    # Remove samples missing lat/lon values
    df = df[(~pd.isna(df["latitude"])) & (~pd.isna(df["longitude"]))]
    # Add outpath, so we can determine which rows are the same image
    df["outpath"] = benthicnet.io.determine_outpath(df)
    # Initialize columns to use for partition output and working values,
    # minimum distances from samples in train/test partition.
    df["partition"] = ""
    df["_dist_from_train"] = np.inf
    df["_dist_from_test"] = np.inf

    # Trim spaces either side of label
    df[label_col] = df[label_col].str.strip()

    # Drop unlabelled samples
    df = df[~pd.isna(df[label_col])]

    # Reset index and tell pandas this isn't a copy, so it doesn't complain
    df = df.reset_index(drop=True)
    df.is_copy = False

    if verbosity >= 1:
        print(
            f"After discarding rows without label or coords, dataframe has {len(df)} rows"
        )

    all_labels = df[label_col].unique()

    label_counts = df[label_col].value_counts(ascending=True, dropna=False)

    if verbosity >= 2:
        max_label_name_len = max(len(x) for x in label_counts.index)
        for x in label_counts.index:
            x_ = x + " " * (max_label_name_len - len(x))
            print(f"{x_} {label_counts[x]:>4d}")

    # Reduce to only unique combinations of label and image, to check counts
    dfu = df.drop_duplicates(subset=["outpath", label_col])

    total_ulabel_counts = dfu[label_col].value_counts(ascending=True, dropna=False)

    if verbosity >= 2:
        for x in total_ulabel_counts.index:
            x_ = x + " " * (max_label_name_len - len(x))
            print(f"{x_} {total_ulabel_counts[x]:>4d}")

    def show_label_locations():
        """
        Plot partition distribution on a map.
        """
        c = df[label_col].apply(lambda x: np.nonzero(all_labels == x)[0][0])
        ax = benthicnet.plotting.plot_samples(
            df["latitude"],
            df["longitude"],
            c=c,
            s=20,
            global_map=False,
            show_map="two-tone-alt",
            figsize=(25, 10),
        )
        ax.gridlines(draw_labels=True, linewidth=1, color="w")
        plt.title("Class distribution", size=20)
        plt.show()

    target_test_total_1 = round(
        np.median(total_ulabel_counts[total_ulabel_counts > 3]) * target_pc_median / 100
    )
    target_test_total_2 = round(total_ulabel_counts[-1] * target_pc_largest / 100)
    target_test_total = min(target_test_total_1, target_test_total_2)
    if verbosity >= 1:
        print(f"Will try to get {target_test_total} test samples per class")

    # If there is only one label per image, we can do exclusions
    # based only on images with the same label. Otherwise, we can't
    # because the interaction effects mean we would need to do a
    # reverse lookup on each candidate sample we want to add.
    if only_exclude_same_label is None:
        only_exclude_same_label = len(df) == len(df["outpath"].unique())
        if verbosity < 1:
            pass
        elif only_exclude_same_label:
            print("Only one label per image.")
            print("only_exclude_same_label:", only_exclude_same_label)
        else:
            print("Multiple labels per image.")
            print("only_exclude_same_label:", only_exclude_same_label)

    # Initialize BallTree with the coordinates of all the samples
    xy = np.stack([df["latitude"], df["longitude"]], axis=-1)
    xy_r = np.radians(xy)
    tree = sklearn.neighbors.BallTree(xy_r, metric="haversine")

    def print_table(pc=False):
        """
        Print current partitioning tallies.

        Parameters
        ----------
        pc : bool, default=False
            Whether to print values as a percentage.
        """
        dfu = df.drop_duplicates(subset=["outpath", label_col])
        partitions = sorted(df["partition"].unique())
        max_label_len = max(len(label) for label in total_ulabel_counts.index)
        fmt_label = "{:<" + str(max_label_len) + "s}"
        s = fmt_label.format("Label")
        s += "    Total |"
        for partition in partitions:
            if partition == "":
                partition = "[unallc]"
            s += f" {partition[:8]:>8s}"
        print("\n" + s)
        print("-" * len(s))
        for label in total_ulabel_counts.index:
            s = fmt_label.format(label)
            s += f" {total_ulabel_counts[label]:>8d} |"
            for partition in partitions:
                n = sum((dfu[label_col] == label) & (dfu["partition"] == partition))
                if pc:
                    n = 100 * n / total_ulabel_counts[label]
                    s += f" {n:>7.2f}%"
                else:
                    s += f" {n:>8d}"
            print(s)
        print("-" * len(s))
        sdfu = dfu.drop_duplicates(subset=["outpath"])
        s = fmt_label.format("Total")
        n = n_total = len(sdfu)
        s += f" {n:>8d} |"
        for partition in partitions:
            n = sum(sdfu["partition"] == partition)
            if pc:
                n = 100 * n / n_total
                s += f" {n:>7.2f}%"
            else:
                s += f" {n:>8d}"
        print(s)
        print("-" * len(s))
        s = fmt_label.format("Total")
        n = n_total
        s += f" {n:>8d} |"
        for partition in partitions:
            n = sum(sdfu["partition"] == partition)
            if not pc:
                n = 100 * n / n_total
                s += f" {n:>7.2f}%"
            else:
                s += f" {n:>8d}"
        print(s + "\n")

    def show_sample_partitioning():
        """
        Plot samples on map, with colour-coded partitions.
        """
        c = df["partition"].apply(lambda x: PARTITION2COLOR.get(x, "#888"))
        ax = benthicnet.plotting.plot_samples(
            df["latitude"],
            df["longitude"],
            c=c,
            s=20,
            global_map=False,
            show_map="two-tone-alt",
            figsize=(25, 10),
        )
        ax.gridlines(draw_labels=True, linewidth=1, color="w")
        plt.show()

    def report(pc=False, plot=False):
        """
        Print current partitioning tallies.

        Parameters
        ----------
        pc : bool, default=False
            Whether to print values as a percentage.
        plot : bool, default=False
            Whether to show map indicating partition sample locations.
        """
        # Print partitioning
        print_table(pc=pc)
        if not plot:
            return
        # Show map
        show_sample_partitioning()

    def update_neighbourhood_partition_name(current, neighbour):
        """
        Update partition name of a neighbour of a partition.
        """
        if current in {neighbour, neighbour + "_exclusion", "excluded"}:
            return current
        if current == "":
            return neighbour + "_exclusion"
        if current.endswith("_exclusion"):
            return "excluded"
        if verbosity >= 4:
            print(
                f"Warning: Can't exclude a point neighbouring {neighbour} that's"
                f" already in {current} partition."
            )
        return current

    def allocate_partition(
        partition_name="",
        target=1,
        thr_avail=None,
        min_pc=None,
        max_pc=None,
        only_exclude_same_label=only_exclude_same_label,
        exclusion_distance=exclusion_distance,
        partition_radius=partition_radius,
        select_closest_pc=10,
        force=None,
        force_min_distance=exclusion_distance_strict,
        labels_to_update=None,
        n_updates=1,
    ):
        """
        Allocate sample(s) to a train/test partition.

        Updates are made to the global ``df`` DataFrame.

        - Find classes with insufficient samples. These are labels
          which have fewer than ``target`` samples or ``min_pc`` of the
          total samples with this label in the ``partition_name`` partition.
        - Iterate over labels, in order of fewest samples available to
          allocate to the partition.
        - Iteration ceases once ``n_updates`` partition movement operations
          have occurred.
        - A sample with that label is selected to be moved to the partition.
          It is selected at random from the candidate samples.
          Samples which are close to other members of that partition are
          prioritised.
          Samples neighbouring other partitions will not be selected unless
          ``force`` mode is enabled.
        - Every time a sample is selected to be moved to the partition,
          neighbouring samples within ``partition_radius`` of it are immediately
          moved to that partition too.
        - Neighbours more than ``partition_radius`` from the selected sample,
          but less than ``exclusion_distance`` from it are placed in a
          working partition named ``<partition_name>_exclusion``, and the
          distance from the partition recorded in ``_dist_from_<partition_name>``.
          These properties are utilized in later update steps.

        Parameters
        ----------
        partition_name : str
            Name of the partition to allocate samples to.
        target : int, default=1
            Number of samples to aim for in the partition.
        thr_avail : float or None
            Skip a label if there are still at least ``thr_avail``
            samples available to be added to this partition.
        min_pc : float, optional
            Minimum percentage to be allocated to this partition.
        max_pc : float, optional
            Maximum percentage to be allocated to this partition.
        only_exclude_same_label : only_exclude_same_label
            Whether to only move samples with the same label
            to <partition>_exclusion placeholder partition.
        exclusion_distance : exclusion_distance
            Exclusion radius. Distance over which samples should
            be placed in the same partition (if possible).
        partition_radius : partition_radius
            All samples within this distance from the sample being
            move to the partition will also be moved to the
            partition ``partition_name``.
        select_closest_pc : float, default=10
            Where samples are in the exclusion radius of the target
            partition, select a new sample from the closest
            ``select_closest_pc``% of those samples.
        force : bool or None, default=None
            Whether to forcibly place a sample from excluded or
            a different exclusion area into the partition.
            If ``None``, then ``force`` is used for classes which
            need samples to be allocated urgently to prevent the
            count being able to fall below ``min_pc`` percentage
            of total.
        force_min_distance : float, default=exclusion_distance_strict
            Minimum distance away from an opposing partition
            for a sample to be forcibly added to ``partition_name``.
        labels_to_update : list or set, optional
            Subset of the labels which will be updated.
        n_updates : int, default=1
            Number of updates to occur before halting.
        """
        if not partition_name:
            raise ValueError("Partition name must be provided")

        dfu = df.drop_duplicates(subset=["outpath", label_col])

        sdf = dfu[dfu["partition"] == partition_name]
        existing_label_counts = sdf[label_col].value_counts(
            ascending=True, dropna=False
        )
        labels_with_insufficient_samp = existing_label_counts.index[
            existing_label_counts < target
        ]
        labels_with_insufficient_samp = set(labels_with_insufficient_samp)

        # Add in labels where there are none in the partition
        labels_with_no_samp = set(all_labels).difference(sdf[label_col].unique())
        labels_with_insufficient_samp = set(labels_with_insufficient_samp).union(
            labels_with_no_samp
        )

        # Exclude labels where adding more samples would exceed the maximum %
        if max_pc and max_pc < 100:
            to_remove = []
            for label in set(all_labels).difference(labels_with_insufficient_samp):
                if total_ulabel_counts.get(label, 0) == 0:
                    if verbosity >= 3:
                        print(f"Getting count for label '{label}', which is 0")
                    to_remove.append(label)
                    continue
                if (
                    100
                    * (existing_label_counts.get(label, 0) + 1)
                    / total_ulabel_counts[label]
                    > max_pc
                ):
                    to_remove.append(label)
            labels_with_insufficient_samp = labels_with_insufficient_samp.difference(
                to_remove
            )

        # Sort by number of available images, starting with fewest available
        sdf = dfu[dfu["partition"].isin(["", partition_name + "_exclusion"])]
        avail_label_counts = sdf[label_col].value_counts(ascending=True, dropna=False)
        sdfx = dfu[
            dfu["partition"].isin(
                [
                    "",
                    partition_name + "_exclusion",
                    "train_exclusion",
                    "test_exclusion",
                    "excluded",
                ]
            )
        ]
        avail_label_counts_x = sdfx[label_col].value_counts(
            ascending=True, dropna=False
        )

        # Check if we need to act now to ensure a sufficiently high percentage
        # can get in the partition
        labels_with_urgency = set()
        if min_pc:
            for label in all_labels:
                if avail_label_counts_x.get(label, 0) == 0:
                    continue
                if (
                    100 * total_ulabel_counts[label] * min_pc < 1.5
                    and partition_name != "train"
                ):
                    # Don't allocate minimum test when there are hardly any samples,
                    # prioritize minimum train instead.
                    continue
                if (
                    existing_label_counts.get(label, 0)
                    + avail_label_counts.get(label, 0)
                    - 2
                ) <= total_ulabel_counts[label] * min_pc / 100:
                    labels_with_urgency.add(label)
            labels_with_insufficient_samp.union(labels_with_urgency)

        if labels_to_update:
            # Only consider the subset of labels we asked to update
            labels_with_urgency = labels_with_urgency.intersection(labels_to_update)
            labels_with_insufficient_samp = labels_with_insufficient_samp.intersection(
                labels_to_update
            )

        # Use the labels which need to be done, in order of fewest available
        todo = []
        if len(labels_with_urgency) > 0:
            todo += [
                label
                for label in avail_label_counts_x.index
                if label in labels_with_urgency
            ]
        todo += [
            label
            for label in avail_label_counts.index
            if label in labels_with_insufficient_samp
        ]
        if force:
            todo += [
                label
                for label in avail_label_counts_x.index
                if label in labels_with_insufficient_samp and label not in todo
            ]

        if verbosity >= 5:
            print("labels_with_urgency:", labels_with_urgency)
            print("labels_with_insufficient_samp:", labels_with_insufficient_samp)
            print("todo:", todo)

        n_updated = 0
        for label in todo:
            # Might skip if there's plenty of this label left available
            if thr_avail:
                # Check if there are only just enough left in total
                select = (df[label_col] == label) & (
                    df["partition"].isin(
                        ["", partition_name + "_exclusion", partition_name]
                    )
                )
                if sum(select) == target:
                    # Only just enough left! We must allocate now.
                    pass
                # Otherwise, check if there are plenty still left to allocate
                elif avail_label_counts.get(label, 0) > thr_avail:
                    # Skipping this label for now
                    if verbosity >= 4:
                        print(
                            f"Skipping {label} because there are still"
                            f" {avail_label_counts.get(label, 0)} > {thr_avail}"
                            f" available to allocate"
                        )
                    todo.remove(label)
                    continue
            # Prefer to select a label already in the exclusion zone
            select = (df[label_col] == label) & (
                df["partition"] == partition_name + "_exclusion"
            )
            if sum(select) == 0:
                # If there are none, we can select something which is completely unallocated
                select = (df[label_col] == label) & (
                    df["partition"].isin(["", partition_name + "_exclusion"])
                )
            elif select_closest_pc is not None and select_closest_pc < 100:
                # Prefer images which are closer to existing images in this partition
                dists_from_existing = df.loc[select, "_dist_from_" + partition_name]
                dists_from_existing = dists_from_existing[
                    np.isfinite(dists_from_existing)
                ]
                if len(dists_from_existing) > 0:
                    dist_close_enough = np.percentile(
                        dists_from_existing, select_closest_pc
                    )
                    if np.isfinite(dist_close_enough):
                        select = select & (
                            df["_dist_from_" + partition_name] <= dist_close_enough
                        )
            indices = np.nonzero(select.values)[0]
            if len(indices) == 0:
                if not force and not (force is None and label in labels_with_urgency):
                    if verbosity >= 4:
                        print(f"Nothing available to allocate for {label}!")
                    continue
                # Must select something! But there is nothing meeting the criteria!
                # Use the sample furthest away from the opposing partition
                if partition_name == "train":
                    reverse_partition = "test"
                elif partition_name == "test":
                    reverse_partition = "train"
                else:
                    raise ValueError(
                        f"Forced sample addition unsupported for partition {partition_name}"
                    )
                select = (df[label_col] == label) & df["partition"].isin(
                    ["excluded", reverse_partition + "_exclusion"]
                )
                idx = np.arange(len(df))[select][
                    df.loc[select, "_dist_from_" + reverse_partition].argmax()
                ]
                if df.iloc[idx]["_dist_from_" + reverse_partition] < force_min_distance:
                    if verbosity >= 4:
                        print(
                            "Nothing to forcibly select for label {}, partition {}."
                            " Best would be {}m from {}.".format(
                                label,
                                partition_name,
                                df.iloc[idx]["_dist_from_" + reverse_partition],
                                reverse_partition,
                            )
                        )
                    continue
                if verbosity >= 4:
                    print(
                        "Forcibly selected {}, with label {} and distance {} from {}".format(
                            idx,
                            df.iloc[idx][label_col],
                            df.iloc[idx]["_dist_from_" + reverse_partition],
                            reverse_partition,
                        )
                    )
            else:
                idx = np.random.choice(indices)
            if verbosity >= 3:
                print(f"{label} : {idx} -> {partition_name}")
            # Put the sample in the test partition
            df.loc[df.index.values[idx], "partition"] = partition_name
            # Set other annotations on the same image to have the same partition
            select = df["outpath"] == df.loc[df.index.values[idx], "outpath"]
            df.loc[select, "partition"] = partition_name

            allocated_indices = [idx]
            neighbours_pre = sum(select)

            if partition_radius:
                # Find neighbouring images within the same-partition zone
                neighbours = tree.query_radius(
                    [xy_r[idx]], partition_radius / EARTH_RADIUS
                )
                neighbours = np.unique(np.concatenate(neighbours))
                # Update the partition of unassigned points in the neighbourhood
                df.loc[df.index.values[neighbours], "partition"] = df.loc[
                    df.index.values[neighbours], "partition"
                ].apply(
                    lambda x: partition_name
                    if x in ["", partition_name + "_exclusion"]
                    else x
                )
                allocated_indices = neighbours
                if verbosity >= 4:
                    print(
                        f"{len(neighbours)} neighbours assigned to the {partition_name} partition."
                    )
                neighbours_pre = len(neighbours)

            # Move neighbouring images into exclusion zone
            neighbours, dist = tree.query_radius(
                xy_r[allocated_indices],
                exclusion_distance / EARTH_RADIUS,
                return_distance=True,
            )
            dist *= EARTH_RADIUS  # Convert distances into metres
            # Merge to get neighbours of any of the points we searched around
            neighbours_all = np.unique(np.concatenate(neighbours))
            # Update the partition of unassigned points in the neighbourhood
            select = df.index.values[neighbours_all]
            if only_exclude_same_label and not partition_radius:
                # Only select samples which have the same label
                sdf = df.loc[select]
                outpaths = sdf.loc[sdf[label_col] == label, "outpath"]
                # But extend this to be samples on the same image bearing that label
                select = df["outpath"].isin(outpaths)
            df.loc[select, "partition"] = df.loc[select, "partition"].apply(
                functools.partial(
                    update_neighbourhood_partition_name, neighbour=partition_name
                )
            )
            # Record the closest distance to a partition focii
            df.loc[
                df.index.values[neighbours[0]], "_dist_from_" + partition_name
            ] = np.minimum(
                df.loc[df.index.values[neighbours[0]], "_dist_from_" + partition_name],
                dist[0],
            )
            extra_txt = " (of neighbours)" if partition_radius else ""
            if verbosity >= 4:
                print(
                    f"{len(neighbours_all) - neighbours_pre} neighbours{extra_txt}"
                    f" assigned to the {partition_name}_exclusion partition."
                )
            n_updated += 1
            if n_updates and n_updated >= n_updates:
                break

        return n_updated, len(todo) - n_updated

    allocate_train = functools.partial(
        allocate_partition,
        partition_name="train",
        thr_avail=5,
        min_pc=min_train_pc,
        force=True,
    )
    allocate_test = functools.partial(
        allocate_partition, partition_name="test", min_pc=min_test_pc, max_pc=35
    )

    def main_step():
        if verbosity >= 1:
            report()
            print(f"Allocating up to {target_test_total} test samples per class...")
        for target in range(target_test_total + 1):
            if verbosity >= 2:
                print()
                print(f"Working toward {target} images in the test set...")
                print()
            test_update_count = 1
            any_updates = False
            while test_update_count > 0:
                train_update_count = 1
                while train_update_count > 0:
                    target_train = round(target * 1.2)
                    train_update_count, train_todo_count = allocate_train(
                        target=max(2, target_train),
                        thr_avail=target_train + 5,
                        labels_to_update=[
                            lab
                            for lab in all_labels
                            if round(total_ulabel_counts[lab] * min_train_pc / 100)
                            >= target_train
                        ],
                    )
                    if verbosity >= 5:
                        print("train_updates", train_update_count)
                    any_updates |= train_update_count > 0
                test_update_count, test_todo_count = allocate_test(target=target)
                if verbosity >= 5:
                    print("test_updates", test_update_count)
                    print()
                any_updates |= test_update_count > 0
            if any_updates and verbosity >= 3:
                report()
                print()

    def monotonic_test():
        partition_name = "test"
        if verbosity >= 1:
            print(
                f"Trying to make number of {partition_name} samples monotonic"
                f" with total number of samples..."
            )
            print()
        for i_label, label in enumerate(total_ulabel_counts.index):
            if i_label == 0:
                continue

            dfu = df.drop_duplicates(subset=["outpath", label_col])

            partition_counts = dfu.loc[
                df["partition"] == partition_name, label_col
            ].value_counts(ascending=True, dropna=False)
            target = max(
                [
                    partition_counts.get(lab, 0)
                    for lab in total_ulabel_counts.index[:i_label]
                ]
            )
            if target <= partition_counts.get(label, 0):
                continue
            if verbosity >= 2:
                print()
                print(
                    f"Working toward {target} images in the {partition_name} set of {label}..."
                )
                print()
            test_update_count = 1
            any_updates = False
            while test_update_count > 0:
                train_update_count = 1
                while train_update_count > 0:
                    train_update_count, train_todo_count = allocate_train(
                        target=max(2, round(target * 1.2)),
                        thr_avail=round(target * 1.2) + 5,
                    )
                    if verbosity >= 5:
                        print("train_updates", train_update_count)
                    any_updates |= train_update_count > 0
                test_update_count, test_todo_count = allocate_test(
                    target=target, labels_to_update=[label], force=True
                )
                if verbosity >= 5:
                    print("test_updates", test_update_count)
                    print()
                any_updates |= test_update_count > 0
            if any_updates and verbosity >= 3:
                report()
                print()

    def force_min_pc_test():
        partition_name = "test"
        if verbosity >= 1:
            print(
                f"Ensuring at least {min_test_pc}% the samples are in {partition_name} partition..."
            )
            print()
        for label in total_ulabel_counts.index:
            target = round(total_ulabel_counts[label] * min_test_pc / 100)
            if verbosity >= 2:
                print()
                print(
                    f"Working toward {target} images in the {partition_name} set of {label}..."
                )
                print()
            test_update_count = 1
            any_updates = False
            while test_update_count > 0:
                train_update_count = 1
                while train_update_count > 0:
                    train_update_count, train_todo_count = allocate_train(
                        target=max(2, round(target * 1.2)),
                        thr_avail=round(target * 1.2) + 5,
                    )
                    if verbosity >= 5:
                        print("train_updates", train_update_count)
                    any_updates |= train_update_count > 0
                test_update_count, test_todo_count = allocate_test(
                    target=target,
                    labels_to_update=[label],
                    force=True,
                )
                if verbosity >= 5:
                    print("test_updates", test_update_count)
                    print()
                any_updates |= test_update_count > 0
            if any_updates and verbosity >= 3:
                report()
                print()

    def force_min_pc_train():
        partition_name = "train"
        if verbosity >= 1:
            print(
                f"Ensuring at least {min_train_pc}% the samples are in"
                f" {partition_name} partition..."
            )
            print()
        for label in total_ulabel_counts.index:
            target = np.ceil(total_ulabel_counts[label] * min_train_pc / 100)
            if verbosity >= 2:
                print()
                print(
                    f"Working toward {target} images in the {partition_name} set of {label}..."
                )
                print()
            train_update_count = 1
            any_updates = False
            while train_update_count > 0:
                train_update_count, train_todo_count = allocate_train(
                    target=target,
                    thr_avail=None,
                    labels_to_update=[label],
                )
                if verbosity >= 5:
                    print("train_updates", train_update_count)
                any_updates |= train_update_count > 0
            if any_updates and verbosity >= 3:
                report()
                print()

    def monotonic_train():
        partition_name = "train"
        if verbosity >= 1:
            print(
                f"Trying to make number of {partition_name} samples monotonic"
                f" with total number of samples..."
            )
            print()
        for i_label, label in enumerate(total_ulabel_counts.index):
            if i_label == 0:
                continue
            dfu = df.drop_duplicates(subset=["outpath", label_col])
            partition_counts = dfu.loc[
                df["partition"] == partition_name, label_col
            ].value_counts(ascending=True, dropna=False)
            target = max(
                [
                    partition_counts.get(lab, 0)
                    for lab in total_ulabel_counts.index[:i_label]
                ]
            )
            if target <= partition_counts.get(label, 0):
                continue
            if verbosity >= 2:
                print()
                print(
                    f"Working toward {target} images in the {partition_name} set of {label}..."
                )
                print()
            train_update_count = 1
            any_updates = False
            while train_update_count > 0:
                train_update_count, train_todo_count = allocate_train(
                    target=target,
                    thr_avail=None,
                    labels_to_update=[label],
                )
                if verbosity >= 5:
                    print("train_updates", train_update_count)
                any_updates |= train_update_count > 0
            if any_updates and verbosity >= 3:
                report()
                print()

    def handle_unallocated():
        # Move unallocated into train partition
        partition_name = "train"
        select = df["partition"] == ""
        if sum(select) == 0:
            if verbosity >= 1:
                print("No unallocated images to allocate")
        else:
            if verbosity >= 1:
                print(f"Moving {sum(select)} unallocated to {partition_name}...")
            df.loc[select, "partition"] = partition_name
            neighbours_pre = sum(select)

            # Move neighbouring images into exclusion zone
            neighbours, dist = tree.query_radius(
                xy_r[select], exclusion_distance / EARTH_RADIUS, return_distance=True
            )
            dist *= EARTH_RADIUS  # Convert distances into metres
            # Merge to get neighbours of any of the points we searched around
            neighbours_all = np.unique(np.concatenate(neighbours))
            # Update the partition of unassigned points in the neighbourhood
            df.loc[df.index.values[neighbours_all], "partition"] = df.loc[
                df.index.values[neighbours_all], "partition"
            ].apply(
                functools.partial(
                    update_neighbourhood_partition_name, neighbour=partition_name
                )
            )
            # Record the closest distance to a partition focii
            df.loc[
                df.index.values[neighbours[0]], "_dist_from_" + partition_name
            ] = np.minimum(
                df.loc[df.index.values[neighbours[0]], "_dist_from_" + partition_name],
                dist[0],
            )
            if verbosity >= 4:
                print(
                    f"{len(neighbours_all) - neighbours_pre} neighbours assigned"
                    f" to the {partition_name}_exclusion partition."
                )

    def handle_exclusion_zones():
        # Move as many exclusion images into partitions as we can
        if verbosity >= 1:
            print(
                "Moving images in single-partition exclusion zones into those"
                " partitions as possible..."
            )
        df["_dist_from_any"] = np.inf
        # Add a _dist_from_any column
        for partition_name in ["train", "test"]:
            select = df["partition"] == partition_name + "_exclusion"
            df.loc[select, "_dist_from_any"] = df.loc[
                select, "_dist_from_" + partition_name
            ]
        while True:
            select = df["partition"].str.endswith("_exclusion")
            if verbosity >= 2:
                print(
                    f"There are {sum(select)} samples in undetermined exclusion zones"
                )
            if sum(select) == 0:
                break
            idx = df.index[select][df.loc[select, "_dist_from_any"].argmin()]

            partition_name = df.loc[idx, "partition"].replace("_exclusion", "")

            if verbosity >= 3:
                print(f"Moving index {idx} to {partition_name}")

            # Set other annotations on the same image to have the same partition
            select = df["outpath"] == df.loc[idx, "outpath"]
            df.loc[select, "partition"] = partition_name

            neighbours_pre = sum(select)

            # Move neighbouring images into exclusion zone
            neighbours, dist = tree.query_radius(
                xy_r[select], exclusion_distance / EARTH_RADIUS, return_distance=True
            )
            dist *= EARTH_RADIUS  # Convert distances into metres
            # Merge to get neighbours of any of the points we searched around
            neighbours_all = np.unique(np.concatenate(neighbours))
            # Update the partition of unassigned points in the neighbourhood
            df.loc[df.index.values[neighbours_all], "partition"] = df.loc[
                df.index.values[neighbours_all], "partition"
            ].apply(
                functools.partial(
                    update_neighbourhood_partition_name, neighbour=partition_name
                )
            )
            # Record the closest distance to a partition focii
            df.loc[
                df.index.values[neighbours[0]], "_dist_from_" + partition_name
            ] = np.minimum(
                df.loc[df.index.values[neighbours[0]], "_dist_from_" + partition_name],
                dist[0],
            )
            df.loc[df.index.values[neighbours[0]], "_dist_from_any"] = np.minimum(
                df.loc[df.index.values[neighbours[0]], "_dist_from_any"],
                dist[0],
            )
            if verbosity >= 4:
                print(
                    f"{len(neighbours_all) - neighbours_pre} neighbours assigned"
                    f" to the {partition_name}_exclusion partition."
                )
                print()

    def redistribute_excluded():
        if verbosity >= 1:
            print("Redistribute fully excluded images outside strict radius...")
        # Minimize number of excluded samples
        select = df["_dist_from_train"] > exclusion_distance_strict
        select |= df["_dist_from_test"] > exclusion_distance_strict
        select &= df["partition"] == "excluded"

        if verbosity >= 2:
            print(f"Trying to redistribute {sum(select)} excluded images")

        select_train = df["_dist_from_train"] <= df["_dist_from_test"]
        df.loc[select & select_train, "partition"] = "train"
        df.loc[select & ~select_train, "partition"] = "test"

    main_step()
    monotonic_test()
    force_min_pc_test()
    force_min_pc_train()
    monotonic_train()
    handle_unallocated()
    handle_exclusion_zones()
    redistribute_excluded()

    if verbosity >= 1:
        print()
        print("Final partitioning:")
        report()
        report(pc=True)
        if show_plots:
            show_label_locations()
            report(plot=True)

    return df
