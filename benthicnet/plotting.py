"""
Plotting utilities.
"""

import cartopy.crs
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import kde_tools

DATASET2ORG = {
    "julia": "4D Oceans",
    "Julia_2020": "4D Oceans",
    "shreya": "4D Oceans",
    "Shreya_2020": "4D Oceans",
    "sabrina": "AADC",
    "Sabrina_2017": "AADC",
    "vms": "AADC",
    "VMS_2011": "AADC",
    "ngu": "NGU",
    "NGU_2010": "NGU",
    "NGU_2014": "NGU",
    "NGU_2015": "NGU",
    "NGU_2017": "NGU",
    "bastos": "Bastos Lab",
    "dfo_eelgrass": "DFO",
    "george2000": "DFO",
    "george2002": "DFO",
    "Georges_Bank_2000": "DFO",
    "Georges_Bank_2002": "DFO",
    "german2003": "DFO",
    "German_Bank_2003": "DFO",
    "german2006": "DFO",
    "German_Bank_2006": "DFO",
    "german2010": "DFO",
    "German_Bank_2010": "DFO",
    "noaa_habcam": "DFO",
    "NOAA_HabCam_2015": "DFO",
    "eac": "EAC",
    "EAC_2021": "EAC",
    "Doc Ricketts": "MBARI",
    "i2MAP": "MBARI",
    "Mini ROV": "MBARI",
    "MiniROV": "MBARI",
    "Tiburon": "MBARI",
    "Ventana": "MBARI",
    "prnpr2018": "Hakai",
    "hakai_rov": "Hakai",
    "Hakai_ROV_2019": "Hakai",
    "hakai_video": "Hakai",
    "Hakai_Video_2020": "Hakai",
    "hogkins_h1685": "Merlin Best",
    "dellwood_h1682": "Merlin Best",
    "dellwood_h1683": "Merlin Best",
    "dellwood_south_h1690": "Merlin Best",
    "explorer_h1691": "Merlin Best",
    "sgaan_h1684": "Merlin Best",
    "sgaan_h1686": "Merlin Best",
    "LISMARC12_SEABOSS": "MGDS",
    "LISMARC12_ISIS": "MGDS",
    "LISMARC13_SEABOSS": "MGDS",
    "LISMARC13_ROV": "MGDS",
    "mgds": "MGDS",
    "bedford": "SEAM",
    "Bedford_2017": "SEAM",
    "bay_of_fundy": "SEAM",
    "Bay_of_Fundy_2019": "SEAM",
    "st_anns_bank": "SEAM",
    "King_George_Bransfield_2018": "USAP-DC",
    "lmg1311": "USAP-DC",
    "lmg1703": "USAP-DC",
    "nbp1402": "USAP-DC",
    "nbp1502": "USAP-DC",
    "crocker2014": "USGS",
    "Crocker_2014": "USGS",
    "frrp2011": "USGS",
    "frrp_2011": "USGS",
    "nantuckett": "USGS",
    "pulley-ridge": "USGS",
    "pulley_ridge": "USGS",
    "Pulley_Ridge_2003": "USGS",
    "tortugas2009": "USGS",
    "Tortugas_2009": "USGS",
    "tortugas2011": "USGS",
    "Tortugas_2011": "USGS",
    "AT18-12": "WHOI",
    "chesterfield": "Ben Misiuk",
    "frobisher": "Ben Misiuk",
    "qik": "Ben Misiuk",
    "Qikiqtarjuaq": "Ben Misiuk",
    "wager": "Ben Misiuk",
}
ORG2COLOR = {
    "DFO": "#FB9A99",  # l.red / pink
    "FathomNet": "#a65628",  # brown
    "IMOS": "#4daf4a",  # green
    "SQUIDLE/IMOS": "#4daf4a",
    "MBARI": "#805B2F",  # MBARI logo teal-blue: #004360, #7A7064
    "NOAA": "#0078BC",  # NOAA logo db/lb: #243C72, #0078BC
    "NRCan": "#D32823",  # Canadian flag red: #D32823
    "PANGAEA": "#009C84",  # PANGAEA Website d.teal/l.teal: #004D60 009C84
    "RLS": "#984ea3",  # purple
    "SQUIDLE": "#ff7f00",  # orange
    "SOI": "#FECA2D",  # SOI logo y/b/lg: #FECA2D #0B61BE #C4C52A
    "USGS": "#00264C",  # USGS d.blue: #00264C
    "Other": "#606060",  # grey
    "Research groups": "#6a6a6a",  # "#670575",  # purple
    "4D Oceans": "#6a6a6a",  # Research groups
    "AADC": "#057002",  # banner blue #01426A; banner blue #09588C
    "Alex Schimel": "#232321",  # dark grey NGU #232321; Norway blue #00205A
    "NGU": "#232321",  # dark grey NGU #232321; Norway blue #00205A
    "Bastos Lab": "#6a6a6a",  # Research groups
    "Ben Misiuk": "#6a6a6a",  # Research groups
    "EAC": "#6a6a6a",  # Research groups
    "Hakai": "#cf5108",  # Logo: #B52025
    "Merlin Best": "#6a6a6a",  # Research groups
    "MGDS": "#114891",  # logo blue #114891; secondary grey #3F465C
    "SEAM": "#6a6a6a",  # Research groups
    "USAP-DC": "#024A61",  # logo light teal #92B9C5; secondary teal #024A61
    "WHOI": "#114891",  # MGDS
}
ORG2COLOR_BW = {
    "DFO": "#FB9A99",
    "FathomNet": "#a65628",
    "IMOS": "#52BDEC",  # IMOS logo lb/b: #52BDEC #3A6F8F
    "SQUIDLE/IMOS": "#52BDEC",
    "MBARI": "#805B2F",  # Secondary colour brown: "#7A7064", but needed to increase saturation
    "NOAA": "#0078BC",
    "NRCan": "#D32823",
    "PANGAEA": "#009C84",
    "RLS": "#984ea3",
    "SQUIDLE": "#ff7f00",
    "SOI": "#FECA2D",
    "USGS": "#4daf4a",  # green
    "Other": "#909090",  # grey
    "Research groups": "#8E8E8E",  # "#670575",  # purple
    "4D Oceans": "#8E8E8E",  # Research groups
    "AADC": "#057002",  # banner blue #01426A; banner blue #09588C
    "Alex Schimel": "#232321",  # dark grey NGU #232321; Norway blue #00205A
    "NGU": "#232321",  # dark grey NGU #232321; Norway blue #00205A
    "Bastos Lab": "#8E8E8E",  # Research groups
    "Ben Misiuk": "#8E8E8E",  # Research groups
    "EAC": "#8E8E8E",  # Research groups
    "Hakai": "#cf5108",  # Logo: #B52025
    "Merlin Best": "#8E8E8E",  # Research groups
    "MGDS": "#114891",  # logo blue #114891; secondary grey #3F465C
    "SEAM": "#8E8E8E",  # Research groups
    "USAP-DC": "#92B9C5",  # logo light teal #92B9C5; secondary teal #024A61
    "WHOI": "#114891",  # MGDS
}


def rgb_white2alpha(
    rgb,
    ensure_increasing=False,
    ensure_linear=False,
    lsq_linear=False,
):
    """
    Convert a set of RGB colors to RGBA with maximum transparency.

    The transparency is maximised for each color individually, assuming
    that the background is white.

    Parameters
    ----------
    rgb : array_like shaped (N, 3)
        Original colors.
    ensure_increasing : bool, default=False
        Ensure that alpha values increase monotonically.
    ensure_linear : bool, default=False
        Ensure alpha values increase linear from initial to final value.
    lsq_linear : bool, default=False
        Use least-squares linear fit for alpha.

    Returns
    -------
    rgba : numpy.ndarray shaped (N, 4)
        Colors with maximum possible transparency, assuming a white
        background.
    """
    # The most transparent alpha we can use is given by the min of RGB
    # Convert it from saturation to opacity
    alpha = 1.0 - np.min(rgb, axis=1)
    if lsq_linear:
        # Make a least squares fit for alpha
        indices = np.arange(len(alpha))
        A = np.stack([indices, np.ones_like(indices)], axis=-1)
        m, c = np.linalg.lstsq(A, alpha, rcond=None)[0]
        # Use our least squares fit to generate a linear alpha
        alpha = c + m * indices
        alpha = np.clip(alpha, 0, 1)
    elif ensure_linear:
        # Use a linearly increasing/decreasing alpha from start to finish
        alpha = np.linspace(alpha[0], alpha[-1], rgb.shape[0])
    elif ensure_increasing:
        # Let's also ensure the alpha value is monotonically increasing
        a_max = alpha[0]
        for i, a in enumerate(alpha):
            alpha[i] = a_max = np.maximum(a, a_max)
    alpha = np.expand_dims(alpha, -1)
    # Rescale colors to discount the white that will show through from transparency
    rgb = rgb + alpha - 1
    rgb = np.divide(rgb, alpha, out=np.zeros_like(rgb), where=(alpha > 0))
    rgb = np.clip(rgb, 0, 1)
    # Concatenate our alpha channel
    rgba = np.concatenate((rgb, alpha), axis=1)
    return rgba


def extend_rgba_to_transparent(rgba, append=None):
    """
    Extend RGBA colors to reach completely transparent.

    Parameters
    ----------
    rgba : numpy.ndarray
        Shaped ``(n, 4)``.
    append : bool or None, optional
        Whether to append (if ``True``) or prepend (if ``False``) additional colors.
        If ``None``, colors will be added to whichever end has the smallest alpha value.

    Returns
    -------
    rgba : numpy.ndarray
        Shaped ``(n + k, 4)``.
    """
    alpha = rgba[:, -1]
    if append is None:
        append = alpha[-1] < alpha[0]
    if append:
        alpha = alpha[::-1]
    # Check if we are already reaching full transparency
    if alpha[0] == 0:
        return rgba
    # Get an exponential moving average of the gradient
    alpha_rates = (alpha[1:] - alpha[0]) / np.arange(1, len(alpha))
    weights = np.power(0.5, np.arange(1, len(alpha)))
    weights /= sum(weights)
    alpha_rate_estimate = np.abs(sum(alpha_rates * weights))
    # Get alpha values to use for the extra colours
    n_extra = max(1, int(np.round(0.9 * alpha[0] / alpha_rate_estimate)))
    alpha_extra = np.linspace(0, alpha[0], n_extra + 1)[:-1]
    # Add extra colours
    if append:
        rgba_extra = np.tile(rgba[-1], [n_extra, 1])
        rgba_extra[:, -1] = alpha_extra[::-1]
        rgba = np.concatenate([rgba, rgba_extra])
    else:
        rgba_extra = np.tile(rgba[0], [n_extra, 1])
        rgba_extra[:, -1] = alpha_extra
        rgba = np.concatenate([rgba_extra, rgba])
    return rgba


def cmap_white2alpha(
    name,
    extend_to_transparent=False,
    ensure_increasing=False,
    ensure_linear=False,
    lsq_linear=False,
    register=False,
):
    """
    Add as much transparency as possible to a colormap, assuming white background.

    See https://stackoverflow.com/a/68809469/1960959

    Parameters
    ----------
    name : str
        Name of builtin (or registered) colormap.
    extend_to_transparent : bool, default=False
        Extend the colormap to reach full transparency.
    ensure_increasing : bool, default=False
        Ensure that alpha values are strictly increasing.
    ensure_linear : bool, default=False
        Ensure alpha values increase linear from initial to final value.
    lsq_linear : bool, default=False
        Use least-squares linear fit for alpha.
    register : bool, default=False
        Whether to register the new colormap.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap with alpha set as low as possible.
    """
    # Fetch the cmap callable
    cmap = matplotlib.cm.get_cmap(name)
    # Get the colors out from the colormap LUT
    rgb = cmap(np.arange(cmap.N))[:, :3]  # N-by-3
    # Convert white to alpha
    rgba = rgb_white2alpha(
        rgb,
        ensure_increasing=ensure_increasing,
        ensure_linear=ensure_linear,
        lsq_linear=lsq_linear,
    )
    if extend_to_transparent:
        rgba = extend_rgba_to_transparent(rgba)
    # Create a new Colormap object
    new_name = name + "_white2alpha"
    cmap_alpha = matplotlib.colors.ListedColormap(rgba, name=new_name)
    if register:
        matplotlib.cm.register_cmap(name=new_name, cmap=cmap_alpha)
    return cmap_alpha


def show_land_sea_features(ax=None, land=None, water=None):
    """
    Show global land, sea, and lakes in one flat colour each.

    Parameters
    ----------
    ax : matplotlib.Axes, optional
        The axes onto which land, sea, and lakes will be added.
    """
    if ax is None:
        ax = plt.gca()

    if land is None:
        land = cartopy.feature.COLORS["land"]
    if water is None:
        water = cartopy.feature.COLORS["water"]

    scale = "10m"  # use data at this scale
    land = cartopy.feature.NaturalEarthFeature(
        "physical",
        "land",
        scale=scale,
        edgecolor="none",
        facecolor=land,
    )
    ocean = cartopy.feature.NaturalEarthFeature(
        "physical",
        "ocean",
        scale=scale,
        edgecolor="none",
        facecolor=water,
    )
    lakes = cartopy.feature.NaturalEarthFeature(
        "physical",
        "lakes",
        scale=scale,
        edgecolor="none",
        facecolor=water,
    )
    ax.add_feature(land)
    ax.add_feature(ocean)
    ax.add_feature(lakes)


def plot_samples(
    latitude,
    longitude,
    projection="EqualEarth",
    figsize=(25, 8),
    show_map=True,
    s=1,
    c="r",
    **kwargs,
):
    """
    Scatter plot of dataset sample locations overlaid on a map.

    Parameters
    ----------
    latitude : array-like
        Latitude of the coordinates for the samples.
    longitude : array-like
        Longitude of the coordinates for the samples.
    projection : str or cartopy.crs.Projection, default="EqualEarth"
        The projection to use for the map.
    figsize : tuple, default=(25, 8)
        Size of the figure to create.
    show_map : bool, default=True
        Whether to show the map, otherwise coastlines are drawn instead.
    s : float, default=1
        The marker size in points**2.
    color : array-like or list of colors or color, default="r"
        The marker colors.
    **kwargs
        Additional arguments as per :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used to render the plot.
    """
    # Get the projection
    if isinstance(projection, str):
        projection = getattr(cartopy.crs, projection)
    elif not isinstance(projection, callable):
        raise ValueError("projection must be either a string or callable.")
    projection = projection()

    # Create a plot using the projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=projection)

    # Display gridlines and either the world map, or outline of coastlines
    if show_map in ["none", "coastlines"] or not show_map:
        ax.gridlines()
    else:
        ax.gridlines(color="w")
    if show_map == "none":
        pass
    elif show_map == "two-tone":
        show_land_sea_features(ax)
    elif show_map == "two-tone-alt":
        water = np.array([168, 192, 224]) / 255
        show_land_sea_features(ax, water=water)
    elif show_map == "coastlines" or not show_map:
        ax.coastlines()
    else:
        ax.stock_img()

    # Plot the datapoints
    ax.scatter(
        longitude,
        latitude,
        s=s,
        c=c,
        marker=".",
        edgecolors="none",
        zorder=999,
        transform=cartopy.crs.PlateCarree(),
        **kwargs,
    )

    # Ensure the full map is shown
    ax.set_global()
    return ax


def plot_kde(
    latitude,
    longitude,
    projection="EqualEarth",
    figsize=(25, 8),
    show_map=False,
    cmap="Reds",
    extend_cmap=True,
    n_grid=181,
    **kwargs,
):
    """
    Contour plot of KDE overlaid on a map.

    Parameters
    ----------
    latitude : array-like
        Latitude of the coordinates for the samples.
    longitude : array-like
        Longitude of the coordinates for the samples.
    projection : str or cartopy.crs.Projection, default="EqualEarth"
        The projection to use for the map.
    figsize : tuple, default=(25, 8)
        Size of the figure to create.
    show_map : bool, default=True
        Whether to show the map, otherwise coastlines are drawn instead.
    cmap : str, default="Reds"
        The colormap to use.
    extend_cmap : bool, default=True
        Whether to extend the colormap to go to fully transparent.
    n_grid : int, default=181
        Number of grid points to use in the longitude dimension. Twice as many
        samples will be used for the latitude.
    **kwargs
        Additional arguments as per :func:`fit_kde_spherical`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used to render the plot.
    """
    # Get the projection
    if isinstance(projection, str):
        projection = getattr(cartopy.crs, projection)
    elif not isinstance(projection, callable):
        raise ValueError("projection must be either a string or callable.")
    projection = projection()

    # Redact data down to finite values
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    valid = (~np.isnan(latitude)) & (~np.isnan(longitude))
    latitude = latitude[valid]
    longitude = longitude[valid]

    # Build transparent colormap to support plotting overlaid on the world map
    if isinstance(cmap, str):
        cmap = cmap_white2alpha(
            cmap, extend_to_transparent=extend_cmap, ensure_increasing=True
        )

    # Fit kernel density estimator and use it to measure samples on a grid
    X, Y, Z = kde_tools.fit_meshgrid_kde_spherical(
        latitude, longitude, n_grid=n_grid, **kwargs
    )

    # Create a plot using the projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=projection)

    # Display gridlines and either the world map, or outline of coastlines
    if show_map in ["none", "coastlines"] or not show_map:
        ax.gridlines()
    else:
        ax.gridlines(color="w")
    if show_map == "none":
        pass
    elif show_map == "two-tone":
        show_land_sea_features(ax)
    elif show_map == "two-tone-alt":
        water = np.array([168, 192, 224]) / 255
        show_land_sea_features(ax, water=water)
    elif show_map == "coastlines" or not show_map:
        ax.coastlines()
    else:
        ax.stock_img()

    # Plot the datapoints
    contours = ax.contourf(
        X,
        Y,
        Z,
        cmap=cmap,
        levels=np.linspace(0, Z.max(), 25),
        zorder=10,
        transform=cartopy.crs.PlateCarree(),
        antialiased=True,
    )
    hcb = plt.colorbar(contours)
    hcb.set_label("Kernel Density Estimate")

    # Ensure the full map is shown
    ax.set_global()
    return ax


def row2organization(row, merge_squidle_imos=True):
    """
    Map from row to organization name.

    Parameters
    ----------
    row : dict
        With fields ``"dataset"`` and ``"url"``.

    Returns
    -------
    organization : str
        Name of the organization responsible for the dataset.
    """
    if row["dataset"] in DATASET2ORG:
        return DATASET2ORG[row["dataset"]]
    if row["dataset"].lower() in DATASET2ORG:
        return DATASET2ORG[row["dataset"].lower()]
    if row["dataset"].lower().startswith("pangaea"):
        return "PANGAEA"
    if row["dataset"].lower().startswith("nrcan"):
        return "NRCan"
    if row["dataset"].lower().startswith("rls_") or row["dataset"].lower().startswith(
        "rls "
    ):
        return "RLS"
    if row["dataset"].lower().startswith("fathomnet"):
        return "FathomNet"
    if row["dataset"].startswith("FK200429"):
        return "SOI"

    if "url" not in row or pd.isna(row["url"]):
        if "repository" in row and not pd.isna(row["repository"]):
            if merge_squidle_imos and row["repository"].lower() == "squidle":
                return "SQUIDLE/IMOS"
            return row["repository"]
        if "source" in row and not pd.isna(row["source"]):
            if merge_squidle_imos and row["repository"].lower() == "squidle":
                return "SQUIDLE/IMOS"
            return row["source"]
        return "other"

    if row["url"].startswith("https://s3-ap-southeast-2.amazonaws.com/imos-data"):
        if merge_squidle_imos:
            return "SQUIDLE/IMOS"
        return "IMOS"
    if row["url"].startswith("https://www.nodc.noaa.gov") or row["url"].startswith(
        "https://accession.nodc.noaa.gov"
    ):
        return "NOAA"
    if row["url"].startswith("http://rls.tpac.org.au"):
        return "RLS"
    if (
        row["url"].startswith("http://soi-dfo-data.storage.googleapis.com")
        or row["url"].startswith("http://soi-uos-data.storage.googleapis.com")
        or "fkdata.storage.googleapis.com/FK" in row["url"]
    ):
        return "SOI"

    if "repository" in row and not pd.isna(row["repository"]):
        org = row["repository"]
    elif "source" in row and not pd.isna(row["source"]):
        org = row["source"]
    else:
        org = "other"

    if merge_squidle_imos and org == "squidle":
        return "SQUIDLE/IMOS"

    return org


def plot_samples_by_organization(
    df,
    projection="EqualEarth",
    figsize=(25, 8),
    show_map=True,
    s=10,
    legend=True,
    **kwargs,
):
    """
    Scatter plot of sample locations, coloured by organization, overlaid on a map.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ``"latitude"``, ``"longitude"``.
    projection : str or cartopy.crs.Projection, default="EqualEarth"
        The projection to use for the map.
    figsize : tuple, default=(25, 8)
        Size of the figure to create.
    show_map : bool, default=True
        Whether to show the map, otherwise coastlines are drawn instead.
    s : float, default=1
        The marker size in points**2.
    legend : bool, default=True
        Whether to show the legend.
    **kwargs
        Additional arguments as per :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used to render the plot.
    """
    # Get the projection
    if isinstance(projection, str):
        projection = getattr(cartopy.crs, projection)
    elif not isinstance(projection, callable):
        raise ValueError("projection must be either a string or callable.")
    projection = projection()

    # Switch the colormap for better contrast against the background
    org2color = ORG2COLOR if show_map else ORG2COLOR_BW

    # Work on a copy of the dataframe
    df = df.copy()
    # Remove samples which are missing coordinate information
    df = df[~pd.isna(df["latitude"]) & ~pd.isna(df["longitude"])]
    # Determine organization and colour
    df["organization"] = df.apply(row2organization, axis=1)
    df.loc[df["organization"] == "WHOI", "organization"] = "MGDS"  # Soft override
    df["org_color"] = df["organization"].apply(lambda x: org2color.get(x, "#888"))

    for k in ["Other", "Research groups"]:
        df.loc[df["org_color"] == org2color[k], "organization"] = k

    # Create a plot using the projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=projection)

    # Display gridlines and either the world map, or outline of coastlines
    if show_map in ["none", "coastlines"] or not show_map:
        ax.gridlines()
    else:
        ax.gridlines(color="w")
    if show_map == "none":
        pass
    elif show_map == "two-tone":
        show_land_sea_features(ax)
    elif show_map == "two-tone-alt":
        water = np.array([168, 192, 224]) / 255
        show_land_sea_features(ax, water=water)
    elif show_map == "coastlines" or not show_map:
        ax.coastlines()
    else:
        ax.stock_img()

    # Make legend entries
    df_singles = df.drop_duplicates(subset=["organization"]).sort_values("organization")

    if legend:
        # Plot a marker for each organization value with a size of 0
        leg_handles = []
        leg_labels = []
        for _, row in df_singles.iterrows():
            if (
                row["org_color"] == org2color["Other"]
                and row["organization"] != "Other"
            ):
                continue
            hsc = ax.scatter(
                row["longitude"],
                row["latitude"],
                c=row["org_color"],
                s=0,
                transform=cartopy.crs.PlateCarree(),
            )
            leg_handles.append(hsc)
            leg_labels.append(row["organization"])

        # Move the "Other" label to the end of the list
        for key in ["Misc", "Other", "other", "Other research labs"]:
            if key in leg_labels:
                i = leg_labels.index(key)
                leg_handles.append(leg_handles.pop(i))
                leg_labels.append(leg_labels.pop(i))
        # Show legend
        hlegend = ax.legend(
            leg_handles,
            leg_labels,
            fontsize=16,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        # Set the size of the markers in the legend to be large
        for h_entry in hlegend.legendHandles:
            h_entry._sizes = [50]
        hlegend.get_frame().set_edgecolor("k")

    # Shuffle the order of the samples, so overlapping sites can show through
    df_rand = df.sample(frac=1)

    # Plot the datapoints
    ax.scatter(
        df_rand["longitude"],
        df_rand["latitude"],
        s=s,
        c=df_rand["org_color"],
        marker=".",
        edgecolors="none",
        zorder=999,
        transform=cartopy.crs.PlateCarree(),
        **kwargs,
    )

    # Ensure the full map is shown
    ax.set_global()
    return ax
