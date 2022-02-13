"""
Plotting utilities.
"""

import cartopy.crs
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from . import kde_tools


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


def plot_samples(
    latitude,
    longitude,
    projection="EqualEarth",
    figsize=(25, 8),
    show_map=True,
    alpha=0.15,
    s=1,
    color="r",
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
    alpha : float, default=0.15
        The alpha blending value, between 0 (transparent) and 1 (opaque).
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
    ax.gridlines()
    if show_map:
        ax.stock_img()
    else:
        ax.coastlines()

    # Plot the datapoints
    ax.scatter(
        longitude,
        latitude,
        s=s,
        color=color,
        alpha=alpha,
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
        latitude, longitude, n_grid=101, **kwargs
    )

    # Create a plot using the projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=projection)

    # Display gridlines and either the world map, or outline of coastlines
    ax.gridlines()
    if show_map:
        ax.stock_img()
    else:
        ax.coastlines()

    # Plot the datapoints
    contours = ax.contourf(
        X,
        Y,
        Z,
        cmap=cmap,
        levels=np.linspace(0, Z.max(), 25),
        transform=cartopy.crs.PlateCarree(),
        antialiased=True,
    )
    hcb = plt.colorbar(contours)
    hcb.set_label("Kernel Density Estimate")

    # Ensure the full map is shown
    ax.set_global()
    return ax
