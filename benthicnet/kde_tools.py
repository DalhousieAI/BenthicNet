"""
Methods for spherical kernel density estimation.
"""

import numpy as np
from sklearn.neighbors import KernelDensity


def fit_kde_spherical(
    latitude,
    longitude,
    bw=1.0,
    verbose=0,
    kernel="gaussian",
    **kwargs,
):
    """
    Build a spherical kernel density estimate model on lat/long data.

    Parameters
    ----------
    latitude : array_like
        Latitude of samples, in degrees.
    longitude : array_like
        Longitude of samples, in degrees.
    bw : float or {"silverman", "scott"}, default=1.0
        The bandwidth (in degrees) to use, or name of a kernel bandwidth
        estimation method.
    verbose : int, default=0
        Verbosity level.
    **kwargs
        Additional keyword arguments as per :class:`sklearn.neighbors.KernelDensity`.

    Returns
    -------
    kde : sklearn.neighbors.KernelDensity
        Fit KDE object, which accepts coordinates as ``[lat, lon]`` (in radians)
        and returns the KDE at that location.
    """
    # Convert lat/long from degrees into radians
    x = np.radians(latitude)
    y = np.radians(longitude)
    xy = np.stack([x, y], axis=-1)
    # Bandwidth calculation
    n_samp, n_feat = xy.shape
    if isinstance(bw, float):
        # Convert bandwidth from degrees into radians
        bw = np.radians(bw)
    elif not isinstance(bw, str):
        raise ValueError(
            f"bw must be a float or a string, but {bw.__class__} instance was given"
        )
    elif bw.lower() == "silverman":
        bw = (n_samp * (n_feat + 2) / 4.0) ** (-1.0 / (n_feat + 4))  # silverman
        if verbose >= 1:
            print(f"Automatic bandwidth (Silverman): {bw}")
    elif bw.lower() == "scott":
        bw = n_samp ** (-1.0 / (n_feat + 4))  # scott
        if verbose >= 1:
            print(f"Automatic bandwidth (Scott): {bw}")
    else:
        raise ValueError(f"Unsupported bandwidth estimator: {bw}")
    if verbose >= 1:
        print(f"bandwidth: {bw} radians")
    # KDE
    kde = KernelDensity(
        bandwidth=bw,
        metric="haversine",
        kernel=kernel,
        algorithm="ball_tree",
        **kwargs,
    )
    kde.fit(xy)
    return kde


def meshgrid_kde_spherical(
    kde,
    return_scores=False,
    n_grid=91,
    extent=None,
):
    """
    Sample KDE values at points on a grid of longitude and latitude coordinates.

    Parameters
    ----------
    kde : sklearn.neighbors.KernelDensity
        Pretrained KDE object, whose ``score_samples`` attribute is a callable
        which accepts coordinates as ``[lat, lon]`` (in radians) and returns the
        KDE at that location.
    return_scores : bool, default=False
        Whether to return KDE scores (which is the log-density), or the actual
        density.
    n_grid : int, default=91
        Number of grid points to use in the longitude dimension. Twice as many
        samples will be used for the latitude.
    extent : array_like shaped (4, )
        The extent of the sampling grid in order
        ``(latitude_min, latitude_max, longitude_min, longitude_max)``.

    Returns
    -------
    X : numpy.ndarray shaped (2 * n_grid - 1, n_grid)
        Latitude values (in degrees) for the sampling grid.
    Y : numpy.ndarray shaped (2 * n_grid - 1, n_grid)
        Longitude values (in degrees) for the sampling grid.
    Z : numpy.ndarray shaped (2 * n_grid - 1, n_grid)
        Kernel density (or score if ``return_scores==True``) at the sampling
        grid coordinates.
    """
    # Extent
    if extent:
        xmin, xmax, ymin, ymax = extent
    else:
        xmin = -np.pi
        xmax = np.pi
        ymin = -np.pi / 2
        ymax = np.pi / 2
    # Mesh grid
    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, n_grid * 2 - 1), np.linspace(ymin, ymax, n_grid)
    )
    positions = np.stack([X.ravel(), Y.ravel()], axis=-1)
    # Swap latitude and longitude around
    positions = positions[:, ::-1]
    # Z heights
    Z = np.reshape(kde.score_samples(positions), X.shape)
    if not return_scores:
        Z = np.exp(Z)
    X = np.degrees(X)
    Y = np.degrees(Y)
    return X, Y, Z


def fit_meshgrid_kde_spherical(
    latitude,
    longitude,
    return_scores=True,
    n_grid=91,
    full_grid=True,
    **kwargs,
):
    """
    Fit a spherical KDE and then sample from it on a meshgrid.

    Parameters
    ----------
    latitude : array_like
        Latitude of samples, in degrees.
    longitude : array_like
        Longitude of samples, in degrees.
    return_scores : bool, default=False
        Whether to return KDE scores (which is the log-density), or the actual
        density.
    n_grid : int, default=91
        Number of grid points to use in the longitude dimension. Twice as many
        samples will be used for the latitude.
    full_grid : bool, default=True
        Whether to return samples across the whole world. Otherwise, the samples
        only cover the extent of the ``latitude`` and ``longitude``.
    **kwargs
        Additional keyword arguments as per :func:`fit_kde_spherical`.

    Returns
    -------
    X : numpy.ndarray shaped (2 * n_grid - 1, n_grid)
        Latitude values (in degrees) for the sampling grid.
    Y : numpy.ndarray shaped (2 * n_grid - 1, n_grid)
        Longitude values (in degrees) for the sampling grid.
    Z : numpy.ndarray shaped (2 * n_grid - 1, n_grid)
        Kernel density (or score if ``return_scores==True``) at the sampling
        grid coordinates.
    """
    kde = fit_kde_spherical(latitude, longitude, **kwargs)

    if full_grid:
        extent = None
    else:
        extent = [latitude.min(), latitude.max(), longitude.min(), longitude.max()]
        extent = np.radians(extent)

    return meshgrid_kde_spherical(
        kde, extent=extent, return_scores=return_scores, n_grid=n_grid
    )
