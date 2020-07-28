import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from ._colormap import _CMAP_QUAL
from ._core import _SetupImage


def imgplot(
    data,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    dx=None,
    units=None,
    cbar=True,
    cbar_label=None,
    cbar_fontdict=None,
    cbar_ticks=None,
    showticks=False,
    title=None,
    title_fontdict=None,
):
    """

    Plot data as a 2-D image with options to add scalebar, colorbar, title.

    Args:
        data: Image data (array-like). Supported array shapes are all
            `matplotlib.pyplot.imshow` array shapes
        ax (`matplotlib.axes.Axes`, optional): Matplotlib axes to plot image on.
            If None, figure and axes are auto-generated. Defaults to None.
        cmap (str or `matplotlib.colors.Colormap`, optional): Colormap for image.
            Can be a seaborn-image colormap or default matplotlib colormaps or
            any other colormap converted to a matplotlib colormap. Defaults to None.
        vmin (float, optional): Minimum data value that colormap covers. Defaults to None.
        vmax (float, optional): Maximum data value that colormap covers. Defaults to None.
        dx (float, optional): Size per pixel of the image data. If scalebar
            is required, `dx` and `units` must be sepcified. Defaults to None.
        units (str, optional): Units of `dx`. Defaults to None.
        cbar (bool, optional): Specify if a colorbar is required or not.
            Defaults to True.
        cbar_label (str, optional): Colorbar label. Defaults to None.
        cbar_fontdict (dict, optional): Font specifications for colorbar label - `cbar_label`.
            Defaults to None.
        cbar_ticks (list, optional): List of colorbar ticks. If None, min and max of
            the data are used. If `vmin` and `vmax` are specified, `vmin` and `vmax` values
            are used for colorbar ticks. Defaults to None.
        showticks (bool, optional): Show image x-y axis ticks. Defaults to False.
        title (str, optional): Image title. Defaults to None.
        title_fontdict (dict, optional): [Font specifications for `title`. Defaults to None.

    Raises:
        TypeError: if `cmap` is not str or `matplotlib.colors.Colormap`
        TypeError: if `ax` is not `matplotlib.axes.Axes`
        TypeError: if `cbar` is not bool
        TypeError: if `cbar_label` is not str
        TypeError: if `cbar_fontdict` is not dict
        TypeError: if `showticks` is not bool
        TypeError: if `title` is not str
        TypeError: if `title_fontdict` is not dict

    Returns:
        (tuple): tuple containing:
            (`matplotlib.figure.Figure`): Matplotlib figure.
            (`matplotlib.axes.Axes`): Matplotlib axes where the image is drawn.
            (`matplotlib.axes.Axes`): Colorbar axes

    Example:
        >>> import seaborn_image as isns
        >>> isns.imgplot(data)
        >>> isns.imgplot(data, dx=2, units="nm") # add a scalebar
        >>> isns.imgplot(data, cmap="deep") # specify a colormap


    """

    # add vmin, vmax, dx, units to checks
    if cmap is not None:
        if not isinstance(cmap, (str, Colormap)):
            raise TypeError
    if ax is not None:
        if not isinstance(ax, Axes):
            raise TypeError
    if not isinstance(cbar, bool):
        raise TypeError
    if cbar_label is not None:
        if not isinstance(cbar_label, str):
            raise TypeError
    if cbar_fontdict is not None:
        if not isinstance(cbar_fontdict, dict):
            raise TypeError
    if not isinstance(showticks, bool):
        raise TypeError
    if title is not None:
        if not isinstance(title, str):
            raise TypeError
    if title_fontdict is not None:
        if not isinstance(title_fontdict, dict):
            raise TypeError

    img_plotter = _SetupImage(
        data=data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        dx=dx,
        units=units,
        cbar=cbar,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        title=title,
        fontdict=title_fontdict,
    )

    f, ax, cax = img_plotter.plot()

    return f, ax, cax


def imghist(
    data,
    cmap=None,
    bins=500,
    vmin=None,
    vmax=None,
    dx=None,
    units=None,
    cbar=True,
    cbar_label=None,
    cbar_fontdict=None,
    cbar_ticks=None,
    showticks=False,
    title=None,
    title_fontdict=None,
):
    """Plot data as a 2-D image with histogram showing the distribution of
    the data. Options to add scalebar, colorbar, title.

    Args:
        data: Image data (array-like). Supported array shapes are all
            `matplotlib.pyplot.imshow` array shapes
        cmap (str or `matplotlib.colors.Colormap`, optional): Colormap for image.
            Can be a seaborn-image colormap or default matplotlib colormaps or
            any other colormap converted to a matplotlib colormap. Defaults to None.
        bins (int, optional): Number of histogram bins. Defaults to 500.
        vmin (float, optional): Minimum data value that colormap covers. Defaults to None.
        vmax (float, optional): Maximum data value that colormap covers. Defaults to None.
        dx (float, optional): Size per pixel of the image data. If scalebar
            is required, `dx` and `units` must be sepcified. Defaults to None.
        units (str, optional): Units of `dx`. Defaults to None.
        cbar (bool, optional): Specify if a colorbar is required or not.
            Defaults to True.
        cbar_label (str, optional): Colorbar label. Defaults to None.
        cbar_fontdict (dict, optional): Font specifications for colorbar label - `cbar_label`.
            Defaults to None.
        cbar_ticks (list, optional): List of colorbar ticks. If None, min and max of
            the data are used. If `vmin` and `vmax` are specified, `vmin` and `vmax` values
            are used for colorbar ticks. Defaults to None.
        showticks (bool, optional): Show image x-y axis ticks. Defaults to False.
        title (str, optional): Image title. Defaults to None.
        title_fontdict (dict, optional): [Font specifications for `title`. Defaults to None.

    Raises:
        TypeError: if `bins` is not int

    Returns:
        (tuple): tuple containing:
            (`matplotlib.figure.Figure`): Matplotlib figure.
            (tuple): tuple containing:
                (`matplotlib.axes.Axes`): Matplotlib axes where the image is drawn.
                (`matplotlib.axes.Axes`): Matplotlib axes where the histogram is drawn.
            (`matplotlib.axes.Axes`): Colorbar axes

    Example:
        >>> import seaborn_image as isns
        >>> isns.imghist(data)
        >>> isns.imghist(data, bins=300) # specify the number of histogram bins
        >>> isns.imghist(data, dx=2, units="nm") # add a scalebar
        >>> isns.imghist(data, cmap="deep") # specify a colormap
    """

    if not isinstance(bins, int):
        raise TypeError("'bins' must be a positive integer")
    if not bins > 0:
        raise ValueError("'bins' must be a positive integer")

    f = plt.figure(figsize=(10, 6))  # TODO make figsize user defined
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], figure=f)

    ax1 = f.add_subplot(gs[0])

    f, ax1, cax = imgplot(
        data,
        ax=ax1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        dx=dx,
        units=units,
        cbar=cbar,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        title=title,
        title_fontdict=title_fontdict,
    )

    ax2 = f.add_subplot(gs[1], sharey=cax)

    n, bins, patches = ax2.hist(
        data.ravel(), bins=bins, density=True, orientation="horizontal"
    )

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_frame_on(False)

    if cmap is None:
        cm = _CMAP_QUAL.get("deep").mpl_colormap
    else:
        if cmap in _CMAP_QUAL.keys():
            cm = _CMAP_QUAL.get(cmap).mpl_colormap
        else:
            cm = plt.cm.get_cmap(cmap)

    bin_centers = bins[:-1] + bins[1:]

    # scale values to interval [0,1]
    col = bin_centers - np.min(bin_centers)
    col /= np.max(col)

    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))

    return f, (ax1, ax2), cax
