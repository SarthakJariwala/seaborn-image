from matplotlib.axes import Axes
from matplotlib.colors import Colormap

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

    f, ax = img_plotter.plot()

    return f, ax, data
