import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
from skimage.color import rgb2gray

from ._colormap import _CMAP_QUAL
from ._core import _SetupImage

__all__ = ["imgplot", "imghist"]


def imgplot(
    data,
    ax=None,
    cmap=None,
    gray=None,
    vmin=None,
    vmax=None,
    robust=False,
    perc=(2, 98),
    dx=None,
    units=None,
    dimension=None,
    describe=True,
    cbar=True,
    orientation="v",
    cbar_label=None,
    cbar_fontdict=None,
    cbar_ticks=None,
    showticks=False,
    despine=True,
    title=None,
    title_fontdict=None,
):
    """

    Plot data as a 2-D image with options to ignore outliers, add scalebar, colorbar, title.

    Args:
        data: Image data (array-like). Supported array shapes are all
            `matplotlib.pyplot.imshow` array shapes
        ax (`matplotlib.axes.Axes`, optional): Matplotlib axes to plot image on.
            If None, figure and axes are auto-generated. Defaults to None.
        cmap (str or `matplotlib.colors.Colormap`, optional): Colormap for image.
            Can be a seaborn-image colormap or default matplotlib colormaps or
            any other colormap converted to a matplotlib colormap. Defaults to None.
        gray (bool, optional): If True and data is RGB image, it will be converted to grayscale.
            If True and cmap is None, cmap will be set to "gray".
        vmin (float, optional): Minimum data value that colormap covers. Defaults to None.
        vmax (float, optional): Maximum data value that colormap covers. Defaults to None.
        robust (bool, optional): If True and vmin or vmax are None, colormap range is calculated
            based on the percentiles defined in `percentile` parameter. Defaults to False.
        perc (tuple or list, optional): If `robust` is True, colormap range is calculated based
            on the percentiles specified instead of the extremes. Defaults to (2,98) - 2nd and 98th
            percentiles for min and max values.
        dx (float, optional): Size per pixel of the image data. If scalebar
            is required, `dx` and `units` must be sepcified. Defaults to None.
        units (str, optional): Units of `dx`. Defaults to None.
        dimension (str, optional): dimension of `dx` and `units`.
            Options include (similar to `matplotlib_scalebar`):
                - "si" : scale bar showing km, m, cm, etc.
                - "imperial" : scale bar showing in, ft, yd, mi, etc.
                - "si-reciprocal" : scale bar showing 1/m, 1/cm, etc.
                - "angle" : scale bar showing °, ʹ (minute of arc) or ʹʹ (second of arc).
                - "pixel" : scale bar showing px, kpx, Mpx, etc.
            Defaults to None.
        describe (bool, optional): Brief statistical description of the data.
            Defaults to True.
        cbar (bool, optional): Specify if a colorbar is required or not.
            Defaults to True.
        orientation (str, optional): Specify the orientaion of colorbar.
            Option include :
                - 'h' or 'horizontal' for a horizontal colorbar to the bottom of the image.
                - 'v' or 'vertical' for a vertical colorbar to the right of the image.
            Defaults to 'v'.
        cbar_label (str, optional): Colorbar label. Defaults to None.
        cbar_fontdict (dict, optional): Font specifications for colorbar label - `cbar_label`.
            Defaults to None.
        cbar_ticks (list, optional): List of colorbar ticks. If None, min and max of
            the data are used. If `vmin` and `vmax` are specified, `vmin` and `vmax` values
            are used for colorbar ticks. Defaults to None.
        showticks (bool, optional): Show image x-y axis ticks. Defaults to False.
        despine (bool, optional): Remove axes spines from image axes as well as colorbar axes.
            Defaults to True.
        title (str, optional): Image title. Defaults to None.
        title_fontdict (dict, optional): [Font specifications for `title`. Defaults to None.

    Raises:
        TypeError: if `cmap` is not str or `matplotlib.colors.Colormap`
        TypeError: if `ax` is not `matplotlib.axes.Axes`
        TypeError: if `describe` is not bool
        TypeError: if `robust` is not bool
        TypeError: if `cbar` is not bool
        TypeError: if `orientation` is not str
        TypeError: if `cbar_label` is not str
        TypeError: if `cbar_fontdict` is not dict
        TypeError: if `showticks` is not bool
        TypeError: if `despine` is not bool
        TypeError: if `title` is not str
        TypeError: if `title_fontdict` is not dict
        AssertionError: if `len(perc)` is not equal to 2
        AssertionError: if the first element of `perc` is greater than the second

    Returns:
        (tuple): tuple containing:
            (`matplotlib.figure.Figure`): Matplotlib figure.
            (`matplotlib.axes.Axes`): Matplotlib axes where the image is drawn.
            (`matplotlib.axes.Axes`): Colorbar axes

    Example:
        >>> import seaborn_image as isns

        >>> isns.imgplot(data)

        >>> # add a scalebar
        >>> isns.imgplot(data, dx=2, units="nm")

        >>> # specify a colormap
        >>> isns.imgplot(data, cmap="deep")

        >>> # convert RGB image to grayscale
        >>> isns.imgplot(data, gray=True)

        >>> # exclude the outliers
        >>> isns.imgplot(data, robust=True)

        >>> # change colorbar orientation
        >>> isns.imgplot(data, orientation="h") # horizontal
        >>> isns.imgplot(data, orientation="v") # vertical

        >>> # add a colorbar label
        >>> isns.imgplot(data, cbar_label="Label (au)")
    """

    # TODO add vmin, vmax, dx, units to checks
    if cmap is not None:
        if not isinstance(cmap, (str, Colormap)):
            raise TypeError

    if ax is not None:
        if not isinstance(ax, Axes):
            raise TypeError

    if not isinstance(describe, bool):
        raise TypeError

    if not isinstance(robust, bool):
        raise TypeError("'robust' must be either True or False")

    if robust is True:
        assert len(perc) == 2
        assert perc[0] < perc[1]  # order should be (min, max)

    if not isinstance(cbar, bool):
        raise TypeError

    if not isinstance(orientation, str):
        raise TypeError

    if cbar_label is not None:
        if not isinstance(cbar_label, str):
            raise TypeError

    if cbar_fontdict is not None:
        if not isinstance(cbar_fontdict, dict):
            raise TypeError

    if not isinstance(showticks, bool):
        raise TypeError

    if not isinstance(despine, bool):
        raise TypeError

    if title is not None:
        if not isinstance(title, str):
            raise TypeError

    if title_fontdict is not None:
        if not isinstance(title_fontdict, dict):
            raise TypeError

    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            cbar = False  # set cbar to False if RGB image
            robust = False  # set robust to False if RGB image
            if gray is True:  # if gray is True, convert to grayscale
                data = rgb2gray(data)

    if gray is True and cmap is None:  # set colormap to gray only if cmap is None
        cmap = "gray"

    img_plotter = _SetupImage(
        data=data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        perc=perc,
        dx=dx,
        units=units,
        dimension=dimension,
        cbar=cbar,
        orientation=orientation,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        despine=despine,
        title=title,
        fontdict=title_fontdict,
    )

    f, ax, cax = img_plotter.plot()

    if describe:
        result = ss.describe(data.flatten())
        print(f"No. of Obs. : {result.nobs}")
        print(f"Min. Value : {result.minmax[0]}")
        print(f"Max. Value : {result.minmax[1]}")
        print(f"Mean : {result.mean}")
        print(f"Variance : {result.variance}")
        print(f"Skewness : {result.skewness}")

    return f, ax, cax


def imghist(
    data,
    cmap=None,
    bins=None,
    vmin=None,
    vmax=None,
    robust=False,
    perc=(2, 98),
    dx=None,
    units=None,
    dimension=None,
    describe=True,
    cbar=True,
    orientation="v",
    cbar_label=None,
    cbar_fontdict=None,
    cbar_ticks=None,
    showticks=False,
    despine=True,
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
        bins (int, optional): Number of histogram bins. Defaults to None. If None, 'auto'
            is used.
        vmin (float, optional): Minimum data value that colormap covers. Defaults to None.
        vmax (float, optional): Maximum data value that colormap covers. Defaults to None.
        robust (bool, optional): If True and vmin or vmax are None, colormap range is calculated
            based on the percentiles defined in `percentile` parameter. Defaults to False.
        perc (tuple or list, optional): If `robust` is True, colormap range is calculated based
            on the percentiles specified instead of the extremes. Defaults to (2,98) - 2nd and 98th
            percentiles for min and max values.
        dx (float, optional): Size per pixel of the image data. If scalebar
            is required, `dx` and `units` must be sepcified. Defaults to None.
        units (str, optional): Units of `dx`. Defaults to None.
        dimension (str, optional): dimension of `dx` and `units`.
            Options include :
                - "si" : scale bar showing km, m, cm, etc.
                - "imperial" : scale bar showing in, ft, yd, mi, etc.
                - "si-reciprocal" : scale bar showing 1/m, 1/cm, etc.
                - "angle" : scale bar showing °, ʹ (minute of arc) or ʹʹ (second of arc).
                - "pixel" : scale bar showing px, kpx, Mpx, etc.
            Defaults to None.
        describe (bool, optional): Brief statistical description of the data.
            Defaults to True.
        cbar (bool, optional): Specify if a colorbar is required or not.
            Defaults to True.
        orientation (str, optional): Specify the orientaion of colorbar.
            Option include :
                - 'h' or 'horizontal' for a horizontal colorbar and histogram to the bottom of the image.
                - 'v' or 'vertical' for a vertical colorbar and histogram to the right of the image.
            Defaults to 'v'.
        cbar_label (str, optional): Colorbar label. Defaults to None.
        cbar_fontdict (dict, optional): Font specifications for colorbar label - `cbar_label`.
            Defaults to None.
        cbar_ticks (list, optional): List of colorbar ticks. If None, min and max of
            the data are used. If `vmin` and `vmax` are specified, `vmin` and `vmax` values
            are used for colorbar ticks. Defaults to None.
        showticks (bool, optional): Show image x-y axis ticks. Defaults to False.
        despine (bool, optional): Remove axes spines from image axes as well as colorbar axes.
            Defaults to True.
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

    if bins is None:
        bins = "auto"
    else:
        if not isinstance(bins, int):
            raise TypeError("'bins' must be a positive integer")
        if not bins > 0:
            raise ValueError("'bins' must be a positive integer")

    if orientation in ["v", "vertical"]:
        orientation = "vertical"  # matplotlib doesn't support 'v'
        f = plt.figure(figsize=(10, 6))  # TODO make figsize user defined
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], figure=f)

    elif orientation in ["h", "horizontal"]:
        orientation = "horizontal"  # matplotlib doesn't support 'h'
        f = plt.figure(figsize=(6, 10))  # TODO make figsize user defined
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1], figure=f)

    else:
        raise ValueError(
            "'orientation' must be either : 'horizontal' or 'h' / 'vertical' or 'v'"
        )

    ax1 = f.add_subplot(gs[0])

    f, ax1, cax = imgplot(
        data,
        ax=ax1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        perc=perc,
        dx=dx,
        units=units,
        dimension=dimension,
        describe=describe,
        cbar=cbar,
        orientation=orientation,
        cbar_label=cbar_label,
        cbar_fontdict=cbar_fontdict,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        despine=despine,
        title=title,
        title_fontdict=title_fontdict,
    )

    if orientation == "vertical":
        ax2 = f.add_subplot(gs[1], sharey=cax)

        n, bins, patches = ax2.hist(
            data.ravel(), bins=bins, density=True, orientation="horizontal"
        )

    elif orientation == "horizontal":
        ax2 = f.add_subplot(gs[1], sharex=cax)

        n, bins, patches = ax2.hist(
            data.ravel(), bins=bins, density=True, orientation="vertical"
        )

    if not showticks:
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

    if despine:
        ax2.set_frame_on(False)

    if cmap is None:
        cm = get_cmap()
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
