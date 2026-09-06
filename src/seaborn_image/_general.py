import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import axes_size
from skimage.color import rgb2gray

from ._colormap import _CMAP_QUAL, _CMAP_EXTRA
from ._core import _SetupImage, _get_axes_divider
from .utils import is_documented_by

_IMGHIST_SIZE_ASPECT = 0.3
_IMGHIST_PAD_FRACTION = 0.5

__all__ = ["imgplot", "imghist", "imshow"]


def imgplot(
    data,
    ax=None,
    cmap=None,
    gray=None,
    vmin=None,
    vmax=None,
    alpha=None,
    origin=None,
    interpolation=None,
    norm=None,
    robust=False,
    perc=(2, 98),
    diverging=False,
    dx=None,
    units=None,
    dimension=None,
    describe=False,
    map_func=None,
    cbar=True,
    orientation="v",
    cbar_log=False,
    cbar_label=None,
    cbar_ticks=None,
    showticks=False,
    despine=None,
    extent=None,
    **kwargs,
):
    """Plot data as a 2-D image with options to ignore outliers, add scalebar, colorbar, title.

    Parameters
    ----------
    data : array-like
        Image data. Supported array shapes are all `matplotlib.pyplot.imshow` array shapes
    ax : `matplotlib.axes.Axes`, optional
        Matplotlib axes to plot image on. If None, figure and axes are auto-generated, by default None
    cmap : str or `matplotlib.colors.Colormap`, optional
        Colormap for image. Can be a seaborn-image colormap or default matplotlib colormaps or
        any other colormap converted to a matplotlib colormap, by default None
    gray : bool, optional
        If True and data is RGB image, it will be converted to grayscale.
        If True and cmap is None, cmap will be set to "gray", by default None
    vmin : float, optional
        Minimum data value that colormap covers, by default None
    vmax : float, optional
        Maximum data value that colormap covers, by default None
    alpha : float or array-like, optional
        `matplotlib.pyplot.imshow` alpha blending value from 0 (transparent) to 1 (opaque),
        by default None
    origin : str, optional
        Image origin, by default None
    interpolation : str, optional
        `matplotlib.pyplot.imshow` interpolation method used, by default None
    norm : `matplotlib.colors.Normalize`, optional
        `matplotlib` Normalize instance used to scale scalar data before
        mapping to colors using cmap
    robust : bool, optional
        If True and vmin or vmax are None, colormap range is calculated
        based on the percentiles defined in `perc` parameter, by default False
    perc : tuple or list, optional
        If `robust` is True, colormap range is calculated based
        on the percentiles specified instead of the extremes, by default (2, 98) -
        2nd and 98th percentiles for min and max values
    diverging : bool, optional
        If True, vmax and vmin are adjusted so they have the same absolute value, making the diverging
        color maps show 0 at the middle.
    dx : float, optional
        Size per pixel of the image data. Specifying `dx` and `units` adds a scalebar
        to the image, by default None
    units : str, optional
        Units of `dx`, by default None
    dimension : str, optional
        dimension of `dx` and `units`, by default None
        Options include (similar to `matplotlib_scalebar`):
            - "si" : scale bar showing km, m, cm, etc.
            - "imperial" : scale bar showing in, ft, yd, mi, etc.
            - "si-reciprocal" : scale bar showing 1/m, 1/cm, etc.
            - "angle" : scale bar showing °, ʹ (minute of arc) or ʹʹ (second of arc)
            - "pixel" : scale bar showing px, kpx, Mpx, etc.
    describe : bool, optional
        Brief statistical description of the data, by default False
    map_func : callable, optional
        Transform input image data using this function. All function arguments must be passed as kwargs.
    cbar : bool, optional
        Specify if a colorbar is to be added to the image, by default True.
        If `data` is RGB image, cbar is False
    orientation : str, optional
        Specify the orientaion of colorbar, by default "v".
        Options include :
            - 'h' or 'horizontal' for a horizontal colorbar to the bottom of the image.
            - 'v' or 'vertical' for a vertical colorbar to the right of the image.
    cbar_log : bool, optional
        Log scale colormap and colorbar
    cbar_label : str, optional
        Colorbar label, by default None
    cbar_ticks : list, optional
        List of colorbar ticks, by default None
    showticks : bool, optional
        Show image x-y axis ticks, by default False
    despine : bool, optional
        Remove axes spines from image axes as well as colorbar axes, by default None
    extent : floats (left, right, bottom, top), optional
        The bounding box in data coordinates that the image will fill.
        The image is stretched individually along x and y to fill the box.


    Returns
    -------
    `matplotlib.axes.Axes`
        Matplotlib axes where the image is drawn.

    Raises
    ------
    TypeError
        if `cmap` is not str or `matplotlib.colors.Colormap`
    TypeError
        if `ax` is not `matplotlib.axes.Axes`
    TypeError
        if `describe` is not bool
    TypeError
        if `robust` is not bool
    TypeError
        if `cbar` is not bool
    TypeError
        if `orientation` is not str
    TypeError
        if `cbar_label` is not str
    TypeError
        if `showticks` is not bool
    TypeError
        if `despine` is not bool
    AssertionError
        if `len(perc)` is not equal to 2
    AssertionError
        if the first element of `perc` is greater than the second

    Examples
    --------

    Plot image

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> img = isns.load_image("polymer")
        >>> isns.imgplot(img)

    Get image statistics

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img, describe=True)

    Add a scalebar

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img, dx=15, units="nm")

    Change colormap

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img, cmap="deep")

    Rescale colormap to exclude outliers while plotting

    Image with outliers

    .. plot::
        :context: close-figs

        >>> img_out = isns.load_image("polymer outliers")
        >>> isns.imgplot(img_out)

    Rescale colormap using `robust` parameter

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img_out, robust=True)

    Change percentile for robust parameter

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img_out, robust=True, perc=(0.5,99.5))

    Rescale colormap using the `diverging` parameter

    .. plot::
        :context: close-figs

        >>> img_standard = img - img.mean()
        >>> isns.imgplot(img_standard, diverging=True, cmap="seismic")

    Map a function to transform input image

    .. plot::
        :context: close-figs

        >>> from skimage.exposure import adjust_gamma
        >>> cells = isns.load_image("cells")[:, :, 32]
        >>> isns.imgplot(cells, map_func=adjust_gamma, gamma=0.5)

    Convert RGB image to grayscale

    .. plot::
        :context: close-figs

        >>> from skimage.data import astronaut
        >>> isns.imgplot(astronaut(), gray=True)

    Change colorbar orientation

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img, orientation="h") # horizontal

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img, orientation="v") # vertical

    Add colorbar label

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img, cbar_label="Height (nm)")

    Despine image and colorbar

    .. plot::
        :context: close-figs

        >>> isns.imgplot(img, despine=True)

    Change colorbar and colormap to log scale

    .. plot::
        :context: close-figs

        >>> pl = isns.load_image("fluorescence")
        >>> isns.imgplot(pl, cbar_log=True)

    Change image transparency

    .. plot::
        :context: close-figs

        >>> isns.imgplot(pl, alpha=0.75)
    """

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
        assert perc[0] < perc[1]

    if diverging:
        if vmax is not None:
            assert vmax > 0, "vmax must be greater than 0 when diverging=True"

        if vmin is not None:
            assert vmin < 0, "vmin must be lower than 0 when diverging=True"

    if not isinstance(cbar, bool):
        raise TypeError

    if not isinstance(orientation, str):
        raise TypeError

    if cbar_label is not None:
        if not isinstance(cbar_label, str):
            raise TypeError

    if cbar_log is not None:
        if not isinstance(cbar_log, bool):
            raise TypeError("must be a bool")

    if not isinstance(showticks, bool):
        raise TypeError

    if despine is not None:
        if not isinstance(despine, bool):
            raise TypeError

    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            cbar = False
            robust = False
            if gray is True:
                data = rgb2gray(data)

    if gray is True and cmap is None:
        cmap = "gray"

    if norm is None and cbar_log is True:
        norm = "cbar_log"

    if map_func is not None:
        if not callable(map_func):
            raise TypeError("`map_func` must be a callable function object")

        map_func_kwargs = {}
        map_func_kwargs.update(**kwargs)

        data = map_func(data, **map_func_kwargs)

    img_plotter = _SetupImage(
        data=data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        origin=origin,
        interpolation=interpolation,
        norm=norm,
        robust=robust,
        perc=perc,
        diverging=diverging,
        dx=dx,
        units=units,
        dimension=dimension,
        cbar=cbar,
        orientation=orientation,
        cbar_label=cbar_label,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        despine=despine,
        extent=extent,
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

    return ax


@is_documented_by(imgplot)
def imshow(data, **kwargs):

    ax = imgplot(data, **kwargs)

    return ax


def imghist(
    data,
    cmap=None,
    bins=None,
    vmin=None,
    vmax=None,
    alpha=None,
    origin=None,
    interpolation=None,
    norm=None,
    robust=False,
    perc=(2, 98),
    diverging=False,
    dx=None,
    units=None,
    dimension=None,
    describe=False,
    map_func=None,
    cbar=True,
    orientation="v",
    cbar_log=False,
    cbar_label=None,
    cbar_ticks=None,
    showticks=False,
    despine=None,
    height=5,
    aspect=1.75,
    **kwargs,
):
    """Plot data as a 2-D image with histogram showing the distribution of
    the data. Options to add scalebar, colorbar, title.

    Parameters
    ----------
    data : array-like
        Image data. Supported array shapes are all `matplotlib.pyplot.imshow` array shapes
    bins : int, optional
        Histogram bins, by default None. If None, `auto` is used.
    cmap : str or `matplotlib.colors.Colormap`, optional
        Colormap for image. Can be a seaborn-image colormap or default matplotlib colormaps or
        any other colormap converted to a matplotlib colormap, by default None
    gray : bool, optional
        If True and data is RGB image, it will be converted to grayscale.
        If True and cmap is None, cmap will be set to "gray", by default None
    vmin : float, optional
        Minimum data value that colormap covers, by default None
    vmax : float, optional
        Maximum data value that colormap covers, by default None
    alpha : float or array-like, optional
        `matplotlib.pyplot.imshow` alpha blending value from 0 (transparent) to 1 (opaque),
        by default None
    origin : str, optional
        Image origin, by default None
    interpolation : str, optional
        `matplotlib.pyplot.imshow` interpolation method used, by default None
    norm : `matplotlib.colors.Normalize`, optional
        `matplotlib` Normalize instance used to scale scalar data before
        mapping to colors using cmap
    robust : bool, optional
        If True and vmin or vmax are None, colormap range is calculated
        based on the percentiles defined in `perc` parameter, by default False
    perc : tuple or list, optional
        If `robust` is True, colormap range is calculated based
        on the percentiles specified instead of the extremes, by default (2, 98) -
        2nd and 98th percentiles for min and max values
    diverging : bool, optional
        If True, vmax and vmin are adjusted so they have the same absolute value, making the diverging
        color maps show 0 at the middle.
    dx : float, optional
        Size per pixel of the image data. Specifying `dx` and `units` adds a scalebar
        to the image, by default None
    units : str, optional
        Units of `dx`, by default None
    dimension : str, optional
        dimension of `dx` and `units`, by default None
        Options include (similar to `matplotlib_scalebar`):
            - "si" : scale bar showing km, m, cm, etc.
            - "imperial" : scale bar showing in, ft, yd, mi, etc.
            - "si-reciprocal" : scale bar showing 1/m, 1/cm, etc.
            - "angle" : scale bar showing °, ʹ (minute of arc) or ʹʹ (second of arc)
            - "pixel" : scale bar showing px, kpx, Mpx, etc.
    describe : bool, optional
        Brief statistical description of the data, by default False
    map_func : callable, optional
        Transform input image data using this function. All function arguments must be passed as kwargs.
    cbar : bool, optional
        Specify if a colorbar is to be added to the image, by default True.
        If `data` is RGB image, cbar is False
    orientation : str, optional
        Specify the orientaion of colorbar, by default "v".
        Options include :
            - 'h' or 'horizontal' for a horizontal colorbar to the bottom of the image.
            - 'v' or 'vertical' for a vertical colorbar to the right of the image.
    cbar_log : bool, optional
        Log scale colormap and colorbar
    cbar_label : str, optional
        Colorbar label, by default None
    cbar_ticks : list, optional
        List of colorbar ticks, by default None
    showticks : bool, optional
        Show image x-y axis ticks, by default False
    despine : bool, optional
        Remove axes spines from image axes as well as colorbar axes, by default None
    height : int or float, optional
        Height of the figure in inches, by default 5.
    aspect : int or float, optional
        Figure aspect used to set the figsize. For a vertical histogram,
        figsize is ``(height * aspect, height)``; for a horizontal histogram,
        figsize is ``(height, height * aspect)``. Defaults to 1.75 so a square
        image has room for the colorbar and histogram. The histogram is always
        sized to match the image along the shared axis (height for a vertical
        histogram, width for a horizontal one), independent of this value.

    Returns
    -------
    `matplotlib.figure.Figure`
        Matplotlib figure.

    Raises
    ------
    TypeError
        if `bins` is not a positive integer

    Examples
    --------

    Plot distribution alongside the image

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> img = isns.load_image("polymer")
        >>> isns.imghist(img)

    Change the orientation

    .. plot::
        :context: close-figs

        >>> isns.imghist(img, orientation="h")

    Change the number of bins

    .. plot::
        :context: close-figs

        >>> isns.imghist(img, bins=300)

    Change height and aspect ratio of the figure

    .. plot::
        :context: close-figs

        >>> isns.imghist(img, height=4, aspect=1.5)

    Get image statistics

    .. plot::
        :context: close-figs

        >>> isns.imghist(img, describe=True)

    Add a scalebar

    .. plot::
        :context: close-figs

        >>> isns.imghist(img, dx=15, units="nm")

    Map a function to transform input image

    .. plot::
        :context: close-figs

        >>> from skimage.exposure import adjust_gamma
        >>> cells = isns.load_image("cells")[:, :, 32]
        >>> isns.imghist(cells, map_func=adjust_gamma, gamma=0.5)

    Change colormaps

    .. plot::
        :context: close-figs

        >>> isns.imghist(img, cmap="ice")
    """

    if data.ndim > 2:
        raise ValueError(
            "Currently, `imghist` does not support images with more than 2 dimensions"
        )

    if bins is None:
        bins = "auto"
    else:
        if not isinstance(bins, int):
            raise TypeError("'bins' must be a positive integer")
        if not bins > 0:
            raise ValueError("'bins' must be a positive integer")

    if orientation in ["v", "vertical"]:
        orientation = "vertical"  # matplotlib doesn't support 'v'
        f = plt.figure(figsize=(height * aspect, height))

    elif orientation in ["h", "horizontal"]:
        orientation = "horizontal"  # matplotlib doesn't support 'h'
        f = plt.figure(figsize=(height, height * aspect))

    else:
        raise ValueError(
            "'orientation' must be either : 'horizontal' or 'h' / 'vertical' or 'v'"
        )

    ax1 = f.add_subplot(111)

    ax1 = imgplot(
        data,
        ax=ax1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        origin=origin,
        interpolation=interpolation,
        norm=norm,
        robust=robust,
        perc=perc,
        diverging=diverging,
        dx=dx,
        units=units,
        dimension=dimension,
        describe=describe,
        map_func=map_func,
        cbar=cbar,
        orientation=orientation,
        cbar_log=cbar_log,
        cbar_label=cbar_label,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        despine=despine,
        **kwargs,
    )

    cax = f.axes[1] if cbar and len(f.axes) > 1 else None

    _log = False
    if cbar_log is True:
        _log = True

    if robust:
        _data_min = np.nanpercentile(data, perc[0])
        _data_max = np.nanpercentile(data, perc[1])

        data_robust = data[(data > _data_min) & (data < _data_max)]
    else:
        data_robust = data

    divider = _get_axes_divider(ax1)

    extra_pad = axes_size.Fixed(0)

    if orientation == "vertical":
        hist_size = axes_size.AxesY(ax1, aspect=_IMGHIST_SIZE_ASPECT)
        pad = axes_size.Fraction(_IMGHIST_PAD_FRACTION, hist_size) + extra_pad
        share_kw = {"sharey": cax} if cax is not None else {}
        ax2 = divider.append_axes("right", size=hist_size, pad=pad, **share_kw)
        hist_orientation = "horizontal"

    else:
        hist_size = axes_size.AxesX(ax1, aspect=_IMGHIST_SIZE_ASPECT)
        pad = axes_size.Fraction(_IMGHIST_PAD_FRACTION, hist_size) + extra_pad
        share_kw = {"sharex": cax} if cax is not None else {}
        ax2 = divider.append_axes("bottom", size=hist_size, pad=pad, **share_kw)
        hist_orientation = "vertical"

    n, bins, patches = ax2.hist(
        data_robust.ravel(),
        bins=bins,
        density=True,
        orientation=hist_orientation,
        log=_log,
    )

    if not showticks:
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

    ax2.set_frame_on(False)

    if cmap is None:
        cm = colormaps.get_cmap(None)
    else:
        if cmap in _CMAP_QUAL.keys():
            cm = _CMAP_QUAL.get(cmap).mpl_colormap
        elif cmap in _CMAP_EXTRA.keys():
            cm = _CMAP_EXTRA.get(cmap)
        else:
            cm = colormaps.get_cmap(cmap)

    bin_centers = bins[:-1] + bins[1:]

    if cbar_log is True:
        bin_centers = np.log(bin_centers)

    col = bin_centers - np.min(bin_centers)
    col /= np.max(col)

    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))

    if cax is not None:
        extra_pad.fixed_size = _cbar_overflow_inches(f, cax, orientation)
        f.canvas.draw()

    return f


def _cbar_overflow_inches(f, cax, orientation):
    f.canvas.draw()
    renderer = f.canvas.get_renderer()
    tight = cax.get_tightbbox(renderer)
    if tight is None:
        return 0.0
    bbox = cax.get_window_extent(renderer)
    dpi = renderer.points_to_pixels(72)
    if not dpi:
        return 0.0
    if orientation == "vertical":
        overflow = max(0.0, tight.x1 - bbox.x1)
    else:
        overflow = max(0.0, bbox.y0 - tight.y0)
    return overflow / dpi
