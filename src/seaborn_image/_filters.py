import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as ss
from scipy.fftpack import fftn, fftshift
from skimage.filters import difference_of_gaussians, window

from ._general import imgplot

__all__ = ["filterplot", "fftplot", "implemented_filters"]

implemented_filters = {
    "sobel": ndi.sobel,
    "gaussian": ndi.gaussian_filter,
    "median": ndi.median_filter,
    "max": ndi.maximum_filter,
    "diff_of_gaussians": difference_of_gaussians,
    "gaussian_gradient_magnitude": ndi.gaussian_gradient_magnitude,
    "gaussian_laplace": ndi.gaussian_laplace,
    "laplace": ndi.laplace,
    "min": ndi.minimum_filter,
    "percentile": ndi.percentile_filter,
    "prewitt": ndi.prewitt,
    "rank": ndi.rank_filter,
    "uniform": ndi.uniform_filter,
}


def filterplot(
    data,
    filt="gaussian",
    *,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    robust=False,
    perc=(2, 98),
    dx=None,
    units=None,
    dimension=None,
    describe=False,
    cbar=True,
    orientation="v",
    cbar_label=None,
    cbar_ticks=None,
    showticks=False,
    despine=True,
    **kwargs,
):
    """
    Apply N-dimensional filters and plot the filterd data as 2-D image with options to
    add scalebar, colorbar, titles and configure similar to `imgplot`

    Parameters
    ----------
    data : array-like
        Image data. Supported array shapes are all `matplotlib.pyplot.imshow` array shapes
    filt : str or callable, optional
        Filter name or function to be applied.
        Filter name can be a string from `seaborn_image.implemented_filters` or a callable
        filter. Defaults to "gaussian".
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
    robust : bool, optional
        If True and vmin or vmax are None, colormap range is calculated
        based on the percentiles defined in `perc` parameter, by default False
    perc : tuple or list, optional
        If `robust` is True, colormap range is calculated based
        on the percentiles specified instead of the extremes, by default (2, 98) -
        2nd and 98th percentiles for min and max values
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
        Brief statistical description of the data, by default True
    cbar : bool, optional
        Specify if a colorbar is to be added to the image, by default True.
        If `data` is RGB image, cbar is False
    orientation : str, optional
        Specify the orientaion of colorbar, by default "v".
        Options include :
            - 'h' or 'horizontal' for a horizontal colorbar to the bottom of the image.
            - 'v' or 'vertical' for a vertical colorbar to the right of the image.
    cbar_label : str, optional
        Colorbar label, by default None
    cbar_ticks : list, optional
        List of colorbar ticks, by default None
    showticks : bool, optional
        Show image x-y axis ticks, by default False
    despine : bool, optional
        Remove axes spines from image axes as well as colorbar axes, by default True
    **kwargs : optional
        Any additional parameters to be passed to the specific `filt` chosen.
        For instance, "sigma" or "size" or "mode" etc.

    Returns
    -------
        (tuple): tuple containing:

            (`matplotlib.axes.Axes`): Matplotlib axes where the image is drawn.
            (`matplotlib.axes.Axes`): Colorbar axes
            (`numpy.array`): Filtered image data

    Raises
    ------
        TypeError
            if `filt` is not a string type or callable function
        NotImplementedError
            if a `filt` that is not implemented is specified
        TypeError
            if `describe` is not a `bool`

    Examples
    --------

    Use default gaussian filter

    .. plot::
        :context: close-figs

        >>> import seaborn_image as isns
        >>> img = isns.load_image("polymer")
        >>> isns.filterplot(img, sigma=3)

    Specify an image filter with specific parameters

    .. plot::
        :context: close-figs

        >>> isns.filterplot(img, "percentile", percentile=35, size=10)

    `filter` can also be a function

    .. plot::
        :context: close-figs

        >>> import scipy.ndimage as ndi
        >>> isns.filterplot(img, ndi.gaussian_filter, sigma=2.5)

    Specify other image parameters for visualization along with the filter

    .. plot::
        :context: close-figs

        >>> isns.filterplot(img, "sobel", dx=15, units="nm", cmap="acton")
    """

    if not isinstance(describe, bool):
        raise TypeError("describe must be a bool - 'True' or 'False")

    # check if the filt is implemented in seaborn-image
    if isinstance(filt, str) and filt not in implemented_filters.keys():
        raise NotImplementedError(
            f"'{filt}' filt is not implemented. Following are implented: {implemented_filters.keys()}"
        )

    else:
        if isinstance(filt, str) and filt in implemented_filters.keys():
            filt_func = implemented_filters[filt]

        elif callable(filt):
            filt_func = filt

        else:
            raise TypeError("'filt' must either be a string or a function")

        func_kwargs = {}
        func_kwargs.update(**kwargs)

        filtered_data = filt_func(data, **func_kwargs)

    # finally, plot the filtered image
    f, ax, cax = imgplot(
        filtered_data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        robust=robust,
        perc=perc,
        dx=dx,
        units=units,
        dimension=dimension,
        describe=False,
        cbar=cbar,
        orientation=orientation,
        cbar_label=cbar_label,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        despine=despine,
    )

    # Provide basic statistical results
    if describe:  # TODO move all stats to separate file
        result_1 = ss.describe(filtered_data.flatten())
        print("Original Image")
        print(f"No. of Obs. : {result_1.nobs}")
        print(f"Min. Value : {result_1.minmax[0]}")
        print(f"Max. Value : {result_1.minmax[1]}")
        print(f"Mean : {result_1.mean}")
        print(f"Variance : {result_1.variance}")
        print(f"Skewness : {result_1.skewness}")

    return ax, cax, filtered_data


def fftplot(
    data,
    *,
    ax=None,
    cmap=None,
    cbar=True,
    cbar_label=None,
    cbar_ticks=None,
    showticks=False,
):

    if cmap is None:
        cmap = "sunset-dark"

    # window image to improve fft
    w_data = data * window(
        "hann", data.shape
    )  # TODO provide user option for choosing window

    # perform fft
    data_f_mag = fftshift(np.abs(fftn(w_data)))

    f, ax, cax = imgplot(
        np.log(data_f_mag),
        ax=ax,
        cmap=cmap,
        cbar=cbar,
        cbar_ticks=cbar_ticks,
        showticks=showticks,
        describe=False,
    )

    return ax, cax
